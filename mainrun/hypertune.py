"""
Hyperparameter tuning framework with Successive Halving.

Two modes:
  Plan mode:  run a fixed list of named experiments (each full 7 epochs)
  Sweep mode: grid search with Successive Halving (round 1 = short proxy, round 2 = full)

Usage:
  python hypertune.py plan              # run EXPERIMENT_PLAN
  python hypertune.py plan <name>       # run a single named experiment
  python hypertune.py sweep             # run SEARCH_SPACE grid with SH
  python hypertune.py sweep --dry-run   # print candidates without running
  python hypertune.py optuna [N]        # TPE search, N trials (default 60, resumable)
  python hypertune.py optuna --dry-run  # preview sampled params without running
"""
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment Plan — named experiments we want to run (full 7 epochs each)
# Current best: 1.213 (muon_uniform, no_rezero, token_anchor, tied, decay_frac=0.30)
# ---------------------------------------------------------------------------

EXPERIMENT_PLAN = [
    # (name, description, overrides)

    # --- Regularization / Training ---
    ("nanochat_pkg",       "norm_emb + untied lm_head",     {"norm_emb": True, "tie_weights": False}),
    ("dropout_005",        "dropout=0.05",                  {"dropout": 0.05}),
    ("warmup_10",          "warmup_frac=0.10",              {"warmup_frac": 0.10}),
    ("gqa_2",              "n_kv_heads=2 (GQA)",            {"n_kv_heads": 2}),
    ("token_anchor_off",   "no token anchor (ablation)",    {"use_token_anchor": False}),

    # --- Architecture: iso-param shape sweep (~20-25M params, head_dim=64) ---
    ("arch_256_deep",      "d=256 n=32 head_dim=64",        {"d_model": 256, "n_layer": 32, "n_q_head": 4}),
    ("arch_256_mid",       "d=256 n=24 head_dim=32",        {"d_model": 256, "n_layer": 24, "n_q_head": 8}),
    ("arch_256_wide_head", "d=256 n=28 head_dim=32",        {"d_model": 256, "n_layer": 28, "n_q_head": 8}),
    ("arch_384_mid",       "d=384 n=11 (medium)",           {"d_model": 384, "n_layer": 11, "n_q_head": 6}),
    ("arch_768_shallow",   "d=768 n=3 (wide+shallow)",      {"d_model": 768, "n_layer": 3,  "n_q_head": 12}),
]

# ---------------------------------------------------------------------------
# Sweep Search Space — grid search with Successive Halving
# ---------------------------------------------------------------------------

ROUND1_EPOCHS = 3
FULL_EPOCHS   = 7
KEEP_FRAC     = 0.5

SEARCH_SPACE: dict[str, list] = {
    "block_size":  [64, 128, 256, 512],
    "batch_size":  [128, 256],
    "muon_lr":     [0.015, 0.02, 0.03, 0.04],
    "decay_frac":  [0.50, 0.60, 0.70],
    "warmup_frac": [0.05, 0.10],
}

# Architecture sweep candidates — head_dim=64 throughout (n_q_head = d_model // 64)

# Round 1: full 15M-27M grid (d=256/384/512/768)
# Results: d=384 dominated. Best: d=384 n=12 (1.2056), d=256/d=768 eliminated.
ARCH_CANDIDATES_V1: list[dict] = [
    # d=256
    _arch_overrides(256, 20, 4),  # ~15.5M
    _arch_overrides(256, 24, 4),  # ~18.2M
    _arch_overrides(256, 28, 4),  # ~20.9M
    _arch_overrides(256, 32, 4),  # ~23.6M
    _arch_overrides(256, 36, 4),  # ~26.2M
    # d=384
    _arch_overrides(384,  8, 6),  # ~15.2M
    _arch_overrides(384, 10, 6),  # ~18.2M
    _arch_overrides(384, 12, 6),  # ~21.2M  ← best
    _arch_overrides(384, 14, 6),  # ~24.2M
    # d=512
    _arch_overrides(512,  5, 8),  # ~17.6M
    _arch_overrides(512,  6, 8),  # ~20.3M
    _arch_overrides(512,  7, 8),  # ~23.0M
    _arch_overrides(512,  8, 8),  # ~25.7M
    # d=768
    _arch_overrides(768,  2, 12), # ~18.4M
    _arch_overrides(768,  3, 12), # ~24.4M
]

# Round 2: zoom in on d=384 + explore d=448
ARCH_CANDIDATES_V2: list[dict] = [
    # d=384: push depth further
    _arch_overrides(384, 12, 6),  # ~21.2M  ← current best
    _arch_overrides(384, 14, 6),  # ~24.2M
    _arch_overrides(384, 16, 6),  # ~26.6M
    # d=448: untested, between 384 and 512
    _arch_overrides(448,  8, 7),  # ~20.2M
    _arch_overrides(448,  9, 7),  # ~22.2M
    _arch_overrides(448, 10, 7),  # ~24.3M
]

ARCH_CANDIDATES = ARCH_CANDIDATES_V2


def _arch_overrides(d_model: int, n_layer: int, n_q_head: int, **rest) -> dict:
    """Convert n_layer int → layer_pattern string for train_hybrid.py."""
    return {"d_model": d_model, "layer_pattern": "A" * n_layer, "n_q_head": n_q_head, **rest}


def estimate_params_M(d_model: int, n_layer: int, n_q_head: int,
                      vocab_size: int = 8192, n_kv_heads: int = 1) -> float:
    head_dim = d_model // n_q_head
    attn = d_model * d_model                          # q_proj
    attn += d_model * head_dim * n_kv_heads * 2       # kv_proj (k+v)
    attn += d_model * d_model                         # out proj
    mlp  = d_model * (4 * d_model) * 2               # up + down
    norm = d_model * 2                                # 2x RMSNorm per layer
    per_layer = attn + mlp + norm
    total = per_layer * n_layer + vocab_size * d_model + d_model  # +emb +ln_f
    return total / 1e6

# ---------------------------------------------------------------------------
# Experiments dir (separate from manual runs)
# ---------------------------------------------------------------------------

PLAN_EXPERIMENTS_DIR  = "./experiments/plan"
SWEEP_EXPERIMENTS_DIR = "./experiments/sweep"

# ---------------------------------------------------------------------------
# Core: run a single experiment
# ---------------------------------------------------------------------------

def run(overrides: dict, epochs: int, experiments_dir: str) -> float:
    payload = {**overrides, "epochs": epochs, "experiments_dir": experiments_dir, "evals_per_epoch": 1}
    cmd = [sys.executable, "train_hybrid.py", json.dumps(payload)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        print(f"\n  [FAILED] {overrides}")
        return float("inf")
    return _parse_val_loss("".join(lines))


def _parse_val_loss(stdout: str) -> float:
    for line in reversed(stdout.splitlines()):
        if "validation_step: loss=" in line:
            return float(line.split("loss=")[1].split()[0])
    return float("inf")


def _log(log_path: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# Plan mode
# ---------------------------------------------------------------------------

def run_plan(target_name: str | None = None):
    experiments = EXPERIMENT_PLAN
    if target_name:
        experiments = [(n, d, o) for n, d, o in EXPERIMENT_PLAN if n == target_name]
        if not experiments:
            print(f"Unknown experiment: {target_name!r}")
            print(f"Available: {[n for n, *_ in EXPERIMENT_PLAN]}")
            sys.exit(1)

    log_path = f"{PLAN_EXPERIMENTS_DIR}/progress.log"

    print(f"\n{'='*60}")
    print(f"Experiment Plan: {len(experiments)} run(s)")
    print(f"{'='*60}\n")

    results = []
    for name, desc, overrides in experiments:
        print(f"  [{name}] {desc}")
        print(f"    overrides: {overrides}")
        loss = run(overrides, FULL_EPOCHS, f"{PLAN_EXPERIMENTS_DIR}/{name}")
        results.append((loss, name, desc, overrides))
        status = f"{loss:.4f}" if loss != float("inf") else "FAILED"
        _log(log_path, f"{name:<20} val_loss={status}  {overrides}")
        print(f"    → val_loss = {loss:.4f}\n")

    results.sort(key=lambda x: x[0])
    print(f"\n{'='*60}")
    print("Plan results (best first):")
    for rank, (loss, name, desc, overrides) in enumerate(results, 1):
        print(f"  [{rank}] {loss:.4f}  {name:<20} {desc}")
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Sweep mode — Successive Halving
# ---------------------------------------------------------------------------

def grid_candidates(space: dict[str, list]) -> list[dict]:
    keys   = list(space.keys())
    values = list(space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_sweep(dry_run: bool = False):
    candidates = grid_candidates(SEARCH_SPACE)
    keep = max(1, int(len(candidates) * KEEP_FRAC))
    log_path = f"{SWEEP_EXPERIMENTS_DIR}/progress.log"

    print(f"\n{'='*60}")
    print(f"Successive Halving: {len(candidates)} candidates → keep {keep} → full run")
    print(f"Round 1 epochs: {ROUND1_EPOCHS}  |  Full epochs: {FULL_EPOCHS}")
    print(f"{'='*60}\n")

    if dry_run:
        for i, c in enumerate(candidates):
            print(f"  [{i+1:2d}] {c}")
        return

    # Round 1
    print(f"── Round 1 ({ROUND1_EPOCHS} epochs) ──\n")
    r1 = []
    for i, overrides in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] {overrides} ", end="", flush=True)
        loss = run(overrides, ROUND1_EPOCHS, f"{SWEEP_EXPERIMENTS_DIR}/round1")
        print(f"→ {loss:.4f}")
        _log(log_path, f"r1 [{i+1:2d}/{len(candidates)}] {loss:.4f}  {overrides}")
        r1.append((loss, overrides))

    r1.sort(key=lambda x: x[0])
    survivors = [o for _, o in r1[:keep]]

    keys = list(SEARCH_SPACE.keys())
    header = "  " + "  ".join(f"{k:<12}" for k in ["rank", "val_loss"] + keys)
    print(f"\nRound 1 ranking:")
    print(header)
    for rank, (loss, o) in enumerate(r1, 1):
        marker = "✓" if o in survivors else "✗"
        vals = "  ".join(f"{str(o[k]):<12}" for k in keys)
        print(f"  {marker}[{rank:<3}] {loss:.4f}      {vals}")

    # Round 2
    print(f"\n── Round 2 ({FULL_EPOCHS} epochs, {keep} survivors) ──\n")
    r2 = []
    for i, overrides in enumerate(survivors):
        print(f"  [{i+1}/{keep}] {overrides} ", end="", flush=True)
        loss = run(overrides, FULL_EPOCHS, f"{SWEEP_EXPERIMENTS_DIR}/round2")
        print(f"→ {loss:.4f}")
        _log(log_path, f"r2 [{i+1:2d}/{keep}] {loss:.4f}  {overrides}")
        r2.append((loss, overrides))

    r2.sort(key=lambda x: x[0])
    print(f"\n{'='*60}")
    print("Sweep results (best first):")
    print(header)
    for rank, (loss, o) in enumerate(r2, 1):
        vals = "  ".join(f"{str(o[k]):<12}" for k in keys)
        print(f"  [{rank:<3}] {loss:.4f}      {vals}")
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Arch sweep mode — Successive Halving over ARCH_CANDIDATES
# ---------------------------------------------------------------------------

def run_arch_sweep(dry_run: bool = False):
    candidates = ARCH_CANDIDATES
    keep = max(1, int(len(candidates) * KEEP_FRAC))
    log_path = f"{SWEEP_EXPERIMENTS_DIR}/arch_progress.log"
    expt_dir = f"{SWEEP_EXPERIMENTS_DIR}/arch"

    print(f"\n{'='*60}")
    print(f"Arch Sweep: {len(candidates)} candidates → keep {keep} → full run")
    print(f"Round 1 epochs: {ROUND1_EPOCHS}  |  Full epochs: {FULL_EPOCHS}")
    print(f"{'='*60}\n")

    if dry_run:
        for i, c in enumerate(candidates):
            print(f"  [{i+1:2d}] {c}")
        return

    results = []
    for i, overrides in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] {overrides} ", end="", flush=True)
        loss = run(overrides, FULL_EPOCHS, f"{expt_dir}/full")
        print(f"→ {loss:.4f}")
        _log(log_path, f"[{i+1:2d}/{len(candidates)}] {loss:.4f}  {overrides}")
        results.append((loss, overrides))

    results.sort(key=lambda x: x[0])
    print(f"\n{'='*60}")
    print("Arch sweep results (best first):")
    print(f"  {'rank':<6} {'val_loss':<10} {'d_model':<9} {'n_layer':<9} {'heads':<7} {'params_M'}")
    for rank, (loss, o) in enumerate(results, 1):
        n_layer = len(o['layer_pattern'])
        params = estimate_params_M(o['d_model'], n_layer, o['n_q_head'])
        print(f"  [{rank}]    {loss:.4f}    d={o['d_model']:<6} n={n_layer:<6} h={o['n_q_head']:<4}  {params:.1f}M")
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Optuna mode — TPE sampler + MedianPruner
# ---------------------------------------------------------------------------

OPTUNA_N_TRIALS = 60

def _run_optuna_trial(overrides: dict, epochs: int, experiments_dir: str, trial) -> float:
    """Like run(), but reports intermediate val_losses to Optuna for pruning."""
    import optuna
    payload = {**overrides, "epochs": epochs, "experiments_dir": experiments_dir, "evals_per_epoch": 3}
    cmd = [sys.executable, "train_hybrid.py", json.dumps(payload)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    lines = []
    step = 0
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        if "validation_step: loss=" in line:
            try:
                val_loss = float(line.split("loss=")[1].split()[0])
                trial.report(val_loss, step)
                step += 1
                if trial.should_prune():
                    proc.terminate()
                    proc.wait()
                    raise optuna.TrialPruned()
            except (ValueError, IndexError):
                pass
    proc.wait()
    if proc.returncode != 0:
        return float("inf")
    return _parse_val_loss("".join(lines))


def run_optuna(n_trials: int = OPTUNA_N_TRIALS, dry_run: bool = False):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = Path(SWEEP_EXPERIMENTS_DIR) / "optuna.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="hypertune",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1337),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = {
            "block_size":  trial.suggest_categorical("block_size",  [64, 128, 256, 512]),
            "batch_size":  trial.suggest_categorical("batch_size",  [128, 256]),
            "muon_lr":     trial.suggest_float("muon_lr",  0.01, 0.05, log=True),
            "decay_frac":  trial.suggest_float("decay_frac", 0.40, 0.75),
            "warmup_frac": trial.suggest_categorical("warmup_frac", [0.05, 0.10]),
            "min_lr_frac": trial.suggest_float("min_lr_frac", 0.01, 0.10, log=True),
        }
        if dry_run:
            print(f"  [dry-run] trial {trial.number}: {overrides}")
            return 0.0
        _log(f"{SWEEP_EXPERIMENTS_DIR}/optuna_progress.log",
             f"trial {trial.number:3d} start  {overrides}")
        loss = _run_optuna_trial(overrides, ROUND1_EPOCHS, f"{SWEEP_EXPERIMENTS_DIR}/optuna", trial)
        _log(f"{SWEEP_EXPERIMENTS_DIR}/optuna_progress.log",
             f"trial {trial.number:3d} end    loss={loss:.4f}  {overrides}")
        return loss

    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remaining = max(0, n_trials - completed)
    print(f"\n{'='*60}")
    print(f"Optuna TPE search: {n_trials} trials ({completed} done, {remaining} remaining)")
    print(f"DB: {db_path}  (resumable)")
    print(f"{'='*60}\n")

    if dry_run:
        for i in range(min(5, n_trials)):
            study.ask()  # just to show sampled params
        return

    study.optimize(objective, n_trials=remaining)

    print(f"\n{'='*60}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params:   {study.best_params}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "plan":
        target = args[1] if len(args) > 1 else None
        run_plan(target)
    elif args[0] == "sweep":
        run_sweep(dry_run="--dry-run" in args)
    elif args[0] == "arch_sweep":
        run_arch_sweep(dry_run="--dry-run" in args)
    elif args[0] == "optuna":
        n = int(args[1]) if len(args) > 1 and args[1].isdigit() else OPTUNA_N_TRIALS
        run_optuna(n_trials=n, dry_run="--dry-run" in args)
    else:
        print(__doc__)
        sys.exit(1)
