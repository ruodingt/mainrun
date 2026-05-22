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
    "batch_size":  [128, 256, 512],
    "muon_lr":     [0.015, 0.02, 0.03, 0.04],
    "decay_frac":  [0.25, 0.30, 0.40, 0.50],
    "warmup_frac": [0.05, 0.10],
}

# Architecture sweep candidates — head_dim=64 throughout (n_q_head = d_model // 64)

# Round 1: full 15M-27M grid (d=256/384/512/768)
# Results: d=384 dominated. Best: d=384 n=12 (1.2056), d=256/d=768 eliminated.
ARCH_CANDIDATES_V1: list[dict] = [
    # d=256
    {"d_model": 256, "n_layer": 20, "n_q_head": 4},  # ~15.5M
    {"d_model": 256, "n_layer": 24, "n_q_head": 4},  # ~18.2M
    {"d_model": 256, "n_layer": 28, "n_q_head": 4},  # ~20.9M
    {"d_model": 256, "n_layer": 32, "n_q_head": 4},  # ~23.6M
    {"d_model": 256, "n_layer": 36, "n_q_head": 4},  # ~26.2M
    # d=384
    {"d_model": 384, "n_layer":  8, "n_q_head": 6},  # ~15.2M
    {"d_model": 384, "n_layer": 10, "n_q_head": 6},  # ~18.2M
    {"d_model": 384, "n_layer": 12, "n_q_head": 6},  # ~21.2M  ← best
    {"d_model": 384, "n_layer": 14, "n_q_head": 6},  # ~24.2M
    # d=512
    {"d_model": 512, "n_layer":  5, "n_q_head": 8},  # ~17.6M
    {"d_model": 512, "n_layer":  6, "n_q_head": 8},  # ~20.3M
    {"d_model": 512, "n_layer":  7, "n_q_head": 8},  # ~23.0M
    {"d_model": 512, "n_layer":  8, "n_q_head": 8},  # ~25.7M
    # d=768
    {"d_model": 768, "n_layer":  2, "n_q_head": 12}, # ~18.4M
    {"d_model": 768, "n_layer":  3, "n_q_head": 12}, # ~24.4M
]

# Round 2: zoom in on d=384 + explore d=448
ARCH_CANDIDATES_V2: list[dict] = [
    # d=384: push depth further
    {"d_model": 384, "n_layer": 12, "n_q_head": 6},  # ~21.2M  ← current best
    {"d_model": 384, "n_layer": 14, "n_q_head": 6},  # ~24.2M
    {"d_model": 384, "n_layer": 16, "n_q_head": 6},  # ~26.6M
    # d=448: untested, between 384 and 512
    {"d_model": 448, "n_layer":  8, "n_q_head": 7},  # ~20.2M
    {"d_model": 448, "n_layer":  9, "n_q_head": 7},  # ~22.2M
    {"d_model": 448, "n_layer": 10, "n_q_head": 7},  # ~24.3M
]

ARCH_CANDIDATES = ARCH_CANDIDATES_V2


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
    cmd = [sys.executable, "train.py", json.dumps(payload)]
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
        params = estimate_params_M(o['d_model'], o['n_layer'], o['n_q_head'])
        print(f"  [{rank}]    {loss:.4f}    d={o['d_model']:<6} n={o['n_layer']:<6} h={o['n_q_head']:<4}  {params:.1f}M")
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
    else:
        print(__doc__)
        sys.exit(1)
