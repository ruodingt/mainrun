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
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment Plan — named experiments we want to run (full 7 epochs each)
# Current best: 1.213 (muon_uniform, no_rezero, token_anchor, tied, decay_frac=0.30)
# ---------------------------------------------------------------------------

EXPERIMENT_PLAN = [
    # (name, description, overrides)

    # --- Scheduler ---
    ("decay_40",      "WSD decay 40%",                  {"decay_frac": 0.40}),
    ("decay_50",      "WSD decay 50%",                  {"decay_frac": 0.50}),

    # --- LR tuning ---
    ("muon_lr_01",    "Muon LR 0.01",                   {"muon_lr": 0.01}),
    ("muon_lr_015",   "Muon LR 0.015",                  {"muon_lr": 0.015}),
    ("muon_lr_03",    "Muon LR 0.03",                   {"muon_lr": 0.03}),

    # --- nanochat full package ---
    # norm_emb normalizes embedding before x0 injection → allows token_emb std=0.8
    # tie_weights=False enables lm_head std=0.001 init
    ("nanochat_pkg",  "norm_emb + untied lm_head",      {"norm_emb": True, "tie_weights": False}),

    # --- softcap sensitivity ---
    ("softcap_30",    "Larger softcap = 30",             {"logit_softcap": 30.0}),
    ("softcap_off",   "No softcap",                      {"logit_softcap": 0.0}),

    # --- Architecture ---
    ("wider_model",   "d_model=576 n_layer=6",           {"d_model": 576, "n_q_head": 9}),
    ("deeper_model",  "d_model=448 n_layer=8",           {"d_model": 448, "n_q_head": 7, "n_layer": 8}),
]

# ---------------------------------------------------------------------------
# Sweep Search Space — grid search with Successive Halving
# ---------------------------------------------------------------------------

ROUND1_EPOCHS = 3
FULL_EPOCHS   = 7
KEEP_FRAC     = 0.5

SEARCH_SPACE: dict[str, list] = {
    "muon_lr":    [0.01, 0.015, 0.02, 0.03],
    "decay_frac": [0.30, 0.40, 0.50],
}

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

    print(f"\n{'='*60}")
    print(f"Experiment Plan: {len(experiments)} run(s)")
    print(f"{'='*60}\n")

    results = []
    for name, desc, overrides in experiments:
        print(f"  [{name}] {desc}")
        print(f"    overrides: {overrides}")
        loss = run(overrides, FULL_EPOCHS, f"{PLAN_EXPERIMENTS_DIR}/{name}")
        results.append((loss, name, desc, overrides))
        print(f"    → val_loss = {loss:.4f}\n")

    results.sort(key=lambda x: x[0])
    print(f"\n{'='*60}")
    print("Plan results (best first):")
    for rank, (loss, name, desc, overrides) in enumerate(results, 1):
        print(f"  [{rank}] {loss:.4f}  {name:<16} {desc}")
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
        r1.append((loss, overrides))

    r1.sort(key=lambda x: x[0])
    survivors = [o for _, o in r1[:keep]]

    print(f"\nRound 1 ranking:")
    for rank, (loss, o) in enumerate(r1, 1):
        marker = "✓" if o in survivors else "✗"
        print(f"  {marker} [{rank}] {loss:.4f}  {o}")

    # Round 2
    print(f"\n── Round 2 ({FULL_EPOCHS} epochs, {keep} survivors) ──\n")
    r2 = []
    for i, overrides in enumerate(survivors):
        print(f"  [{i+1}/{keep}] {overrides} ", end="", flush=True)
        loss = run(overrides, FULL_EPOCHS, f"{SWEEP_EXPERIMENTS_DIR}/round2")
        print(f"→ {loss:.4f}")
        r2.append((loss, overrides))

    r2.sort(key=lambda x: x[0])
    print(f"\n{'='*60}")
    print("Sweep results (best first):")
    for rank, (loss, o) in enumerate(r2, 1):
        print(f"  [{rank}] {loss:.4f}  {o}")
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
    else:
        print(__doc__)
        sys.exit(1)
