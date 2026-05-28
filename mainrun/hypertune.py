"""
Optuna TPE hyperparameter search for train_hybrid.py.

Runs full 7-epoch trials — no proxy/full split, no schedule mismatch.
Optuna's MedianPruner kills bad trials early (typically within epoch 1-2).
Best trial results are copied to experiments/hypertune-full/ for sync_ablation.

New study name "hypertune_v2" — old proxy-based results are incompatible.

Usage:
  python hypertune.py [N]       # N trials (default 60, resumable)
  python hypertune.py --dry-run # preview sampled params without running
  python hypertune.py --force   # overwrite hypertune-full/ even if it exists
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import optuna

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCHS          = 7
OPTUNA_N_TRIALS = 40
EXPT_DIR        = "./experiments/hypertune_v2"
FULL_EXPT_DIR   = "./experiments/hypertune_v2_best"

# Best config from ablation (groups 1-9).
BASE_OVERRIDES: dict = {
    "layer_pattern":    "A" * 28,
    "vocab_size":       16000,
    "d_model":          256,
    "norm":             "layernorm",
    "n_q_head":         4,
    "n_kv_heads":       4,
    "use_token_anchor": True,
    "logit_softcap":    0.0,
    "weight_init":      "gpt2",
    "mlp_act":          "swiglu",
    "tie_weights":      True,
    "optimizer_type":   "muon_adamw",
    "pos_emb":          "rope",
    "lr_schedule":      "wsd",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trial_expt_dir(trial_number: int) -> str:
    return f"{EXPT_DIR}/trial_{trial_number:03d}"


def _run_trial(overrides: dict, trial) -> float:
    """Run train_hybrid.py for full EPOCHS, report val_losses for pruning."""
    expt_dir = _trial_expt_dir(trial.number)
    payload = {
        **BASE_OVERRIDES,
        **overrides,
        "epochs":          EPOCHS,
        "evals_per_epoch": 3,
        "experiments_dir": expt_dir,
        "log_file":        f"{expt_dir}/train.log",
    }

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
    for line in reversed(lines):
        if "validation_step: loss=" in line:
            return float(line.split("loss=")[1].split()[0])
    return float("inf")


def _promote_best(study, force: bool):
    """Copy best trial's experiment dir → hypertune-full/ for sync_ablation."""
    full_dir = Path(FULL_EXPT_DIR)
    if not force and list(full_dir.glob("**/results.json")):
        print(f"[hypertune] hypertune-full/ already exists, skipping (--force to overwrite)")
        return

    best = study.best_trial
    src = Path(_trial_expt_dir(best.number))
    if not src.exists():
        print(f"[hypertune] best trial dir not found: {src}")
        return

    if full_dir.exists():
        shutil.rmtree(full_dir)
    shutil.copytree(src, full_dir)
    print(f"[hypertune] best trial #{best.number} (val_loss={best.value:.4f}) → {full_dir}")


def _write_top5(study):
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    top5 = sorted(completed, key=lambda t: t.value)[:5]
    lines = [
        "# Hypertune Top 5 Trials\n",
        f"Base: 28L×256d×vocab16k, layernorm, swiglu, tie, rope, muon+wsd, gpt2_init, token_anchor\n",
        f"Full {EPOCHS} epochs per trial (early-stopped by Optuna pruner)\n\n",
        "| Rank | Trial | val_loss | muon_lr | adamw_lr | emb_lr | decay_frac | warmup_frac | min_lr_frac | dropout |\n",
        "|------|-------|----------|---------|----------|--------|------------|-------------|-------------|----------|\n",
    ]
    for rank, t in enumerate(top5, 1):
        p = t.params
        lines.append(
            f"| {rank} | #{t.number} | {t.value:.4f} "
            f"| {p.get('muon_lr', '-'):.4f} | {p.get('adamw_lr', '-'):.5f} "
            f"| {p.get('emb_lr', '-'):.4f} | {p.get('decay_frac', '-'):.2f} "
            f"| {p.get('warmup_frac', '-')} | {p.get('min_lr_frac', '-'):.4f} "
            f"| {p.get('dropout', '-'):.3f} |\n"
        )
    out = Path(EXPT_DIR) / "top5.md"
    out.write_text("".join(lines))
    print(f"Top 5 written to {out}")


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_path = Path(EXPT_DIR) / "progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _objective(trial, dry_run: bool) -> float:
    overrides = {
        "muon_lr":     trial.suggest_float("muon_lr",     0.005, 0.10,  log=True),
        "adamw_lr":    trial.suggest_float("adamw_lr",    1e-4,  1e-3,  log=True),
        "emb_lr":      trial.suggest_float("emb_lr",      5e-4,  1e-2,  log=True),
        "decay_frac":  trial.suggest_float("decay_frac",  0.30,  0.80),
        "warmup_frac": trial.suggest_categorical("warmup_frac", [0.02, 0.05, 0.10]),
        "min_lr_frac": trial.suggest_float("min_lr_frac", 0.01,  0.10,  log=True),
        "dropout":     trial.suggest_float("dropout",     0.0,   0.20),
    }
    if dry_run:
        print(f"  [dry-run] trial {trial.number}: {overrides}")
        return 0.0

    _log(f"trial {trial.number:3d} start  {overrides}")
    loss = _run_trial(overrides, trial)
    _log(f"trial {trial.number:3d} end    loss={loss:.4f}  {overrides}")
    return loss

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from hparams import Hyperparameters
    Hyperparameters.validate_coverage(BASE_OVERRIDES, context="hypertune BASE_OVERRIDES")

    args = sys.argv[1:]
    dry_run  = "--dry-run" in args
    force    = "--force"   in args
    n_trials = next((int(a) for a in args if a.isdigit()), OPTUNA_N_TRIALS)

    db_path = Path(EXPT_DIR) / "optuna_v2.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="hypertune_v2",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1337),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6),
    )

    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remaining = max(0, n_trials - completed)

    print(f"\n{'='*60}")
    print(f"Optuna TPE search: {n_trials} trials ({completed} done, {remaining} remaining)")
    print(f"Base: 28L×256d×vocab16k, layernorm, swiglu, tie, rope, muon+wsd, gpt2_init, token_anchor")
    print(f"Search: muon_lr, adamw_lr, emb_lr, decay_frac, warmup_frac, min_lr_frac, dropout")
    print(f"Full {EPOCHS} epochs per trial, pruner kills bad trials early")
    print(f"DB: {db_path}")
    print(f"{'='*60}\n")

    if dry_run:
        for _ in range(min(5, n_trials)):
            trial = study.ask()
            _objective(trial, dry_run=True)
        return

    def _progress(study, trial):
        completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
        best = study.best_value if completed > 0 else float("inf")
        _log(f"progress {completed:3d}/{n_trials}  best={best:.4f}")

    study.optimize(lambda t: _objective(t, dry_run=False), n_trials=remaining,
                   callbacks=[_progress])

    print(f"\n{'='*60}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params:   {study.best_params}")
    print(f"{'='*60}\n")

    _write_top5(study)
    _promote_best(study, force=force)


if __name__ == "__main__":
    main()
