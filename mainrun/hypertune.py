"""
Optuna TPE hyperparameter search for train_hybrid.py.

Base config: best arch from ablation (28L×256d×16k, swiglu, tie, rope,
             muon+adamw, WSD, gpt2 init, token_anchor).

Search space: LR and schedule params only.

Usage:
  python hypertune.py [N]          # N trials (default 60, resumable)
  python hypertune.py --dry-run    # preview sampled params without running
"""
import json
import sys
import time
from pathlib import Path
import optuna

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROUND1_EPOCHS = 3
OPTUNA_N_TRIALS = 60
EXPT_DIR = "./experiments/hypertune"

# Fixed overrides applied to every trial — the best config from ablation study.
# group5-9: 28L×256d×16k + token_anchor = 1.1696 (best overall)
BASE_OVERRIDES: dict = {
    # Architecture (groups 5-7: deep-narrow wins; 28L×256d optimal within 40M)
    "layer_pattern":    "A" * 28,
    "vocab_size":       16000,
    "d_model":          256,
    "n_q_head":         4,
    "n_kv_heads":       4,
    # Arch tricks (group8-9: anchor helps at depth, softcap/rezero/value_res hurt)
    "use_token_anchor": True,
    "logit_softcap":    0.0,
    # Optimizer/init findings (groups 1-4)
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

def _run_trial(overrides: dict, trial, log_file: str | None) -> float:
    """Run train_hybrid.py, report intermediate val_losses to Optuna for pruning."""

    payload = {
        **BASE_OVERRIDES,
        **overrides,
        "epochs": ROUND1_EPOCHS,
        "experiments_dir": EXPT_DIR,
        "evals_per_epoch": 3,
    }
    if log_file:
        payload["log_file"] = log_file

    import subprocess
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
    loss = _run_trial(
        overrides, trial,
        log_file=f"./logs/hypertune_{trial.number:03d}.log",
    )
    _log(f"trial {trial.number:3d} end    loss={loss:.4f}  {overrides}")
    return loss

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    n_trials = next((int(a) for a in args if a.isdigit()), OPTUNA_N_TRIALS)

    db_path = Path(EXPT_DIR) / "optuna.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="hypertune",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1337),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
    )

    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remaining = max(0, n_trials - completed)

    print(f"\n{'='*60}")
    print(f"Optuna TPE search: {n_trials} trials ({completed} done, {remaining} remaining)")
    print(f"Base: 28L×256d×vocab16k, swiglu, tie, rope, muon+wsd, gpt2_init, token_anchor")
    print(f"Search: muon_lr, adamw_lr, emb_lr, decay_frac, warmup_frac, min_lr_frac")
    print(f"Proxy: {ROUND1_EPOCHS} epochs  |  DB: {db_path}")
    print(f"{'='*60}\n")

    if dry_run:
        for _ in range(min(5, n_trials)):
            trial = study.ask()
            _objective(trial, dry_run=True)
        return

    study.optimize(lambda t: _objective(t, dry_run=False), n_trials=remaining)

    print(f"\n{'='*60}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params:   {study.best_params}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
