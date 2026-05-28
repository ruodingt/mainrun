"""
Scan ablation .log files, find the run with the lowest val_loss,
and write its quality-affecting hparams to best_config.yaml.

Log location: mainrun/logs/ablation/**/*.log
Each .log is one run: one "hyperparameters_configured" line + N "validation_step" lines.

Usage (from repo root mainrun/):
    python tools/fetch_best_config.py [--dry-run] [--logs-dir PATH]

Writes: best_config.yaml  (overwritten each time)
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
DEFAULT_LOGS = ROOT / "mainrun" / "logs" / "ablation"
OUT_FILE = ROOT / "mainrun" / "configs" / "best_config.yaml"

# Keys present in hyperparameters_configured that are NOT quality-affecting.
_RUNTIME_KEYS = {
    "device", "use_fa2", "use_fused_ce", "use_compile", "use_bf16",
    "evals_per_epoch", "experiments_dir", "log_file", "use_tqdm", "save_weights",
}
_FIXED_KEYS = {"epochs", "seed", "num_titles", "val_frac"}
_META_KEYS  = {"event", "timestamp", "n_layer"}
_EXCLUDE    = _RUNTIME_KEYS | _FIXED_KEYS | _META_KEYS

# Mamba-specific keys — only keep when layer_pattern contains 'M'
_MAMBA_KEYS = {"expand", "d_head", "d_state", "n_groups", "d_conv", "chunk_len"}

# Section order for output YAML
_ARCH_KEYS  = ["layer_pattern", "d_model", "vocab_size", "dropout", "norm",
               "pos_emb", "mlp_act", "mlp_expand", "tie_weights",
               "use_token_anchor", "use_resid_scale", "use_rezero",
               "norm_emb", "logit_softcap", "weight_init"]
_ATTN_KEYS  = ["n_q_head", "n_kv_heads", "separate_kv",
               "use_value_residual", "use_value_residual_x0", "use_value_carry",
               "ve_gate_channels"]
_OPT_KEYS   = ["optimizer_type", "muon_lr", "muon_attn_only", "mlp_lr",
               "adamw_lr", "emb_lr", "scalar_lr", "adamw_wd",
               "sgd_lr", "sgd_wd", "spectral_clip"]
_SCHED_KEYS = ["lr_schedule", "warmup_frac", "decay_frac", "min_lr_frac"]
_TRAIN_KEYS = ["batch_size", "block_size", "clip_norm_mode"]


def parse_log(path: Path) -> tuple[dict, float] | None:
    """
    Parse one log file. Returns (hparams_flat, best_val_loss) or None if incomplete.
    Takes the minimum val_loss seen (= best checkpoint reached during training).
    """
    hparams = None
    best_loss = float("inf")

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = rec.get("event")
        if event == "hyperparameters_configured":
            hparams = rec
        elif event == "validation_step" and hparams is not None:
            loss = rec.get("loss")
            if loss is not None and loss < best_loss:
                best_loss = loss

    if hparams is None or best_loss == float("inf"):
        return None
    return hparams, best_loss


def scan_best(logs_dir: Path) -> tuple[dict, float, str]:
    """Return (flat_hparams, val_loss, log_stem) for the run with lowest val_loss."""
    best_loss = float("inf")
    best_hparams: dict = {}
    best_name = ""

    log_files = sorted(logs_dir.rglob("*.log"))
    if not log_files:
        print(f"No .log files found under {logs_dir}", file=sys.stderr)
        sys.exit(1)

    for log_path in log_files:
        result = parse_log(log_path)
        if result is None:
            continue
        hparams, val_loss = result
        if val_loss < best_loss:
            best_loss = val_loss
            best_hparams = hparams
            best_name = log_path.stem

    if not best_hparams:
        print("No completed runs found.", file=sys.stderr)
        sys.exit(1)

    return best_hparams, best_loss, best_name


def build_config(raw: dict) -> dict:
    """Strip non-quality keys from a flat hyperparameters_configured dict."""
    hparams = {k: v for k, v in raw.items() if k not in _EXCLUDE}
    layer_pattern = hparams.get("layer_pattern", "")
    if "M" not in layer_pattern:
        for k in _MAMBA_KEYS:
            hparams.pop(k, None)
    if "E" not in layer_pattern:
        hparams.pop("ve_gate_channels", None)
    return hparams


def render_yaml(hparams: dict, val_loss: float, exp_name: str) -> str:
    def section(keys):
        return {k: hparams[k] for k in keys if k in hparams}

    covered = set(_ARCH_KEYS + _ATTN_KEYS + _OPT_KEYS + _SCHED_KEYS + _TRAIN_KEYS)
    extra = {k: v for k, v in hparams.items() if k not in covered}

    lines = [
        f"# Best config from logs: {exp_name}",
        f"# val_loss: {val_loss:.4f}",
        "",
        "# --- Architecture ---",
        *yaml.dump(section(_ARCH_KEYS), default_flow_style=False).splitlines(),
        "",
        "# --- Attention ---",
        *yaml.dump(section(_ATTN_KEYS), default_flow_style=False).splitlines(),
        "",
        "# --- Optimizer ---",
        *yaml.dump(section(_OPT_KEYS), default_flow_style=False).splitlines(),
        "",
        "# --- LR Schedule ---",
        *yaml.dump(section(_SCHED_KEYS), default_flow_style=False).splitlines(),
        "",
        "# --- Training ---",
        *yaml.dump(section(_TRAIN_KEYS), default_flow_style=False).splitlines(),
    ]
    if extra:
        lines += ["", "# --- Other ---",
                  *yaml.dump(extra, default_flow_style=False).splitlines()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config instead of writing file")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS,
                        help=f"Root directory to search for .log files (default: {DEFAULT_LOGS})")
    cli = parser.parse_args()

    raw_hparams, val_loss, exp_name = scan_best(cli.logs_dir)
    hparams = build_config(raw_hparams)
    content = render_yaml(hparams, val_loss, exp_name)

    if cli.dry_run:
        print(f"[dry-run] Would write to {OUT_FILE}:\n")
        print(content)
    else:
        OUT_FILE.write_text(content)
        print(f"Written: {OUT_FILE}")
        print(f"  best: {exp_name}  val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
