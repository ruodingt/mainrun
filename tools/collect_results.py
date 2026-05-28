"""
Collect ablation results from local experiments directory.

Walks experiments/ablation/{group}/{exp}/exp*/run*/run.yaml and
produces a JSON file with all results for report generation.

Usage:
  python tools/collect_results.py              # prints JSON to stdout
  python tools/collect_results.py > results.json
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
LOCAL_ABLATION = ROOT / "mainrun" / "experiments" / "ablation"
ABLATIONS_YAML = Path(__file__).parent.parent / "mainrun" / "configs" / "ablations.yaml"


def latest_run_yaml(exp_dir: Path) -> Path | None:
    candidates = sorted(
        exp_dir.glob("exp*/run*/run.yaml"),
        key=lambda p: (p.parent.parent.name, p.parent.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_ablations_base() -> dict:
    """Return the root base hparams from ablations.yaml (flat dict)."""
    return yaml.safe_load(ABLATIONS_YAML.read_text())["base"]


def load_full_yaml(exp_name: str) -> dict:
    """Return the raw parsed run.yaml for a given experiment name."""
    for group_dir in sorted(LOCAL_ABLATION.iterdir()):
        exp_dir = group_dir / exp_name
        if exp_dir.is_dir():
            run_yaml = latest_run_yaml(exp_dir)
            if run_yaml:
                return yaml.safe_load(run_yaml.read_text())
    raise FileNotFoundError(f"No run.yaml found for experiment: {exp_name!r}")


def collect():
    rows = []

    for group_dir in sorted(LOCAL_ABLATION.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name

        for exp_dir in sorted(group_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp = exp_dir.name

            run_yaml = latest_run_yaml(exp_dir)
            if run_yaml is None:
                continue

            data = yaml.safe_load(run_yaml.read_text())
            results = data.get("results", {})
            arch = data.get("arch", {})
            attn = data.get("attention", {})

            n_layer = len(arch.get("layer_pattern", ""))
            d_model = arch.get("d_model")
            params = results.get("total_params")
            tok_s = results.get("avg_tok_s")
            val_loss = results.get("val_loss")

            rows.append({
                "group": group,
                "exp": exp,
                "n_layer": n_layer,
                "d_model": d_model,
                "n_q_head": attn.get("n_q_head"),
                "n_kv_heads": attn.get("n_kv_heads"),
                "vocab_size": arch.get("vocab_size"),
                "params_M": round(params / 1e6, 2) if params else None,
                "tok_s": round(tok_s) if tok_s else None,
                "val_loss": round(val_loss, 4) if val_loss else None,
                # key arch flags
                "mlp_act": arch.get("mlp_act"),
                "tie_weights": arch.get("tie_weights"),
                "use_token_anchor": arch.get("use_token_anchor"),
                "logit_softcap": arch.get("logit_softcap"),
                "use_rezero": arch.get("use_rezero"),
                "use_resid_scale": arch.get("use_resid_scale"),
                "optimizer_type": data.get("optimizer", {}).get("optimizer_type"),
                "lr_schedule": data.get("optimizer", {}).get("lr_schedule"),
                "pos_emb": arch.get("pos_emb"),
                "weight_init": arch.get("weight_init"),
            })

    return rows


if __name__ == "__main__":
    rows = collect()
    json.dump(rows, sys.stdout, indent=2)
    print()
