"""
Generate task commands for ablation experiments from ablations.yaml.

Usage:
  python tools/gen_ablation_cmds.py              # all experiments
  python tools/gen_ablation_cmds.py chain        # chain only
  python tools/gen_ablation_cmds.py isolated     # isolated only
  python tools/gen_ablation_cmds.py 03_rope      # single experiment by name
"""

import json
import sys
from pathlib import Path
import yaml

EXPERIMENTS_BASE = "./experiments/ablation"
YAML_PATH = Path(__file__).parent / "ablations.yaml"


def load_configs() -> list[tuple[str, str, dict]]:
    """Return list of (name, desc, full_config) for all experiments."""
    data = yaml.safe_load(YAML_PATH.read_text())
    base = dict(data["base"])
    results = []

    # Chain: apply deltas cumulatively
    current = dict(base)
    for entry in data["chain"]:
        if "delta" in entry:
            current = {**current, **entry["delta"]}
        results.append((entry["name"], entry.get("desc", ""), dict(current)))

    # Isolated: apply each delta on top of base_ref resolved config
    iso_section = data.get("isolated", {})
    base_ref = iso_section.get("base_ref")
    iso_base = dict(current)  # default: last chain step
    if base_ref:
        for name, _, cfg in results:
            if name == base_ref:
                iso_base = dict(cfg)
                break

    for entry in iso_section.get("experiments", []):
        cfg = {**iso_base, **entry.get("delta", {})}
        results.append((entry["name"], entry.get("desc", ""), cfg))

    return results


def make_cmd(name: str, cfg: dict) -> str:
    payload = {**cfg, "experiments_dir": f"{EXPERIMENTS_BASE}/{name}"}
    return f"task train-remote -- '{json.dumps(payload)}'"


def main():
    filter_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    configs = load_configs()

    if filter_arg == "chain":
        yaml_data = yaml.safe_load(YAML_PATH.read_text())
        chain_names = {e["name"] for e in yaml_data["chain"]}
        configs = [(n, d, c) for n, d, c in configs if n in chain_names]
    elif filter_arg == "isolated":
        yaml_data = yaml.safe_load(YAML_PATH.read_text())
        chain_names = {e["name"] for e in yaml_data["chain"]}
        configs = [(n, d, c) for n, d, c in configs if n not in chain_names]
    elif filter_arg != "all":
        configs = [(n, d, c) for n, d, c in configs if n == filter_arg]
        if not configs:
            print(f"unknown experiment: {filter_arg!r}", file=sys.stderr)
            sys.exit(1)

    for name, desc, cfg in configs:
        if desc:
            print(f"# {name}: {desc}")
        else:
            print(f"# {name}")
        print(make_cmd(name, cfg))
        print()


if __name__ == "__main__":
    main()
