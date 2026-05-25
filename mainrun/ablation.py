"""
Ablation study runner. Configs live in tools/ablations.yaml.

Config resolution per experiment: root base → group_base → delta.

Usage:
  python ablation.py --group group1       # run a group
  python ablation.py --group group1 --list
  python ablation.py --name g1_01_muon   # single experiment
  python ablation.py --force             # re-run even if results exist
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

YAML_PATH = Path(__file__).parent.parent / "tools" / "ablations.yaml"
EXPERIMENTS_DIR = "./experiments/ablation"


def load_yaml() -> tuple[dict, list[dict]]:
    data = yaml.safe_load(YAML_PATH.read_text())
    return dict(data["base"]), data["groups"]


def resolve(base: dict, group_base: dict, delta: dict) -> dict:
    return {**base, **group_base, **delta}


def build_payload(cfg: dict, group_name: str, name: str) -> dict:
    return {**cfg, "experiments_dir": f"{EXPERIMENTS_DIR}/{group_name}/{name}"}


def find_result(group_name: str, name: str) -> dict | None:
    exp_base = Path(EXPERIMENTS_DIR) / group_name / name
    if not exp_base.exists():
        return None
    for exp_dir in sorted(exp_base.iterdir()):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            run_yaml = run_dir / "run.yaml"
            if run_yaml.exists():
                data = yaml.safe_load(run_yaml.read_text()) or {}
                if "results" in data:
                    return data["results"]
    return None


def run_experiment(cfg: dict, group_name: str, name: str, desc: str) -> dict | None:
    payload = build_payload(cfg, group_name, name)
    print(f"\n{'='*60}")
    print(f"Running: [{group_name}] {name}")
    print(f"  {desc}")
    print(f"{'='*60}")
    cmd = [sys.executable, "train_hybrid.py", json.dumps(payload)]
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"[ablation] FAILED: {name} (exit {result.returncode})", file=sys.stderr)
        return None
    return find_result(group_name, name)


def print_summary(results: list[tuple[str, dict | None]]):
    print(f"\n{'='*60}")
    print(f"{'Experiment':<30} {'val_loss':>10} {'tok/s':>10} {'params':>12}")
    print(f"{'-'*60}")
    for name, r in results:
        if r is None:
            print(f"{name:<30} {'FAILED':>10}")
        else:
            print(f"{name:<30} {r.get('val_loss', '?'):>10.4f} "
                  f"{r.get('avg_tok_s', 0):>10,.0f} "
                  f"{r.get('total_params', 0):>12,}")


def iter_experiments(groups: list[dict], group_filter: str | None, name_filter: str | None):
    """Yield (group_name, name, desc, resolved_cfg) tuples."""
    base_data, _ = load_yaml()  # not used here, caller passes base
    for group in groups:
        if group_filter and group["name"] != group_filter:
            continue
        gb = dict(group.get("group_base", {}))
        for entry in group.get("experiments", []):
            name = entry["name"]
            if name_filter and name != name_filter:
                continue
            desc = entry.get("desc", "")
            delta = dict(entry.get("delta", {}))
            yield group["name"], name, desc, gb, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", help="run a specific group")
    parser.add_argument("--name",  help="run a single experiment by name")
    parser.add_argument("--list",  action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base, groups = load_yaml()

    collected: list[tuple[str, dict | None]] = []
    for group_name, name, desc, group_base, delta in iter_experiments(
        groups, args.group, args.name
    ):
        cfg = resolve(base, group_base, delta)

        if args.list:
            print(f"\n# [{group_name}] {name}: {desc}")
            print(json.dumps(build_payload(cfg, group_name, name), indent=2))
            continue

        existing = find_result(group_name, name)
        if existing and not args.force:
            print(f"[skip] {name} — results exist (--force to re-run)")
            collected.append((name, existing))
            continue

        result = run_experiment(cfg, group_name, name, desc)
        collected.append((name, result))

    if not args.list:
        print_summary(collected)


if __name__ == "__main__":
    main()
