"""
Sync ablation results from remote to local.

Two steps:
  1. rsync remote experiments/ablation/ → local experiments/ablation/  (full mirror)
  2. Walk local tree, copy each experiment's log.txt → logs/ablation_{name}.log

Idempotent: log files are named after the experiment, overwriting is safe.

Usage:
  python tools/sync_ablation.py              # all groups
  python tools/sync_ablation.py group1       # single group
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REMOTE_HOST = "amd-container"
REMOTE_ABLATION = "~/workspace/mainrun/experiments/ablation/"

ROOT = Path(__file__).parent.parent
LOCAL_ABLATION = ROOT / "mainrun" / "experiments" / "ablation"
LOCAL_LOGS = ROOT / "mainrun" / "logs"
YAML_PATH = Path(__file__).parent / "ablations.yaml"


def rsync_ablation():
    LOCAL_ABLATION.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avz", f"{REMOTE_HOST}:{REMOTE_ABLATION}", str(LOCAL_ABLATION) + "/"]
    print(f"[rsync] {REMOTE_HOST}:{REMOTE_ABLATION} → {LOCAL_ABLATION}/")
    subprocess.run(cmd, check=True)


def collect_logs(group_filter: str | None):
    LOCAL_LOGS.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(YAML_PATH.read_text())

    synced, missing = [], []

    for group in data["groups"]:
        group_name = group["name"]
        if group_filter and group_name != group_filter:
            continue
        for entry in group.get("experiments", []):
            if entry.get("skip") is True:
                continue
            name = entry["name"]
            exp_base = LOCAL_ABLATION / group_name / name

            # Find the latest run dir that has a log.txt
            log_src = None
            if exp_base.exists():
                candidates = sorted(
                    exp_base.glob("exp*/run*/log.txt"),
                    key=lambda p: (p.parent.parent.name, p.parent.name),
                    reverse=True,
                )
                if candidates:
                    log_src = candidates[0]

            if log_src is None:
                missing.append(name)
                continue

            dst = LOCAL_LOGS / f"ablation_{name}.log"
            shutil.copy2(log_src, dst)
            synced.append(name)
            print(f"  ok  {name} → logs/ablation_{name}.log")

    print(f"\nSynced: {len(synced)}  |  Not yet run: {len(missing)}")
    if missing:
        print(f"  missing: {', '.join(missing)}")


def main():
    group_filter = sys.argv[1] if len(sys.argv) > 1 else None
    rsync_ablation()
    print()
    collect_logs(group_filter)


if __name__ == "__main__":
    main()
