"""
Sync ablation results from remote to local.

Two steps:
  1. rsync remote experiments/ablation/ → local experiments/ablation/  (full mirror)
  2. Walk local tree, copy each experiment's log.txt →
       logs/ablation/{group}/mainrun_{exp_name}.log

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
REMOTE_HYPERTUNE_FULL = "~/workspace/mainrun/experiments/hypertune-full/"

ROOT = Path(__file__).parent.parent
LOCAL_ABLATION = ROOT / "mainrun" / "experiments" / "ablation"
LOCAL_HYPERTUNE_FULL = ROOT / "mainrun" / "experiments" / "hypertune-full"
LOCAL_LOGS = ROOT / "mainrun" / "logs"
YAML_PATH = Path(__file__).parent / "ablations.yaml"


def rsync_ablation():
    LOCAL_ABLATION.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avz", f"{REMOTE_HOST}:{REMOTE_ABLATION}", str(LOCAL_ABLATION) + "/"]
    print(f"[rsync] {REMOTE_HOST}:{REMOTE_ABLATION} → {LOCAL_ABLATION}/")
    subprocess.run(cmd, check=True)


def rsync_hypertune_full():
    LOCAL_HYPERTUNE_FULL.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avz", f"{REMOTE_HOST}:{REMOTE_HYPERTUNE_FULL}", str(LOCAL_HYPERTUNE_FULL) + "/"]
    print(f"[rsync] {REMOTE_HOST}:{REMOTE_HYPERTUNE_FULL} → {LOCAL_HYPERTUNE_FULL}/")
    subprocess.run(cmd, check=True)


def collect_hypertune_log():
    """Copy each hypertune-full exp's log.txt → logs/ablation/hypertune/mainrun_{exp}.log"""
    log_dir = LOCAL_LOGS / "ablation" / "hypertune"
    synced = 0
    for exp_dir in sorted(LOCAL_HYPERTUNE_FULL.glob("exp*")):
        candidates = sorted(
            exp_dir.glob("run*/log.txt"),
            key=lambda p: p.parent.name,
            reverse=True,
        )
        if not candidates:
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        dst = log_dir / f"mainrun_{exp_dir.name}.log"
        shutil.copy2(candidates[0], dst)
        print(f"  ok  hypertune-full/{exp_dir.name} → logs/ablation/hypertune/mainrun_{exp_dir.name}.log")
        synced += 1
    if synced == 0:
        print("  [hypertune-full] no logs found yet")


def collect_logs(group_filter: str | None):
    data = yaml.safe_load(YAML_PATH.read_text())

    synced, missing = [], []

    for group in data["groups"]:
        group_name = group["name"]
        if group_filter and group_name != group_filter:
            continue

        group_log_dir = LOCAL_LOGS / "ablation" / group_name
        group_log_dir.mkdir(parents=True, exist_ok=True)

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

            dst = group_log_dir / f"mainrun_{name}.log"
            shutil.copy2(log_src, dst)
            synced.append(name)
            print(f"  ok  {name} → logs/ablation/{group_name}/mainrun_{name}.log")

    print(f"\nSynced: {len(synced)}  |  Not yet run: {len(missing)}")
    if missing:
        print(f"  missing: {', '.join(missing)}")


def main():
    group_filter = sys.argv[1] if len(sys.argv) > 1 else None
    rsync_ablation()
    rsync_hypertune_full()
    print()
    collect_logs(group_filter)
    collect_hypertune_log()


if __name__ == "__main__":
    main()
