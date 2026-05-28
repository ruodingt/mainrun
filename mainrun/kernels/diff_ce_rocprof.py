"""Print side-by-side kernel timing: standard CE vs fused CE v2."""
import csv
import os
import sys

out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/tmp/rocprof_ce")


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.reader(f):
            if len(r) < 3:
                continue
            try:
                rows[r[0].strip('"')] = int(r[2])
            except ValueError:
                pass
    return rows


std   = load(f"{out}/std.stats.csv")
fused = load(f"{out}/fused.stats.csv")

all_k = sorted(set(std) | set(fused), key=lambda k: -(std.get(k, 0) + fused.get(k, 0)))

print(f"{'Kernel':<55}  {'std ms':>8}  {'fused ms':>8}")
print("-" * 75)
for k in all_k[:25]:
    s  = std.get(k, 0) / 1e6
    fu = fused.get(k, 0) / 1e6
    print(f"{k[:55]:<55}  {s:8.1f}  {fu:8.1f}")
