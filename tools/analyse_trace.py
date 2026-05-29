#!/usr/bin/env python3
"""
Analyse a PyTorch Chrome Trace (.pt.trace.json) for GPU utilisation and
kernel fusion opportunities.

Usage:
    python3 tools/analyse_trace.py profile_out/<file>.pt.trace.json
    python3 tools/analyse_trace.py profile_out/<file>.pt.trace.json --top 30
"""

import json
import sys
import bisect
import argparse
import collections


def load_kernels(path: str) -> list[dict]:
    with open(path) as f:
        trace = json.load(f)
    kernels = [
        e for e in trace.get("traceEvents", [])
        if e.get("ph") == "X" and e.get("cat") == "kernel"
    ]
    kernels.sort(key=lambda e: e["ts"])
    return kernels


def short_name(name: str) -> str:
    if name.startswith("Cijk"):
        return "GEMM:" + name[4:8]
    if "multi_tens" in name:
        return "multi_tensor_op"
    if "distribution" in name:
        return "dropout_fill"
    if "vectorized_elementwise" in name:
        return "elementwise"
    return name[:50]


def classify_gap_source(prev_name: str) -> str:
    if "multi_tens" in prev_name:
        return "Muon multi_tensor_op"
    if "triton_poi_fused_add_copy__div_lerp" in prev_name:
        return "Muon nesterov step"
    return prev_name[:50]


def analyse(path: str, top: int = 20) -> None:
    print(f"Loading {path} ...")
    kernels = load_kernels(path)
    n = len(kernels)
    print(f"GPU kernel events: {n}\n")

    total_span    = kernels[-1]["ts"] + kernels[-1]["dur"] - kernels[0]["ts"]
    total_compute = sum(k["dur"] for k in kernels)
    total_gap     = total_span - total_compute

    print("=" * 60)
    print("GPU UTILISATION")
    print("=" * 60)
    print(f"  Time span:      {total_span/1e3:8.1f} ms")
    print(f"  Compute:        {total_compute/1e3:8.1f} ms  ({100*total_compute/total_span:.1f}%)")
    print(f"  Idle (gaps):    {total_gap/1e3:8.1f} ms  ({100*total_gap/total_span:.1f}%)")

    # --- Gap distribution ---
    print("\n" + "=" * 60)
    print("GAP DISTRIBUTION")
    print("=" * 60)
    bins = [0, 2, 5, 10, 20, 50, 100, 500, 1000, 5000]
    bin_counts = [0] * len(bins)
    bin_total  = [0.0] * len(bins)
    small_gap_total = 0.0
    large_gaps = []

    for i in range(1, n):
        gap = kernels[i]["ts"] - (kernels[i-1]["ts"] + kernels[i-1]["dur"])
        if gap <= 0:
            continue
        idx = bisect.bisect_right(bins, gap) - 1
        bin_counts[idx] += 1
        bin_total[idx]  += gap
        if gap <= 200:
            small_gap_total += gap
        else:
            large_gaps.append({
                "gap":      gap,
                "prev":     kernels[i-1]["name"],
                "prev_dur": kernels[i-1]["dur"],
                "next":     kernels[i]["name"],
            })

    for i, lo in enumerate(bins):
        hi = bins[i+1] if i+1 < len(bins) else "∞"
        print(f"  [{lo:5} – {str(hi):5} µs]: {bin_counts[i]:5} gaps  {bin_total[i]/1e3:7.1f} ms")

    # --- Large gap summary ---
    large_gap_total = sum(g["gap"] for g in large_gaps)
    print(f"\n  Launch overhead (≤200µs): {small_gap_total/1e3:.1f} ms")
    print(f"  Real bubbles   (>200µs):  {large_gap_total/1e3:.1f} ms  ({len(large_gaps)} events)")

    if large_gaps:
        by_src = collections.defaultdict(list)
        for g in large_gaps:
            by_src[classify_gap_source(g["prev"])].append(g["gap"])
        print("\n  Real bubbles by source:")
        for src, gs in sorted(by_src.items(), key=lambda x: -sum(x[1])):
            print(f"    {sum(gs)/1e3:6.1f} ms | {len(gs):3}x avg {sum(gs)/len(gs):.0f} µs | {src}")

    # --- Top kernels by total GPU time ---
    print("\n" + "=" * 60)
    print(f"TOP {top} KERNELS BY TOTAL GPU TIME")
    print("=" * 60)
    by_name = collections.defaultdict(list)
    for k in kernels:
        by_name[k["name"]].append(k["dur"])

    stats = sorted(
        [(sum(d), len(d), sum(d)/len(d), name) for name, d in by_name.items()],
        reverse=True,
    )
    print(f"  {'Total(ms)':>10} {'Count':>6} {'Avg(µs)':>9}  Name")
    for total, cnt, avg, name in stats[:top]:
        print(f"  {total/1e3:>10.2f} {cnt:>6} {avg:>9.1f}  {name[:70]}")

    # --- Short-lived kernels (fusion candidates) ---
    print("\n" + "=" * 60)
    print("SHORT-LIVED KERNELS  (avg < 20µs, count ≥ 50)")
    print("=" * 60)
    short = [(t, c, a, nm) for t, c, a, nm in stats if a < 20 and c >= 50]
    if short:
        tiny_compute  = sum(t for t, c, a, nm in short)
        tiny_count    = sum(c for t, c, a, nm in short)
        launch_cost   = tiny_count * 5  # ~5µs HIP launch latency
        print(f"  {len(short)} distinct kernel types, {tiny_count} total calls")
        print(f"  Actual compute: {tiny_compute/1e3:.1f} ms")
        print(f"  Estimated launch overhead: ~{launch_cost/1e3:.1f} ms  (assuming 5µs/launch)")
        print(f"\n  {'Total(ms)':>10} {'Count':>6} {'Avg(µs)':>9}  Name")
        for total, cnt, avg, name in short:
            print(f"  {total/1e3:>10.2f} {cnt:>6} {avg:>9.1f}  {name[:70]}")
    else:
        print("  None found.")

    # --- Frequent kernel pairs with large average gaps ---
    print("\n" + "=" * 60)
    print("KERNEL PAIRS WITH NOTABLE GAPS  (≥10 occurrences, avg gap ≥5µs)")
    print("=" * 60)
    pair_gaps = collections.defaultdict(list)
    for i in range(1, n):
        gap = kernels[i]["ts"] - (kernels[i-1]["ts"] + kernels[i-1]["dur"])
        if gap <= 0:
            continue
        key = (short_name(kernels[i-1]["name"]), short_name(kernels[i]["name"]))
        pair_gaps[key].append(gap)

    candidates = [
        (sum(gs), len(gs), sum(gs)/len(gs), p, nxt)
        for (p, nxt), gs in pair_gaps.items()
        if len(gs) >= 10 and sum(gs)/len(gs) >= 5
    ]
    candidates.sort(reverse=True)
    if candidates:
        print(f"  {'TotalGap(ms)':>13} {'N':>5} {'AvgGap(µs)':>11}  Pair")
        for total, cnt, avg, p, nxt in candidates[:20]:
            print(f"  {total/1e3:>13.2f} {cnt:>5} {avg:>11.1f}  {p} → {nxt}")
    else:
        print("  None found.")


def main():
    parser = argparse.ArgumentParser(description="Analyse PyTorch Chrome Trace for GPU fusion opportunities")
    parser.add_argument("trace", help="Path to .pt.trace.json file")
    parser.add_argument("--top", type=int, default=20, help="Number of top kernels to show (default 20)")
    args = parser.parse_args()
    analyse(args.trace, top=args.top)


if __name__ == "__main__":
    main()
