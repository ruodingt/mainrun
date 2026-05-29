#!/usr/bin/env python3
"""
Analyse a PyTorch Chrome Trace (.pt.trace.json) for GPU utilisation and bubbles.

Bubble finding:
  - Voids are found directly as first-class objects: all time ranges on the GPU
    stream where no kernel is executing (assumes single stream, confirmed by cid).
  - For each void, the next kernel's `correlation` id is used to look up the
    exact CPU dispatch event, giving a precise dispatch timestamp rather than
    guessing from ac2g proximity.

Usage:
    python3 tools/analyse_trace.py profile_out/<file>.pt.trace.json
    python3 tools/analyse_trace.py profile_out/<file>.pt.trace.json --top 30 --bubble-top 10 --bubble-threshold 200
"""

import json
import bisect
import argparse
import collections


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_kernels(events: list) -> list:
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    kernels.sort(key=lambda e: e["ts"])
    return kernels


def build_correlation_map(events: list) -> dict:
    """correlation_id → CPU dispatch event (cuda_runtime X events).

    GPU kernel events carry args.correlation matching the CPU-side launch call.
    This lets us find exactly when the CPU dispatched a given GPU kernel.
    """
    corr_map = {}
    for e in events:
        if e.get("cat") == "cuda_runtime" and e.get("ph") == "X":
            corr = (e.get("args") or {}).get("correlation") or e.get("correlation")
            if corr is not None:
                corr_map[corr] = e
    return corr_map


def build_cpu_ops_index(events: list) -> tuple[list, list]:
    """Returns (cpu_ops sorted by ts, list of ts values) for bisect lookups."""
    cpu_ops = sorted(
        [e for e in events if e.get("cat") == "cpu_op" and e.get("ph") == "X"],
        key=lambda e: e["ts"],
    )
    return cpu_ops, [e["ts"] for e in cpu_ops]


# ---------------------------------------------------------------------------
# Void finding
# ---------------------------------------------------------------------------

def find_voids(kernels: list, min_gap: float = 0.0) -> list:
    """Find GPU idle periods directly as first-class objects.

    A void is any contiguous time range where no kernel is executing on the
    (single) GPU stream. With one stream, this is unambiguous: it is the
    complement of the union of kernel intervals.
    """
    voids = []
    for i in range(1, len(kernels)):
        t0  = kernels[i-1]["ts"] + kernels[i-1]["dur"]
        t1  = kernels[i]["ts"]
        gap = t1 - t0
        if gap > min_gap:
            voids.append({
                "t0":          t0,
                "t1":          t1,
                "gap":         gap,
                "prev_kernel": kernels[i-1],
                "next_kernel": kernels[i],
            })
    return voids


def enrich_void(void: dict, corr_map: dict,
                cpu_ops: list, cpu_ops_ts: list) -> dict:
    """Add dispatch_ts and cpu_ops_during to a void dict.

    dispatch_ts: when the CPU dispatched the next kernel (via correlation id).
    cpu_ops_during: CPU ops whose ts falls within [t0, t1].
    """
    t0, t1 = void["t0"], void["t1"]

    # Find dispatch time via correlation
    corr = (void["next_kernel"].get("args") or {}).get("correlation") \
           or void["next_kernel"].get("correlation")
    cpu_launch = corr_map.get(corr)
    void["dispatch_ts"] = cpu_launch["ts"] if cpu_launch else None

    # Find CPU ops overlapping the void window
    lo = bisect.bisect_left(cpu_ops_ts, t0)
    hi = bisect.bisect_right(cpu_ops_ts, t1)
    void["cpu_ops_during"] = [
        e for e in cpu_ops[lo:hi]
        if e["ts"] + e.get("dur", 0) > t0
    ]
    return void


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(path: str, top: int = 20, bubble_top: int = 10,
            bubble_threshold: int = 200) -> None:
    print(f"Loading {path} ...")
    trace   = load_trace(path)
    events  = trace.get("traceEvents", [])
    kernels = load_kernels(events)
    n       = len(kernels)
    print(f"GPU kernel events: {n}\n")

    corr_map              = build_correlation_map(events)
    cpu_ops, cpu_ops_ts   = build_cpu_ops_index(events)

    # --- GPU utilisation ---
    total_span    = kernels[-1]["ts"] + kernels[-1]["dur"] - kernels[0]["ts"]
    total_compute = sum(k["dur"] for k in kernels)
    total_void    = total_span - total_compute

    print("=" * 60)
    print("GPU UTILISATION")
    print("=" * 60)
    print(f"  Time span:   {total_span/1e3:8.1f} ms")
    print(f"  Compute:     {total_compute/1e3:8.1f} ms  ({100*total_compute/total_span:.1f}%)")
    print(f"  Void:        {total_void/1e3:8.1f} ms  ({100*total_void/total_span:.1f}%)")

    # --- Find all voids ---
    all_voids = find_voids(kernels, min_gap=0)
    small_voids = [v for v in all_voids if v["gap"] <= bubble_threshold]
    large_voids = [v for v in all_voids if v["gap"] >  bubble_threshold]

    # --- Void size distribution ---
    print("\n" + "=" * 60)
    print("VOID DISTRIBUTION")
    print("=" * 60)
    bins = [0, 2, 5, 10, 20, 50, 100, 500, 1000, 5000]
    bin_counts = [0] * len(bins)
    bin_total  = [0.0] * len(bins)
    for v in all_voids:
        idx = bisect.bisect_right(bins, v["gap"]) - 1
        bin_counts[idx] += 1
        bin_total[idx]  += v["gap"]
    for i, lo in enumerate(bins):
        hi = bins[i+1] if i+1 < len(bins) else "∞"
        print(f"  [{lo:5} – {str(hi):5} µs]: {bin_counts[i]:5} voids  {bin_total[i]/1e3:7.1f} ms")

    small_total = sum(v["gap"] for v in small_voids)
    large_total = sum(v["gap"] for v in large_voids)
    print(f"\n  Launch overhead (≤{bubble_threshold}µs): {small_total/1e3:.1f} ms  ({len(small_voids)} voids)")
    print(f"  Bubbles        (>{bubble_threshold}µs): {large_total/1e3:.1f} ms  ({len(large_voids)} voids)")

    # --- Bubble drill-down ---
    print("\n" + "=" * 60)
    print(f"BUBBLE DRILL-DOWN  (top {bubble_top} by size)")
    print("=" * 60)
    print("  dispatch@ = when CPU sent the next kernel (via correlation id).")
    print("  CPU ops   = what the CPU ran during the void.\n")

    large_voids.sort(key=lambda v: -v["gap"])
    for v in large_voids[:bubble_top]:
        enrich_void(v, corr_map, cpu_ops, cpu_ops_ts)
        t0, t1, gap = v["t0"], v["t1"], v["gap"]

        print(f"  ── {gap:,.0f}µs void ──────────────────────────────────────")
        print(f"     prev: {v['prev_kernel']['name'][:65]}")
        print(f"     next: {v['next_kernel']['name'][:65]}")

        if v["dispatch_ts"] is not None:
            print(f"     dispatch@ +{v['dispatch_ts'] - t0:.0f}µs  "
                  f"(CPU dispatched next kernel {t1 - v['dispatch_ts']:.0f}µs before it started)")
        else:
            print(f"     dispatch: no matching correlation found")

        ops = v["cpu_ops_during"]
        if ops:
            by_name = {}
            for e in ops:
                nm = e["name"]
                if nm not in by_name or e.get("dur", 0) > by_name[nm].get("dur", 0):
                    by_name[nm] = e
            print(f"     CPU ops ({len(ops)} events, {len(by_name)} unique):")
            for e in sorted(by_name.values(), key=lambda x: x["ts"]):
                print(f"       +{e['ts'] - t0:6.0f}µs  {e.get('dur', 0):5.0f}µs  {e['name'][:55]}")
        else:
            print(f"     CPU ops: none (CPU between interpreter ticks or blocking)")
        print()

    # --- Top kernels by total GPU time ---
    print("=" * 60)
    print(f"TOP {top} KERNELS BY TOTAL GPU TIME")
    print("=" * 60)
    by_name = collections.defaultdict(list)
    for k in kernels:
        by_name[k["name"]].append(k["dur"])
    stats = sorted(
        [(sum(d), len(d), sum(d)/len(d), nm) for nm, d in by_name.items()],
        reverse=True,
    )
    print(f"  {'Total(ms)':>10} {'Count':>6} {'Avg(µs)':>9}  Name")
    for total, cnt, avg, nm in stats[:top]:
        print(f"  {total/1e3:>10.2f} {cnt:>6} {avg:>9.1f}  {nm[:70]}")

    # --- Short-lived kernels ---
    print("\n" + "=" * 60)
    print("SHORT-LIVED KERNELS  (avg < 20µs, count ≥ 50)")
    print("=" * 60)
    short = [(t, c, a, nm) for t, c, a, nm in stats if a < 20 and c >= 50]
    if short:
        tiny_compute = sum(t for t, c, a, nm in short)
        tiny_count   = sum(c for t, c, a, nm in short)
        print(f"  {len(short)} distinct types, {tiny_count} total calls, {tiny_compute/1e3:.1f} ms compute")
        print(f"  Estimated launch overhead: ~{tiny_count * 5 / 1e3:.1f} ms  (5µs/launch)")
        print(f"\n  {'Total(ms)':>10} {'Count':>6} {'Avg(µs)':>9}  Name")
        for total, cnt, avg, nm in short:
            print(f"  {total/1e3:>10.2f} {cnt:>6} {avg:>9.1f}  {nm[:70]}")
    else:
        print("  None found.")

    # --- Kernel pairs with notable voids ---
    print("\n" + "=" * 60)
    print("KERNEL PAIRS WITH NOTABLE VOIDS  (≥10 occurrences, avg void ≥5µs)")
    print("=" * 60)
    pair_gaps = collections.defaultdict(list)
    for v in all_voids:
        key = (short_name(v["prev_kernel"]["name"]), short_name(v["next_kernel"]["name"]))
        pair_gaps[key].append(v["gap"])
    candidates = [
        (sum(gs), len(gs), sum(gs)/len(gs), p, nxt)
        for (p, nxt), gs in pair_gaps.items()
        if len(gs) >= 10 and sum(gs)/len(gs) >= 5
    ]
    candidates.sort(reverse=True)
    if candidates:
        print(f"  {'TotalVoid(ms)':>13} {'N':>5} {'AvgVoid(µs)':>12}  Pair")
        for total, cnt, avg, p, nxt in candidates[:20]:
            print(f"  {total/1e3:>13.2f} {cnt:>5} {avg:>12.1f}  {p} → {nxt}")
    else:
        print("  None found.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse PyTorch Chrome Trace — GPU utilisation and bubble root cause"
    )
    parser.add_argument("trace", help="Path to .pt.trace.json file")
    parser.add_argument("--top", type=int, default=20,
                        help="Top N kernels to show (default 20)")
    parser.add_argument("--bubble-top", type=int, default=10,
                        help="Top N bubbles to drill into (default 10)")
    parser.add_argument("--bubble-threshold", type=int, default=200,
                        help="Void threshold µs for 'bubble' vs 'launch overhead' (default 200)")
    args = parser.parse_args()
    analyse(args.trace, top=args.top, bubble_top=args.bubble_top,
            bubble_threshold=args.bubble_threshold)


if __name__ == "__main__":
    main()
