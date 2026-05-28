"""
ProfilerMixin — mixed into Trainer to add profile() without bloating train_hybrid.py.

Assumes self has: args, device, model, model_params, opt, train_ids,
                  _setup_data(), _build_model(), _build_optimizer(), _compile_model()
"""

import contextlib
import datetime
import os
import time

import torch
from data_utils import get_batch




def _cuda_us(e) -> float:
    """ROCm uses self_cuda_time_total; CUDA uses cuda_time_total."""
    return getattr(e, "cuda_time_total", None) or getattr(e, "self_cuda_time_total", 0)


def _top_overhead(key_avgs, skip=("hipDeviceSynchronize", "record_function")):
    rows = []
    for e in key_avgs:
        if any(s in e.key for s in skip):
            continue
        if e.cpu_time_total > 0 and e.count > 0:
            rows.append((e.cpu_time_total, e))
    rows.sort(reverse=True)
    return rows


class ProfilerMixin:

    # TODO：Tactical solution kto keep trainer small
    #  highly relying on Trainer right now, more decoupling needed later - using mock data
    def profile(self, warmup: int = 3, steps: int = 10,
                output: str = "profile_out", topk: int = 20):
        from torch.profiler import profile as tprofile, record_function, ProfilerActivity

        self._setup_data()
        self._build_model()
        self._build_optimizer()
        self._compile_model()

        device     = self.device
        block_size = self.args.train.block_size
        batch_size = self.args.train.batch_size
        amp_ctx    = torch.amp.autocast(device_type=device, dtype=torch.bfloat16) \
                     if self.args.runtime.use_bf16 else contextlib.nullcontext()
        ptr = 0

        def _step():
            nonlocal ptr
            xb, yb, ptr = get_batch(self.train_ids, ptr, block_size, batch_size, device)
            self.opt.zero_grad(set_to_none=True)
            with amp_ctx:
                with record_function("forward"):
                    _, loss = self.model(xb, yb)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer_step"):
                self.opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            return loss.item()

        self.model.train()
        print(f"Profiler: {warmup} warmup + {steps} profile steps")
        print(f"  model: {self.model_params/1e6:.1f}M params  device: {device}")

        graph_break_info = self._analyze_graph_breaks(block_size, batch_size, device)

        for _ in range(warmup):
            _step()
        print("  warmup done")

        os.makedirs(output, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats()

        with tprofile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output),
        ) as prof:
            t0 = time.time()
            for _ in range(steps):
                _step()
                prof.step()
            elapsed = time.time() - t0  # measure before trace file write
        tok_s    = steps * block_size * batch_size / elapsed
        key_avgs = prof.key_averages()
        total_cuda = sum(_cuda_us(e) for e in key_avgs)
        sorted_avgs = sorted(key_avgs, key=_cuda_us, reverse=True)
        overhead    = _top_overhead(key_avgs)

        alloc = rsvd = 0.0
        if device == "cuda":
            alloc = torch.cuda.max_memory_allocated() / 1e9
            rsvd  = torch.cuda.max_memory_reserved() / 1e9

        rocm_mode, gpu_busy_us = self._bubble_stats(key_avgs, total_cuda, elapsed)

        self._print_report(tok_s, elapsed, sorted_avgs, overhead, total_cuda,
                           gpu_busy_us, rocm_mode, alloc, rsvd, topk, output)
        self._write_md(tok_s, elapsed, sorted_avgs, overhead, total_cuda,
                       gpu_busy_us, rocm_mode, alloc, rsvd, graph_break_info,
                       topk, output)

    # ------------------------------------------------------------------

    def _analyze_graph_breaks(self, block_size, batch_size, device):
        print("\nGraph break analysis...")
        xb, yb, _ = get_batch(self.train_ids, 0, block_size, batch_size, device)
        explanation = torch._dynamo.explain(self.model)(xb, yb)
        n_graphs = len(explanation.graphs)
        n_breaks = len(explanation.break_reasons)
        print(f"  graphs: {n_graphs}  breaks: {n_breaks}")
        if explanation.break_reasons:
            seen = {}
            for br in explanation.break_reasons:
                r = str(br.reason)[:80]
                seen[r] = seen.get(r, 0) + 1
            for r, count in sorted(seen.items(), key=lambda x: -x[1]):
                print(f"    [{count}x] {r}")
        return {"n_graphs": n_graphs, "n_breaks": n_breaks,
                "reasons": [str(br.reason)[:120] for br in explanation.break_reasons]}

    def _bubble_stats(self, key_avgs, total_cuda, elapsed):
        rocm_mode = total_cuda == 0
        if rocm_mode:
            sync_ev = next((e for e in key_avgs if "hipDeviceSynchronize" in e.key), None)
            gpu_busy_us = sync_ev.cpu_time_total if sync_ev else 0
        else:
            gpu_busy_us = total_cuda
        return rocm_mode, gpu_busy_us

    def _print_report(self, tok_s, elapsed, sorted_avgs, overhead, total_cuda,
                      gpu_busy_us, rocm_mode, alloc, rsvd, topk, output):
        elapsed_us = elapsed * 1e6
        bubble_us  = max(elapsed_us - gpu_busy_us, 0)
        busy_pct   = gpu_busy_us / elapsed_us * 100 if elapsed_us > 0 else 0

        print(f"\nThroughput: {tok_s:,.0f} tok/s (approximate)  ({elapsed:.1f}s)")

        col_w  = [48, 7, 12, 7, 10, 12]
        fmt    = "  ".join(f"{{:<{w}}}" for w in col_w)
        header = ["Op", "Count", "CUDA Total", "CUDA%", "Avg/call", "CPU Total"]
        if not rocm_mode:
            print(f"\n{'='*72}\nTop-{topk} ops by CUDA self time\n{'='*72}")
            print(fmt.format(*header))
            print("  ".join("-" * w for w in col_w))
            for e in sorted_avgs[:topk]:
                cuda_us = _cuda_us(e)
                pct     = cuda_us / total_cuda * 100 if total_cuda > 0 else 0
                avg_us  = cuda_us / e.count if e.count > 0 else 0
                print(fmt.format(e.key[:48], str(e.count),
                                 f"{cuda_us/1e3:.1f}ms", f"{pct:.1f}%",
                                 f"{avg_us:.0f}us", f"{e.cpu_time_total/1e3:.1f}ms"))

        if alloc:
            print(f"\nPeak memory: {alloc:.2f} GB allocated  /  {rsvd:.2f} GB reserved")

        print(f"\n{'='*72}\nGPU bubble analysis\n{'='*72}")
        if rocm_mode:
            print("  (ROCm: using hipDeviceSynchronize as GPU busy proxy)")
        print(f"  Wall time   : {elapsed_us/1e3:.1f} ms")
        print(f"  GPU busy    : {gpu_busy_us/1e3:.1f} ms  ({busy_pct:.1f}%)")
        print(f"  Bubble      : {bubble_us/1e3:.1f} ms  ({100-busy_pct:.1f}%)")

        oh_fmt = "  {:<48}  {:>8}  {:>10}  {:>10}"
        print(f"\nTop CPU-dispatch overhead (bubble sources):")
        print(oh_fmt.format("Op", "Count", "CPU time", "per call"))
        print(oh_fmt.format("-"*48, "-"*8, "-"*10, "-"*10))
        for cpu_oh, e in overhead[:10]:
            print(oh_fmt.format(e.key[:48], str(e.count),
                                f"{cpu_oh/1e3:.1f}ms", f"{cpu_oh/e.count:.0f}us"))

        print(f"\nTrace → {output}/")
        print(f"  TensorBoard: tensorboard --logdir {output}")

    def _write_md(self, tok_s, elapsed, sorted_avgs, overhead, total_cuda,
                  gpu_busy_us, rocm_mode, alloc, rsvd, graph_break_info,
                  topk, output):
        elapsed_us = elapsed * 1e6
        bubble_us  = max(elapsed_us - gpu_busy_us, 0)
        busy_pct   = gpu_busy_us / elapsed_us * 100 if elapsed_us > 0 else 0
        cfg        = self.args

        lines = [
            "# Profile Report", "",
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"**Config:** `{cfg.arch.layer_pattern}`  ",
            f"**Model:** {self.model_params/1e6:.1f}M params, "
            f"{cfg.arch.n_layer}L × {cfg.arch.d_model}d, vocab={cfg.arch.vocab_size}  ",
            f"**Throughput:** {tok_s:,.0f} tok/s (approximate — excludes trace file write)  ",
            "",
            "## Graph Breaks", "",
            "| | |", "|---|---|",
            f"| Graphs | {graph_break_info['n_graphs']} |",
            f"| Breaks | {graph_break_info['n_breaks']} |",
        ]
        if graph_break_info["reasons"]:
            seen = {}
            for r in graph_break_info["reasons"]:
                seen[r] = seen.get(r, 0) + 1
            lines += ["", "**Break reasons:**", ""]
            for r, count in sorted(seen.items(), key=lambda x: -x[1]):
                lines.append(f"- `[{count}x]` {r}")

        lines += ["", "## Memory", ""]
        if alloc:
            lines += ["| | GB |", "|---|---|",
                      f"| Allocated (peak) | {alloc:.2f} |",
                      f"| Reserved (peak)  | {rsvd:.2f} |"]

        lines += [
            "", "## Top CPU-Dispatch Overhead", "",
            "| Op | Count | CPU time | per call |", "|---|---|---|---|",
        ]
        for cpu_oh, e in overhead[:topk]:
            lines.append(f"| `{e.key[:60]}` | {e.count} "
                         f"| {cpu_oh/1e3:.1f}ms | {cpu_oh/e.count:.0f}µs |")

        if not rocm_mode:
            lines += [
                "", "## Top Ops by CUDA Self Time", "",
                "| Op | Count | CUDA Total | CUDA% | Avg/call | CPU Total |",
                "|---|---|---|---|---|---|",
            ]
            for e in sorted_avgs[:topk]:
                cuda_us = _cuda_us(e)
                pct    = cuda_us / total_cuda * 100 if total_cuda > 0 else 0
                avg_us = cuda_us / e.count if e.count > 0 else 0
                lines.append(f"| `{e.key[:60]}` | {e.count} | {cuda_us/1e3:.1f}ms "
                             f"| {pct:.1f}% | {avg_us:.0f}µs | {e.cpu_time_total/1e3:.1f}ms |")

        md_path = os.path.join(output, "report.md")
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Markdown:    {md_path}")
