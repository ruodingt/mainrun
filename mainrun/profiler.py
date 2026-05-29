"""
# TODO：Experimental Only, GPU util computed is different from rocm-smi
ProfilerMixin — mixed into Trainer to add profile() without bloating train_hybrid.py.

Outputs:
  - Chrome Trace (.pt.trace.json) via tensorboard_trace_handler
  - Graph break analysis (stdout)

Assumes self has: args, device, model, model_params, opt, train_ids,
                  _setup_data(), _build_model(), _build_optimizer(), _compile_model()
"""

import contextlib
import os
import time

import torch
from data_utils import get_batch


class ProfilerMixin:

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
            return loss.item()

        self.model.train()
        print(f"Profiler: {warmup} warmup + {steps} profile steps")
        print(f"  model: {self.model_params/1e6:.1f}M params  device: {device}")

        self._analyze_graph_breaks(block_size, batch_size, device)

        for _ in range(warmup):
            _step()
        print("  warmup done")

        os.makedirs(output, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with tprofile(
            activities=activities,
            record_shapes=False,
            with_stack=False,  # ROCm does not populate stackFrames; True adds profiler overhead without benefit
            profile_memory=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output),
        ) as prof:
            t0 = time.time()
            for _ in range(steps):
                _step()
                prof.step()
            elapsed = time.time() - t0

        tok_s = steps * block_size * batch_size / elapsed
        print(f"\nThroughput: {tok_s:,.0f} tok/s ({elapsed:.1f}s for {steps} steps)")
        print(f"Trace → {output}/")
        print(f"  Open in Chrome: chrome://tracing  (load the .pt.trace.json file)")
        print(f"  Or TensorBoard: tensorboard --logdir {output}")

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
        else:
            print("  No graph breaks — torch.compile captured the full graph.")
