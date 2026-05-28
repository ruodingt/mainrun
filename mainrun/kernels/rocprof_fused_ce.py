"""
Minimal single-variant runner for rocprof profiling.

Run under rocprof --stats to compare kernel-level timings:

  rocprof --stats -o /tmp/ce_std.csv   python kernels/rocprof_fused_ce.py
  rocprof --stats -o /tmp/ce_fused.csv python kernels/rocprof_fused_ce.py --fused

Usage:
  python kernels/rocprof_fused_ce.py [--fused] [--steps N] [--softcap F]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from hparams import Hyperparameters
from hybrid import HybridLM

DEVICE = "cuda"
BATCH  = 64
SEQLEN = 128
VOCAB  = 10240


def make_model(fused: bool, softcap: float) -> HybridLM:
    hp = Hyperparameters()
    hp.runtime.use_fused_ce = fused
    hp.arch.logit_softcap   = softcap
    hp.arch.dropout         = 0.0
    return HybridLM(hp, vocab_size=VOCAB).to(DEVICE)


def synth_batch():
    x = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    y = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused",   action="store_true")
    parser.add_argument("--steps",   type=int,   default=10)
    parser.add_argument("--warmup",  type=int,   default=3)
    parser.add_argument("--softcap", type=float, default=15.0)
    args = parser.parse_args()

    label = "fused CE v2" if args.fused else "standard CE"
    print(f"rocprof target: {label}  steps={args.steps}  warmup={args.warmup}")

    model = make_model(args.fused, args.softcap)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(args.warmup):
        x, y = synth_batch()
        _, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    print("warmup done")

    t0 = time.perf_counter()
    for _ in range(args.steps):
        x, y = synth_batch()
        _, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms = elapsed / args.steps * 1e3
    tok_s = BATCH * SEQLEN * args.steps / elapsed
    print(f"{label}: {ms:.1f} ms/step  {tok_s:,.0f} tok/s")


if __name__ == "__main__":
    main()
