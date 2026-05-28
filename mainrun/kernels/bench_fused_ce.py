"""
Benchmark: standard CE  vs  fused CE v2  in a real training step.

Measures forward+backward+optimizer over N steps using synthetic data.
No data loading — just random token IDs so results reflect pure compute.

Run (from mainrun/):
    python kernels/bench_fused_ce.py
    python kernels/bench_fused_ce.py --softcap 0
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from hparams import Hyperparameters
from hybrid import HybridLM

N_WARMUP = 5
N_STEPS  = 30
DEVICE   = "cuda"

BATCH   = 64
SEQLEN  = 128
VOCAB   = 10240


def make_model(use_fused_ce: bool, logit_softcap: float) -> HybridLM:
    hp = Hyperparameters()
    hp.runtime.use_fused_ce = use_fused_ce
    hp.arch.logit_softcap = logit_softcap
    hp.arch.dropout = 0.0
    return HybridLM(hp, vocab_size=VOCAB).to(DEVICE)


def synth_batch():
    x = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    y = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    return x, y


def run_steps(model, n_steps, n_warmup, label):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    for _ in range(n_warmup):
        x, y = synth_batch()
        _, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for _ in range(n_steps):
        x, y = synth_batch()
        _, loss = model(x, y)
        losses.append(loss.item())
        loss.backward()
        opt.step(); opt.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    ms_per_step = elapsed / n_steps * 1e3
    print(f"  {label:<30}  {ms_per_step:7.2f} ms/step   peak {peak_mb:7.1f} MB   "
          f"loss {losses[0]:.4f} → {losses[-1]:.4f}")
    return ms_per_step, peak_mb, losses


def bench(logit_softcap: float):
    hp = Hyperparameters()
    cap_str = f"softcap={logit_softcap}" if logit_softcap > 0 else "no softcap"
    print(f"\n{'─'*65}")
    print(f"  B={BATCH}  T={SEQLEN}  V={VOCAB}  d={hp.arch.d_model}  L={hp.arch.n_layer}  ({cap_str})")
    print(f"{'─'*65}")

    model_std = make_model(use_fused_ce=False, logit_softcap=logit_softcap)
    sd = {k: v.clone() for k, v in model_std.state_dict().items()}
    ms_std, mem_std, losses_std = run_steps(model_std, N_STEPS, N_WARMUP, "standard CE")
    del model_std; torch.cuda.empty_cache()

    model_fuse = make_model(use_fused_ce=True, logit_softcap=logit_softcap)
    model_fuse.load_state_dict(sd)
    ms_fuse, mem_fuse, losses_fuse = run_steps(model_fuse, N_STEPS, N_WARMUP, "fused CE v2 (tl.dot)")
    del model_fuse; torch.cuda.empty_cache()

    print(f"\n  speedup  : {ms_std / ms_fuse:.2f}x  ({'faster' if ms_fuse < ms_std else 'slower'})")
    print(f"  mem save : {mem_std / mem_fuse:.2f}x")
    loss_drift = max(abs(a - b) for a, b in zip(losses_std, losses_fuse))
    print(f"  loss drift (max |std - fused|) : {loss_drift:.4f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--softcap", type=float, default=15.0)
    args = parser.parse_args()
    bench(args.softcap)
    if args.softcap != 0.0:
        bench(0.0)
