"""
Benchmark: standard CE  vs  fused CE v2  in a real training step.

Measures forward+backward+optimizer over N steps using synthetic data.
No data loading — just random token IDs so results reflect pure compute.

Run:
    python bench_fused_ce.py
    python bench_fused_ce.py --softcap 0   # compare without softcap too
"""
import argparse
import sys
import time

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Pull model + config directly from train.py
# --------------------------------------------------------------------------
sys.path.insert(0, ".")
from train import GPT, GPTConfig

# --------------------------------------------------------------------------

N_WARMUP = 5
N_STEPS  = 30
DEVICE   = "cuda"

# Default training shape
BATCH   = 128
SEQLEN  = 64
VOCAB   = 8192
D_MODEL = 384
N_LAYER = 12
N_HEADS = 6


def make_model(use_fused_ce: bool, logit_softcap: float) -> GPT:
    cfg = GPTConfig(
        vocab_size=VOCAB,
        block_size=SEQLEN,
        n_layer=N_LAYER,
        n_q_head=N_HEADS,
        n_kv_heads=1,
        d_model=D_MODEL,
        dropout=0.0,
        use_fa2=True,
        use_rezero=False,
        use_rmsnorm=True,
        use_token_anchor=True,
        use_resid_scale=False,
        weight_init="muon_uniform",
        tie_weights=True,
        norm_emb=False,
        logit_softcap=logit_softcap,
        mlp_act="relu_sq",
        use_fused_ce=use_fused_ce,
    )
    return GPT(cfg).to(DEVICE)


def synth_batch():
    x = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    y = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    return x, y


def run_steps(model, n_steps, n_warmup, label):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    # warmup
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
    cap_str = f"softcap={logit_softcap}" if logit_softcap > 0 else "no softcap"
    print(f"\n{'─'*65}")
    print(f"  B={BATCH}  T={SEQLEN}  V={VOCAB}  d={D_MODEL}  L={N_LAYER}  ({cap_str})")
    print(f"{'─'*65}")

    # ── Standard CE ──────────────────────────────────────────────────────
    model_std = make_model(use_fused_ce=False, logit_softcap=logit_softcap)
    # copy weights for apples-to-apples loss comparison
    sd = {k: v.clone() for k, v in model_std.state_dict().items()}

    ms_std, mem_std, losses_std = run_steps(
        model_std, N_STEPS, N_WARMUP, "standard CE"
    )
    del model_std; torch.cuda.empty_cache()

    # ── Fused CE v2 ───────────────────────────────────────────────────────
    model_fuse = make_model(use_fused_ce=True, logit_softcap=logit_softcap)
    model_fuse.load_state_dict(sd)   # same init → comparable loss

    ms_fuse, mem_fuse, losses_fuse = run_steps(
        model_fuse, N_STEPS, N_WARMUP, "fused CE v2 (tl.dot)"
    )
    del model_fuse; torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
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
