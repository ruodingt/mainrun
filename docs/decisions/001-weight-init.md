# 001 — Weight Initialization Strategy

**Status**: active experiment  
**Hyperparameter**: `weight_init: str` — select per run  
**Options**: `"gpt2"` (default) | `"muon_uniform"`

---

## Context

Weight initialization interacts differently with AdamW vs Muon:

- **AdamW** is scale-sensitive: init std directly affects early gradient magnitudes and loss trajectory.
- **Muon** is approximately scale-invariant: it orthogonalizes the gradient update direction via Polar Express, so the learning rate (not init scale) controls update magnitude. For Muon-optimized params, *distribution shape* (Normal vs Uniform) matters more than *scale*.

Our model has two optimizer groups:
- **Muon**: all `ndim >= 2` weight matrices except embeddings and lm_head — i.e. `q_proj`, `kv_proj`, `attn.proj`, `mlp.net[0]`, `mlp.net[2]`
- **AdamW**: `token_emb`, norms, biases, ReZero scalars, and `lm_head` (tied to `token_emb`)

---

## `gpt2` — GPT-2 Style (current default)

```python
Normal(mean=0.0, std=0.02)  # all Linear and Embedding weights
zeros                        # all biases
```

**Pros**
- Battle-tested, zero surprises
- Works well with AdamW; acceptable with Muon

**Cons**
- `std=0.02` is arbitrary — not derived from fan-in, not tuned for Muon
- Normal distribution has tails → outlier singular values → slightly rougher Polar Express convergence
- No special treatment for residual exits: blocks do not start as identity

**When to use**: baseline, first run of any new architecture change.

---

## `muon_uniform` — nanoamd-Inspired (Muon-aware)

Inspired by [nanoamd/nanochat/gpt.py](../../../nanoamd/nanochat/gpt.py).

```
token_emb:       Normal(0, 0.8)           — large std, tokens well-separated
q_proj, kv_proj: Uniform(-s, s)           — s = sqrt(3) / sqrt(d_model)
attn.proj:       zeros                    — residual exit, block starts as identity
mlp.net[0]:      Uniform(-0.4s, 0.4s)    — 0.4x scale compensates 4x dim expansion
mlp.net[2]:      zeros                    — residual exit
lm_head:         inherits token_emb       — weight-tied, cannot init independently
```

### Why Uniform instead of Normal for Muon params?

`Normal(0, σ)` produces a random matrix whose singular value spectrum follows the Marchenko-Pastur law with a right tail. Polar Express (Muon's orthogonalization) must "flatten" this spectrum to reach an orthogonal matrix; outlier singular values slow convergence and require more Newton-Schulz iterations.

`Uniform[-s, s]` with the same std has **no tails** → flatter initial singular value spectrum → cleaner orthogonalization from step 1.

### Why std = 1/sqrt(fan_in) instead of 0.02?

Fan-in scaling keeps pre-activation variance constant regardless of layer width. For `d_model=512`: `1/sqrt(512) ≈ 0.044` vs `0.02`. More signal at step 0, principled derivation.

### Why zero residual exits?

`attn.proj` and `mlp.net[2]` are the only paths from sublayer back to the residual stream. Zero-initializing them makes every Block start as exact identity — gradients flow through the full network from step 0 without any sublayer interference. This compounds with `use_rezero=True` (learnable scalars also init to 0) for double protection.

### Why 0.4x scale on MLP up-proj?

The MLP expands `d_model → 4*d_model`. Without scaling, the pre-activation variance at the nonlinearity input is 4× larger than in the attention path. `0.4x` roughly equalizes variance across sublayers.

### Known tradeoff: lm_head init

**Ideal**: `lm_head` at `std=0.001` → near-uniform logits → CE loss starts at `ln(vocab_size) ≈ 9.0` → large, well-conditioned gradients from step 0.

**Reality**: `lm_head.weight` is tied to `token_emb.weight` (weight tying saves 4.2M params). Writing to one overwrites the other. Independently initializing both would require untying — spending the exact 4.2M params saved by halving vocab size from 16384→8192.

**Decision**: keep weight tying, accept `lm_head` inheriting `std=0.8`. Loss starts slightly below `ln(vocab_size)` but remains well above zero. Not a blocker.

**Future option (Plan C)**: untie weights, init `lm_head` to `std=0.001`. Re-evaluate if Plan B shows strong gains and we have parameter budget to spend.

---

## Comparison

| Property | Plan A | Plan B |
|---|---|---|
| Muon param distribution | Normal | Uniform |
| Muon param std | 0.02 (fixed) | 1/sqrt(fan_in) |
| Residual exits | random | zero |
| MLP up-proj scale | 1.0× | 0.4× |
| Embedding std | 0.02 | 0.8 |
| lm_head std | 0.02 | 0.8 (tied) |
| Block identity at step 0 | no | yes (with ReZero) |
| Complexity | minimal | moderate |

---

## Experiment Log

| Run | `weight_init` | val_loss | notes |
|---|---|---|---|
| — | gpt2         | TBD | baseline |
| — | muon_uniform | TBD | — |
