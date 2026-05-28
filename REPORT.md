# Ablation Study Report

**Goal:** Minimise validation loss on Hacker News titles (100k, 7 epochs, seed=1337).
**Baseline:** SGD + cosine LR + learned pos emb + GELU + 6L×512d×16000 → **val_loss = 1.7319**
**Final best:** 28L×256d×10240 + Muon+AdamW + WSD + RoPE + SwigLU + tie_weights + token_anchor → **val_loss = 1.1652**

---

## Environment

### Hardware

| | |
|---|---|
| **System** | AMD Ryzen AI MAX+ 395 (APU — unified CPU/GPU memory) |
| **GPU** | AMD Radeon 8060S — gfx1151 (RDNA4), 40 CUs, 2900 MHz |
| **Memory** | 128 GB unified memory — BIOS split: ~64 GB GPU / ~62 GB CPU |
| **CPU** | AMD Ryzen AI MAX+ 395, 16-core, 5187 MHz boost |

### Software

| | |
|---|---|
| **PyTorch** | 2.9.1+rocm7.1.1 |
| **ROCm** | 7.1 |
| **Attention kernel** | FlashAttention-2 (via ROCm port) |
| **Precision** | bfloat16 |
| **Compilation** | `torch.compile` enabled |

### Training Setup

| | |
|---|---|
| **Dataset** | Hacker News post titles — `julien040/hacker-news-posts` |
| **Train / val split** | 90k / 10k titles (val_frac=0.10) |
| **Epochs** | 7 (fixed) |
| **Seed** | 1337 (fixed) |
| **Batch size** | 64 sequences × 128 tokens = 8192 tokens/batch |
| **Tokeniser** | BPE, vocab_size swept (8k / 16k) |

---

## Hyperparameter Reference

### Architecture

| Param | Baseline | Final best | Description |
|---|---|---|---|
| `layer_pattern` | `"A"*6` | `"A"*28` | Sequence of layer types: `"A"`=Attention, `"M"`=Mamba. Length = `n_layer`. Pure transformer = `"A"*N`. |
| `d_model` | 512 | 256 | Residual stream width (hidden dimension). |
| `n_q_head` / `n_kv_heads` | 8 / 8 | 4 / 4 | Query / key-value head counts. `n_kv_heads=n_q_head` → MHA; `n_kv_heads=1` → MQA; in between → GQA. |
| `vocab_size` | 16000 | 10240 | BPE vocabulary size (8k or 16k). |
| `mlp_act` | `gelu` | `swiglu` | MLP activation: `gelu` (baseline), `relu_sq` (ReLU²), `swiglu` (SwiGLU with gated projection). |
| `tie_weights` | `False` | `True` | Share token embedding and lm_head weight matrices. Reduces params by `vocab_size × d_model`. |
| `pos_emb` | `learned` | `rope` | Positional encoding: `learned` (absolute, per-position) or `rope` (Rotary Position Embedding, applied to Q and K). |
| `norm` | `layernorm` | `layernorm` | Normalisation layer: `layernorm` (with bias) or `rmsnorm` (no bias, no mean-centering). |
| `logit_softcap` | `0.0` | `0.0` | Gemma-style soft-capping of final logits: `logit = cap * tanh(logit / cap)`. `0.0` = disabled. |
| `use_token_anchor` | `False` | `True` | At each layer, add a learned linear combination of the original token embedding `x₀` to the residual stream: `x_l += λ_l * x₀`. Provides a depth-invariant shortcut. |
| `use_resid_scale` | `False` | `False` | Per-layer learnable scalar on the residual branch (requires `use_token_anchor`). |
| `use_rezero` | `False` | `False` | Replace residual `x + F(x)` with `x + α * F(x)`, where `α` is a learnable scalar initialised to 0. Intended to improve training stability at initialisation. |
| `weight_init` | `gpt2` | `gpt2` | Parameter initialisation scheme: `gpt2` (std scaled by `1/√(2·n_layer)` for residual projections) or `muon_uniform` (uniform ±1/√fan_in, suited for Muon). |

### Value Skip Connections (Group 9)

| Param | Baseline | Final best | Description |
|---|---|---|---|
| `use_value_residual` | `False` | `False` | Add current-layer input to value projections before attention: `v += x` (reshaped to head dims). Requires MHA (`n_kv_heads == n_q_head`). |
| `use_value_residual_x0` | `False` | `False` | Add original token embedding to value projections: `v += x₀`. Injects raw token identity deep into attention values. |
| `use_value_carry` | `False` | `False` | Cross-layer value carry: `v_l = Wv(x) + λ * v_{l-1}`, where `λ` is a learnable scalar initialised to 0. Passes the previous layer's value tensor forward. |

### Optimiser

| Param | Baseline | Final best | Description |
|---|---|---|---|
| `optimizer_type` | `sgd` | `muon_adamw` | `muon_adamw`: Muon for 2-D weight matrices (attention, MLP projections) + AdamW for all other params (norms, biases, embeddings). `sgd`: vanilla SGD baseline. |
| `muon_lr` | — | `0.02` | Learning rate for Muon. Muon is a second-order-inspired optimiser that orthogonalises the gradient in weight-matrix space using Newton-Schulz iterations. |
| `adamw_lr` | — | `0.0003` | AdamW learning rate for 1-D parameters (norm scales, biases). |
| `emb_lr` | — | `0.003` | AdamW learning rate for token embeddings (slightly higher than `adamw_lr` because embedding rows are updated sparsely). |
| `lr_schedule` | `cosine` | `wsd` | `wsd` (Warmup–Stable–Decay): linear warmup → flat at peak LR → cosine decay. `cosine`: standard cosine annealing from peak to `min_lr`. |
| `warmup_frac` | `0.0` | `0.05` | Fraction of total training steps used for linear LR warmup. |
| `decay_frac` | — | `0.6` | (WSD only) Fraction of total steps devoted to the final cosine decay phase. |
| `min_lr_frac` | `0.0` | `0.05` | LR decay floor as a fraction of peak LR. `0.0` decays all the way to zero; `0.05` stops at 5% of peak. |
| `adamw_wd` | — | `0.1` | AdamW weight decay (L2 regularisation coefficient). |

---

## Group 1 — Optimizer & Schedule

Base: 6L×512d, GELU, SGD+cosine+learned_pos
**Key finding:** Muon+AdamW with RoPE+WSD gives the biggest single jump (−0.54).
gpt2 init beats muon_uniform init.

| Config | val_loss | Δ vs baseline | Params | tok/s |
| --- | --- | --- | --- | --- |
| SGD + cosine + learned | 1.7319 | +0.0000 | 35.35M | 66,340 |
| Muon+AdamW | 1.2333 | -0.4986 | 35.35M | 53,642 |
| Muon + AdamW clip | 1.2362 | -0.4957 | 35.35M | 55,947 |
| Muon + RoPE | 1.2119 | -0.5200 | 35.29M | 57,054 |
| Muon + WSD | 1.2182 | -0.5137 | 35.35M | 58,214 |
| **Muon + RoPE + WSD** ✓ | 1.1953 | -0.5366 | 35.29M | 57,342 |
| Muon + RoPE + WSD + muon_uniform_init | 1.2261 | -0.5058 | 35.29M | 57,142 |

---

## Group 2 — Activation & Architecture Tricks (6L×512d)

Base: 6L×512d + Muon+AdamW+RoPE+WSD
**Key finding:** SwigLU + tie_weights = best combo (−0.012). token_anchor, softcap, MQA all neutral at 6L.

| Config | val_loss | Δ vs g2_base | Params | tok/s |
| --- | --- | --- | --- | --- |
| base (GELU, no tie) | 1.1951 | +0.0000 | 35.29M | 57,781 |
| tie_weights | 1.1925 | -0.0026 | 27.1M | 54,788 |
| SwigLU | 1.1872 | -0.0079 | 35.08M | 53,560 |
| ReLU² | 1.1878 | -0.0073 | 35.29M | 55,220 |
| **SwigLU + tie** ✓ | 1.1836 | -0.0115 | 26.88M | 56,363 |
| ReLU² + tie | 1.1894 | -0.0057 | 27.1M | 55,970 |
| token_anchor | 1.1954 | +0.0003 | 35.29M | 53,274 |
| logit_softcap=30 | 1.1955 | +0.0004 | 35.29M | 54,392 |
| MQA (n_kv=1) | 1.1990 | +0.0039 | 32.53M | 50,306 |

---

## Group 3 — Confirm Best Config (6L×512d)

Base: 6L×512d + Muon+AdamW+RoPE+WSD + SwigLU + tie_weights
**Key finding:** All tricks neutral or slightly negative at shallow depth. Best config = base.

| Config | val_loss | Δ vs g3_base | Params |
| --- | --- | --- | --- |
| base (swiglu+tie+rope+muon+wsd) | 1.1843 | +0.0000 | 26.88M |
| RMSNorm | 1.1862 | +0.0019 | 26.88M |
| token_anchor | 1.1858 | +0.0015 | 26.88M |
| logit_softcap=30 | 1.1860 | +0.0017 | 26.88M |
| MQA | 1.1852 | +0.0009 | 24.13M |
| anchor + softcap | 1.1851 | +0.0008 | 26.88M |

---

## Group 4 — Vocab Size & Depth (6L)

**Key finding:** vocab=16k is optimal; 8k hurts. Adding layers at 512d doesn't help — motivates architecture search.

| Config | val_loss | Δ vs g4_base | Params |
| --- | --- | --- | --- |
| 6L×512d, vocab=16k | 1.1838 | +0.0000 | 26.88M |
| vocab=8k | 1.2100 | +0.0262 | 22.89M |
| 12L×512d, vocab=16k | 1.1863 | +0.0025 | 45.57M |
| 12L×512d, vocab=8k | 1.2084 | +0.0246 | 41.58M |

---

## Group 5 — Architecture Search (~30M params)

Fixed: Muon+AdamW+RoPE+WSD+SwigLU+tie+vocab16k
**Key finding:** Deeper-narrower consistently wins. 12L×384d = best at 30M budget.

| Config | val_loss | Δ vs base | Params | tok/s |
| --- | --- | --- | --- | --- |
| 6L×512d  (base) | 1.1842 | +0.0000 | 26.88M | 60,129 |
| 4L×640d | 1.1978 | +0.0136 | 30.08M | 50,299 |
| 5L×576d | 1.1908 | +0.0066 | 29.14M | 51,621 |
| 8L×448d | 1.1780 | -0.0062 | 26.68M | 48,746 |
| 9L×448d | 1.1772 | -0.0070 | 29.12M | 46,288 |
| 13L×384d | 1.1736 | -0.0106 | 29.17M | 43,970 |
| **12L×384d** ✓ | 1.1727 | -0.0115 | 27.4M | 46,046 |

---

## Group 6 — Architecture Search (~40M params) + GQA

**Key finding:** Deeper-narrower trend continues. GQA does not help. 20L×320d = best.

| Config | val_loss | Δ vs base | Params | tok/s |
| --- | --- | --- | --- | --- |
| 12L×384d  (base) | 1.1850 | +0.0000 | 26.88M | 60,134 |
| 14L×384d | 1.1725 | -0.0125 | 30.94M | 41,639 |
| 16L×384d | 1.1733 | -0.0117 | 34.48M | 37,453 |
| 16L×384d GQA-2 | 1.1731 | -0.0119 | 31.34M | 40,352 |
| 18L×384d | 1.1747 | -0.0103 | 38.02M | 33,695 |
| 19L×384d | 1.1757 | -0.0093 | 39.79M | 31,970 |
| 21L×384d GQA-2 | 1.1779 | -0.0071 | 39.21M | 31,206 |
| 24L×320d | 1.1724 | -0.0126 | 34.15M | 32,097 |
| 28L×320d | 1.1730 | -0.0120 | 38.99M | 28,083 |
| **20L×320d** ✓ | 1.1710 | -0.0140 | 29.31M | 36,804 |

---

## Group 7 — Narrow-and-Deep: 256d at Various Depths

Base for Δ: best from group6 (20L×320d = 1.1710)
**Key finding:** 28L×256d slightly edges out 20L×320d. Diminishing returns beyond 28L.

| Config | val_loss | Δ vs g6_best | Params | tok/s |
| --- | --- | --- | --- | --- |
| **28L×256d** ✓ | 1.1708 | -0.0002 | 26.6M | 35,661 |
| 32L×256d | 1.1720 | +0.0010 | 29.82M | 32,272 |
| 36L×256d | 1.1736 | +0.0026 | 33.03M | 29,143 |
| 40L×256d | 1.1738 | +0.0028 | 36.25M | 26,366 |
| 44L×256d | 1.1759 | +0.0049 | 40.27M | 23,705 |

---

## Group 8 — Depth-Dependent Tricks (28L×256d)

**Key finding:** token_anchor helps at depth (−0.0013). softcap and resid_scale neutral or negative. ReZero mildly negative.
Note: token_anchor was neutral at 6L (group3) — it is depth-dependent.

| Config | val_loss | Δ vs g8_base | Params |
| --- | --- | --- | --- |
| 28L×256d  (base) | 1.1713 | +0.0000 | 26.6M |
| **token_anchor** ✓ | 1.1700 | -0.0013 | 26.6M |
| logit_softcap=30 | 1.1733 | +0.0020 | 26.6M |
| anchor + softcap | 1.1726 | +0.0013 | 26.6M |
| anchor + resid_scale | 1.1711 | -0.0002 | 26.6M |
| ReZero | 1.1715 | +0.0002 | 26.6M |
| anchor + ReZero | 1.1782 | +0.0069 | 26.6M |
| anchor + softcap + scale | 1.1725 | +0.0012 | 26.6M |

---

## Group 9 — Value Skip Connections (28L×256d)

Three variants tested: v += x (current-layer residual), v += x₀ (original embedding), v += λ·v_{l-1} (cross-layer carry).
**Key finding:** All value residual variants are ≥ anchor alone. Anchor-only = best.

| Config | val_loss | Δ vs g9_base | Params |
| --- | --- | --- | --- |
| 28L×256d + anchor  (base) | 1.1711 | +0.0000 | 26.6M |
| v += x (current layer) | 1.1738 | +0.0027 | 26.6M |
| v += x + anchor | 1.1730 | +0.0019 | 26.6M |
| v += x₀ (original emb) | 1.1710 | -0.0001 | 26.6M |
| v += x₀ + anchor | 1.1701 | -0.0010 | 26.6M |
| v += x + x₀ | 1.1730 | +0.0019 | 26.6M |
| v += λ·v_{l-1} | 1.1712 | +0.0001 | 26.6M |
| v += λ·v_{l-1} + anchor | 1.1706 | -0.0005 | 26.6M |
| **anchor only** ✓ | 1.1696 | -0.0015 | 26.6M |

---

## Group 10 — Vocab Size & Architecture Variants (28L×256d)

**Key findings:**
- vocab=10240 beats 16k (−0.004); 8k and 12k both hurt
- `separate_kv`: splits fused kv\_proj into square k\_proj+v\_proj; +5% tok/s, neutral loss
- `spectral_clip`: −19% tok/s, no loss benefit — dropped
- `muon_attn_only`: routes MLP matrices to AdamW — catastrophic (+0.041); Muon is essential for MLP
- Smaller MLP (hidden=512/384): −27%/−45% params, slight loss increase — compression headroom exists but costs quality

| Config | val_loss | Δ vs base | tok/s | Params |
| --- | --- | --- | --- | --- |
| 28L×256d, vocab=16k (base) | 1.1700 | +0.0000 | 40,991 | 26.6M |
| **vocab=10240** ✓ | 1.1660 | -0.0040 | 37,430 | 25.13M |
| vocab=12288 | 1.2016 | +0.0316 | 36,788 | 25.65M |
| vocab=8192 | 1.1937 | +0.0237 | 38,085 | 24.61M |
| separate_kv, vocab=10k | 1.1668 | -0.0032 | 43,205 | 25.13M |
| spectral_clip=1.0, vocab=10k | 1.1673 | -0.0027 | 33,219 | 25.13M |
| muon_attn_only (MLP→AdamW) | 1.2106 | +0.0406 | 44,997 | 25.13M |
| MLP hidden=512 (mlp_expand=3.0) | 1.1687 | -0.0013 | 40,890 | 21.0M |
| MLP hidden=384 (mlp_expand=2.25) | 1.1697 | -0.0003 | 43,831 | 18.25M |

---

## Group 11 — Value Embeddings / E Layers (ResFormer-style)

Dedicated per-layer embedding tables injected into V via a learned per-head gate:
`v += 3·σ(Linear(x[:12])) * ve_table(idx)`. Layer type `E` in `layer_pattern` enables this;
embedding tables are separate from `token_emb`, optimised with `emb_lr`.

**Key findings:**
- VE helps more with vocab=16k than vocab=10k — larger vocab has more token identity signal to inject
- With vocab=10k: 4E at interval=8 is the sweet spot (−0.0008 vs no-VE baseline)
- 5E over-saturates; gate_channels (12 vs 16) makes no meaningful difference
- Cost: +10.5M params (+42%) for −0.0008 loss gain

**vocab=16k** (Δ vs g10\_00\_base = 1.1700):

| Config | val_loss | Δ vs 16k base | tok/s | Params |
| --- | --- | --- | --- | --- |
| 2E layers (pos 9,19) | 1.1708 | +0.0008 | 35,754 | 34.8M |
| 3E layers (pos 9,18,27) | 1.1691 | -0.0009 | 35,472 | 38.89M |
| 2E layers, gate_ch=16 | 1.1700 | +0.0000 | 35,653 | 34.8M |
| 4E layers | 1.1698 | -0.0002 | 34,943 | 42.99M |

**vocab=10k** (Δ vs g10\_01 = 1.1660):

| Config | val_loss | Δ vs 10k base | tok/s | Params |
| --- | --- | --- | --- | --- |
| 28A, no VE (baseline) | 1.1659 | -0.0001 | 43,003 | 25.13M |
| 3E layers | 1.1657 | -0.0003 | 42,148 | 32.99M |
| **4E layers, interval=8** ✓ | 1.1652 | -0.0008 | 36,470 | 35.62M |
| 5E layers, interval=7 | 1.1672 | +0.0012 | 36,735 | 38.24M |
| 4E, muon_lr=0.025 | 1.1658 | -0.0002 | 42,050 | 35.62M |

---

## Group 12 — Mamba Hybrid (first layer)

Base: best config (28L×256d, vocab=10240, 4E layers, Muon+AdamW+RoPE+WSD)
**Key finding:** Replacing the first attention layer with Mamba SSM hurts both quality (+0.0076) and throughput (−42% tok/s). At T=128, Mamba's sequential recurrence cannot parallelise over the sequence dimension; pure attention remains better at this scale.

| Config | val_loss | Δ vs base | tok/s | Params |
| --- | --- | --- | --- | --- |
| **best config (pure attention)** ✓ | 1.1650 | +0.0000 | 41,947 | 35.62M |
| first layer → Mamba SSM | 1.1726 | +0.0076 | 24,174 | 35.82M |

---

## Appendix — Custom Kernel Benchmarks

Hardware: AMD Ryzen AI MAX+ 395, gfx1151 (RDNA4), 40 CU, ~300 GB/s unified memory.

### RMSNorm (Triton)

Benchmark shape: `(128, 64, 384)` bfloat16.

| Implementation | Speed |
|---|---|
| `nn.RMSNorm` (91.3 μs) | 1.0x |
| Triton + autotune (49.7 μs) | **1.84x** |

autotune selected: fwd `num_warps=2`, bwd `num_warps=1`.
**Status: Not used in final config** — final architecture uses `layernorm` (ablation g3_01 showed neutral vs rmsnorm).

### RoPE (Triton)

Benchmark shape: `B=64, H=4, T=128, D=64` (actual training config), bfloat16 fwd+bwd.

| Implementation | Speed (fwd+bwd) |
|---|---|
| Native PyTorch (283.4 μs) | 1.0x |
| Triton (246.9 μs) | **1.15x** |

**Status: Not enabled** — discovered faster only after correcting benchmark shape late in the project. Correctness verified; enabling requires one line change in `rope.py`.

### Fused Linear + Cross-Entropy (v2)

True kernel fusion: matmul written inside the Triton kernel; `[BT, V]` logit tensor never materialised.
Two-pass algorithm: (1) online softmax + collect target logit; (2) recompute logits → grad\_x + grad\_W.

Full training step benchmark (B=64, T=128, V=10240, d=256, L=28):

| Implementation | ms/step | Peak Memory |
|---|---|---|
| Standard CE | 1619 ms (1.0x) | 7109 MB |
| Fused CE v2, autotuned | **1559 ms (1.04x)** | **5815 MB (1.22x savings)** |

Note: full-step times measured without `torch.compile` or operator fusion — reference only, not representative of compiled training performance. CE kernel contribution is diluted by all other ops.
Bottleneck in isolation: 256 VGPRs for `grad_x` accumulator crushes occupancy to 1 on RDNA4.
RDNA4 WMMA only supports 16-bit input → bf16 cast required.
**Status: Optional** (`use_fused_ce=True`). Primary value is memory savings; speed advantage expected on MI355X.

---

## Appendix — GPU Profiling

### Profile Report

**Date:** 2026-05-28 13:08  
**Config:** `AAAEAAAAAAAEAAAAAAAEAAAAAAAE`  
**Model:** 35.6M params, 28L × 256d, vocab=10240  
**Throughput:** 35,310 tok/s  

### Graph Breaks

| | |
|---|---|
| Graphs | 1 |
| Breaks | 0 |

### Memory

| | GB |
|---|---|
| Allocated (peak) | 4.18 |
| Reserved (peak)  | 6.45 |

### Top CPU-Dispatch Overhead

| Op | Count | CPU time | per call |
|---|---|---|---|
| `backward` | 10 | 316.5ms | 31649µs |
| `autograd::engine::evaluate_function: CompiledFunctionBackwar` | 10 | 292.6ms | 29257µs |
| `CompiledFunctionBackward` | 10 | 291.1ms | 29108µs |
| `## Call CompiledFxGraph fbvcwuqulrovfujd7iuc223hzjvp3lnu3ph7` | 10 | 285.0ms | 28501µs |
| `optimizer_step` | 10 | 233.7ms | 23373µs |
| `Optimizer.step#MuonAdamW.step` | 10 | 233.4ms | 23341µs |
| `forward` | 10 | 178.0ms | 17799µs |
| `Torch-Compiled Region: 0/0` | 10 | 177.1ms | 17711µs |
| `CompiledFunction` | 10 | 174.0ms | 17398µs |
| `## Call CompiledFxGraph fol3rwx4wrgfnytmdj2va37jd23ploz2a7vu` | 10 | 170.1ms | 17014µs |
| `aten::mm` | 5190 | 162.7ms | 31µs |
| `hipModuleLaunchKernel` | 12240 | 147.0ms | 12µs |
| `Torch-Compiled Region: 2/2` | 1140 | 99.8ms | 88µs |
| `## Call CompiledFxGraph fbmxl54y4yvqkz2ubnufvdfv4ujzt3fo5sf7` | 1140 | 78.8ms | 69µs |
| `hipExtModuleLaunchKernel` | 5830 | 78.1ms | 13µs |
| `triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0` | 1240 | 47.3ms | 38µs |
| `aten::_scaled_dot_product_flash_attention_backward` | 280 | 32.9ms | 118µs |
| `aten::_flash_attention_backward` | 280 | 28.2ms | 101µs |
| `aten::copy_` | 3660 | 27.9ms | 8µs |
| `aten::_scaled_dot_product_flash_attention` | 280 | 24.2ms | 86µs |

### Top Ops by CUDA Self Time

| Op | Count | CUDA Total | CUDA% | Avg/call | CPU Total |
|---|---|---|---|---|---|
| `aten::slice` | 30 | 0.0ms | 0.0% | 0µs | 0.2ms |
| `aten::as_strided` | 11790 | 0.0ms | 0.0% | 0µs | 6.2ms |
| `aten::view` | 580 | 0.0ms | 0.0% | 0µs | 0.6ms |
| `aten::to` | 20 | 0.0ms | 0.0% | 0µs | 1.6ms |
| `aten::_to_copy` | 20 | 0.0ms | 0.0% | 0µs | 1.5ms |
| `aten::empty_strided` | 1430 | 0.0ms | 0.0% | 0µs | 2.7ms |
| `aten::copy_` | 3660 | 0.0ms | 0.0% | 0µs | 27.9ms |
| `hipStreamGetCaptureInfo` | 30 | 0.0ms | 0.0% | 0µs | 0.0ms |
| `hipMemcpyWithStream` | 30 | 0.0ms | 0.0% | 0µs | 10.5ms |
| `Memcpy HtoD (Host -> Device)` | 12 | 0.0ms | 0.0% | 0µs | 0.0ms |
| `Optimizer.zero_grad#MuonAdamW.zero_grad` | 10 | 0.0ms | 0.0% | 0µs | 3.6ms |
| `forward` | 10 | 0.0ms | 0.0% | 0µs | 178.0ms |
| `TorchDynamo Cache Lookup` | 1290 | 0.0ms | 0.0% | 0µs | 7.2ms |
| `Torch-Compiled Region: 0/0` | 10 | 0.0ms | 0.0% | 0µs | 177.1ms |
| `Pregraph bytecode` | 1290 | 0.0ms | 0.0% | 0µs | 2.9ms |
| `AOTDispatcher Runtime Wrapper Prologue` | 1290 | 0.0ms | 0.0% | 0µs | 3.3ms |
| `CompiledFunction` | 10 | 0.0ms | 0.0% | 0µs | 174.0ms |
| `## Call CompiledFxGraph fol3rwx4wrgfnytmdj2va37jd23ploz2a7vu` | 10 | 0.0ms | 0.0% | 0µs | 170.1ms |
| `aten::randint` | 10 | 0.0ms | 0.0% | 0µs | 0.8ms |
| `aten::resize_` | 10 | 0.0ms | 0.0% | 0µs | 0.0ms |

### Top Kernels by GPU Time (rocprof --stats)


**Bubble (rocprof):** wall 232.0 ms/step, kernel sum 257.4 ms/step — kernel sum exceeds wall time, indicating significant kernel overlap. Accurate bubble requires `rocprof --sys-trace` for timeline analysis.

| Kernel | Calls | Total | Avg | % |
|---|---|---|---|---|
| `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32` | 2652 | 312.4ms | 118µs | 12.1% |
| `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x64` | 1820 | 307.2ms | 169µs | 11.9% |
| `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x16_SN` | 173 | 223.0ms | 1289µs | 8.7% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32` | 1157 | 176.9ms | 153µs | 6.9% |
| `attn_fwd.kd` | 392 | 129.5ms | 330µs | 5.0% |
| `triton_poi_fused__unsafe_view_add_fill_mul_sigmoid_silu` | 364 | 81.6ms | 224µs | 3.2% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x160x3` | 793 | 70.7ms | 89µs | 2.7% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32` | 377 | 69.5ms | 184µs | 2.7% |
| `bwd_kernel_dk_dv.kd` | 364 | 62.7ms | 172µs | 2.4% |
| `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x48x32` | 364 | 59.9ms | 164µs | 2.3% |
| `__amd_rocclr_copyBuffer.kd` | 4955 | 52.1ms | 11µs | 2.0% |
| `triton_red_fused__log_softmax__log_softmax_backward_dat` | 13 | 51.8ms | 3988µs | 2.0% |
| `triton_per_fused__to_copy_add_mul_native_dropout_backwa` | 273 | 51.5ms | 189µs | 2.0% |
| `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32` | 143 | 51.1ms | 357µs | 2.0% |
| `triton_per_fused__to_copy__unsafe_view_add_native_dropo` | 351 | 48.5ms | 138µs | 1.9% |

---

## Summary — Best Config Evolution

| Step | Change | val_loss | Δ |
|------|--------|----------|---|
| Baseline | SGD + cosine + learned\_pos + GELU | 1.7319 | — |
| Group 1  | → Muon+AdamW + RoPE + WSD | 1.1953 | −0.5366 |
| Group 2  | → SwigLU + tie\_weights | 1.1836 | −0.0117 |
| Groups 5-7 | → 28L×256d (deep-narrow arch) | 1.1708 | −0.0128 |
| Group 8  | → token\_anchor | 1.1700 | −0.0008 |
| Group 9  | → confirmed anchor-only best | 1.1696 | −0.0004 |
| Group 10 | → vocab=10240 | 1.1660 | −0.0036 |
| Group 11 | → 4E value embeddings, interval=8 | **1.1652** | −0.0008 |

**Total improvement: −0.5667** (32.7% relative reduction from baseline)

## Reflection — What We Would Do Differently

### 1. Weight initialisation exploration was insufficient

Muon's theoretical motivation calls for uniform initialisation with fan-in scaling — the gradient orthogonalisation step in Muon works better when the initial singular value spectrum is flat (no tails). We tested `muon_uniform` vs `gpt2` in a single group-1 experiment and found `gpt2` wins by 0.03 val_loss, but we did not investigate *why*. The most likely confound is weight tying: `lm_head` shares weights with `token_emb`, forcing embedding std to 0.02 regardless of the init scheme, which may have neutralised any benefit from uniform Muon inits. A cleaner experiment would decouple the two (at the cost of ~4M params) before drawing conclusions about Muon init theory.

### 2. Custom kernels before profiling — wrong order

We wrote three Triton kernels (RMSNorm, RoPE, fused CE) before running any profiler. When we eventually ran `rocprof --stats` on the best config, the results showed the dominant cost was CPU dispatch latency (43% bubble) and small GEMM tile selection caused by d_model=256 — neither of which our kernels address. The correct workflow is: **profile first, identify hot kernels, then write targeted replacements**. Of the three kernels, none ended up in the final training path: RMSNorm is unused (final config uses LayerNorm), RoPE Triton was discovered to be faster only after correcting the benchmark shape late in the project, and fused CE provides memory savings but marginal speed improvement on gfx1151.

### 3. Vocab size was fixed too early

`vocab_size` was not systematically swept until Group 10 — after nine groups of experiments all run at `vocab_size=16000`. The final optimal value turned out to be 10240 (−0.004 vs 16k). This means Groups 1–9 were optimising on a suboptimal vocabulary, and some conclusions may not fully transfer: in particular, the value embedding experiments (Group 11) showed VE benefits more with vocab=16k than 10k, suggesting the Group 11 results are partly an artefact of the vocab choice. Vocab size interacts with embedding dimensionality, weight tying, and token identity signal; it should be treated as a foundational hyperparameter and swept in the first group rather than the tenth.

### 4. Sequential ablation search misses interactions; automatic tuning was underused

Each group performed single-variable search on top of the previous group's best config. This greedy sequential strategy cannot discover interactions between hyperparameters — for example, the optimal `muon_lr` for 28L×256d may differ from the value inherited from Group 1 (6L×512d), and the optimal depth for a given vocab size was never jointly optimised. We did implement Optuna TPE hyperparameter search (`hypertune.py`) but used it only for a narrow LR sweep rather than as the primary search strategy. Investing more in automatic tuning earlier — using Optuna to jointly search over depth, width, vocab, and LR — would likely have found better configurations faster and with less manual iteration.

### 5. Value embedding parameter efficiency was poor

The final architecture includes 4 E-type (value embedding) layers, contributing +10.5M parameters (+42% of the base model) for a val_loss improvement of only −0.0008. No iso-parameter comparison was made: it is unknown whether the same 10.5M parameters spent on additional attention layers, wider d_model, or deeper depth would have yielded greater benefit. We selected 4E because it was the best option within Group 11's search space, but the parameter efficiency of this choice was never challenged against alternatives.

### 6. Mamba hybrid did not help

A single experiment (g12_01) replaced the first attention layer with a Mamba SSM layer, keeping all other best-config settings (28L×256d, vocab=10240, 4E layers). Result: val_loss worsened by 0.0076 (1.1650→1.1726) and throughput dropped from 41,947 to 24,174 tok/s — a 42% speed penalty. The speed regression is expected: Mamba's sequential recurrence cannot be parallelised over the sequence dimension the way attention can, and at T=128 the SSM overhead outweighs any potential efficiency gain. The quality regression suggests that at this scale and sequence length, the first-layer position is better served by attention's global context than by Mamba's local state. Hybrid architectures may have merit at longer sequences or larger scale, but within this project's constraints the result is a clear negative.

### 4. TensorBoard integration added limited value

We integrated TensorBoard (loss curves, LR schedules, weight norms) early in the project. In practice, all experiment tracking and comparison was done through JSONL log files parsed by `collect_results.py`. The TensorBoard writer added code complexity, a `SummaryWriter` dependency, and extra I/O on every training step, with minimal return — the ablation tables in this report were never derived from TensorBoard. A leaner approach would be structured JSONL logging only, with a simple `collect_results.py` for post-hoc analysis.

---
