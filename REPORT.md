# Ablation Study Report

**Goal:** Minimise validation loss on Hacker News titles (100k, 7 epochs, seed=1337).
**Baseline:** SGD + cosine LR + learned pos emb + GELU + 6L×512d×16000 → **val_loss = 1.7319**
**Final best:** 28L×256d×16000 + Muon+AdamW + WSD + RoPE + SwigLU + tie_weights + token_anchor → **val_loss = 1.1696**

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
| `vocab_size` | 16000 | 16000 | BPE vocabulary size (8k or 16k). |
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
| `use_value_residual` | — | `False` | Add current-layer input to value projections before attention: `v += x` (reshaped to head dims). Requires MHA (`n_kv_heads == n_q_head`). |
| `use_value_residual_x0` | — | `False` | Add original token embedding to value projections: `v += x₀`. Injects raw token identity deep into attention values. |
| `use_value_carry` | — | `False` | Cross-layer value carry: `v_l = Wv(x) + λ * v_{l-1}`, where `λ` is a learnable scalar initialised to 0. Passes the previous layer's value tensor forward. |

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

Sweeping vocab size around the 16k optimum, plus architecture experiments.

**Key findings:**
- vocab=10240 beats 16k (−0.004); 8k and 12k both hurt
- `separate_kv`: splits fused kv_proj into k_proj+v_proj; +5% tok/s, neutral loss
- `spectral_clip`: −19% tok/s, no loss benefit — dropped
- `muon_attn_only`: routes MLP matrices to AdamW instead of Muon — catastrophic (+0.041)
- Smaller MLP (hidden=512/384): −27%/−45% params, +0.003/+0.004 loss — MLP has compression headroom but at a cost

| Config | val_loss | Δ vs base | tok/s | Params |
| --- | --- | --- | --- | --- |
| 28L×256d, vocab=16k (base) | 1.1700 | +0.0000 | 40,991 | 26.6M |
| **vocab=10240** ✓ | 1.1660 | −0.0040 | 37,430 | 25.1M |
| vocab=12288 | 1.2016 | +0.0316 | 36,788 | 25.7M |
| vocab=8192 | 1.1937 | +0.0237 | 38,085 | 24.6M |
| separate_kv, vocab=10k | 1.1668 | −0.0032 | 43,205 | 25.1M |
| spectral_clip, vocab=10k | 1.1673 | −0.0027 | 33,219 | 25.1M |
| muon_attn_only | 1.2106 | +0.0406 | 44,997 | 25.1M |
| MLP hidden=512 (mlp_expand=3.0) | 1.1687 | −0.0013 | 40,890 | 21.0M |
| MLP hidden=384 (mlp_expand=2.25) | 1.1697 | −0.0003 | 43,831 | 18.2M |

---

## Group 11 — Value Embeddings (ResFormer-style E layers)

Dedicated per-layer embedding tables injected into V via a learned per-head gate: `v += 3·σ(gate(x[:12])) * ve_table(idx)`. Controlled by `layer_pattern` — `E` = attention with value embedding, `A` = standard attention. Value embed tables are separate from `token_emb` and use `emb_lr`.

**Key findings:**
- VE helps more with vocab=16k than vocab=10k (larger vocab → more token identity information to inject)
- With vocab=10k, 4E layers at interval 8 is the sweet spot: −0.0008 vs no-VE baseline
- 5E layers over-saturates (worse)
- Gate channels (12 vs 16) make no meaningful difference
- Cost: +10.5M params (+42%) for −0.0008 loss

**vocab=16k experiments** (base: g10_00_base 1.1700):

| Config | val_loss | Δ vs base | tok/s | Params |
| --- | --- | --- | --- | --- |
| 28L all-A, vocab=16k (base) | 1.1700 | +0.0000 | 40,991 | 26.6M |
| 2E layers | 1.1708 | +0.0008 | 35,754 | 34.8M |
| 3E layers | 1.1691 | −0.0009 | 35,472 | 38.9M |
| 4E layers | 1.1698 | +0.0002 | 34,943 | 43.0M |
| 2E, gate_ch=16 | 1.1700 | +0.0000 | 35,653 | 34.8M |

**vocab=10k experiments** (base: g10_01 1.1660):

| Config | val_loss | Δ vs base | tok/s | Params |
| --- | --- | --- | --- | --- |
| 28L all-A, vocab=10k (base) | 1.1660 | +0.0000 | 37,430 | 25.1M |
| 3E layers | 1.1657 | −0.0003 | 42,148 | 33.0M |
| **4E layers, interval=8** ✓ | **1.1652** | **−0.0008** | 36,470 | 35.6M |
| 5E layers, interval=7 | 1.1672 | +0.0012 | 36,735 | 38.2M |

---

## Summary — Best Config Evolution

| Step | Change | val_loss | Δ |
|------|--------|----------|---|
| Baseline | SGD + cosine + learned_pos + GELU | 1.7319 | — |
| Group 1  | → Muon+AdamW + RoPE + WSD | 1.1953 | −0.5366 |
| Group 2  | → SwigLU + tie_weights | 1.1836 | −0.0117 |
| Groups 5-7 | → 28L×256d (deep-narrow arch) | 1.1708 | −0.0128 |
| Group 8  | → token_anchor | 1.1700 | −0.0008 |
| Group 9  | → confirmed anchor-only best | 1.1696 | −0.0004 |
| Group 10 | → vocab=10240 | 1.1660 | −0.0036 |
| Group 11 | → 4E value embeddings, interval=8 | **1.1652** | −0.0008 |

**Total improvement: −0.5667** (32.7% relative reduction from baseline)
