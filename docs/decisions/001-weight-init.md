# 001 — Weight Initialization Strategy

**Status**: concluded — `gpt2` wins
**Hyperparameter**: `weight_init: str`
**Options**: `"gpt2"` (default) | `"muon_uniform"`

---

## Options

### `gpt2`
`Normal(0, 0.02)` for all weights, zeros for biases. Standard GPT-2 style.

### `muon_uniform`
`Uniform[-1/√fan_in, 1/√fan_in]` for Muon-optimized matrices, zeros for residual exits, larger embedding std.

---

## Experiment

Setup: 6L×512d + Muon+AdamW + RoPE + WSD

| Run | `weight_init` | val_loss | Δ |
|---|---|---|---|
| g1_11_muon_rope_wsd | `gpt2` | **1.1953** | — |
| g1_12_muon_rope_init_wsd | `muon_uniform` | 1.2261 | +0.0308 |

**Conclusion**: `gpt2` wins by 0.03. Root cause unclear — `gpt2` is kept as default.
