# Kernel Development Decision Record

Hardware: AMD Ryzen AI MAX+ 395 (gfx1151, RDNA4, 40 CU, 64KB LDS, ~300 GB/s unified)
Migration Target: MI355X (gfx942, CDNA3, 228 CU, 64KB LDS, ~5.3 TB/s HBM3)

---

## 1. RMSNorm

**File:** `kernels/rms_norm.py`
**Tests:** `kernels/tests/test_rms_norm.py`

### Results
| | Speed |
|---|---|
| `nn.RMSNorm` | Baseline |
| Triton (default num_warps=4) | 1.86x |
| Triton + autotune | **3.99x** |

autotune selected fwd `num_warps=2`, bwd `num_warps=1`. RMSNorm is a small kernel, fewer warps → higher occupancy → faster.

### Key Decisions
- `BLOCK = next_power_of_2(N)`, must cover the entire row for reduction, cannot be chunked, not included in autotune config
- `num_warps` is the only meaningful tuning dimension
- dw (weight gradient) reduction is left to PyTorch (`dy * x * rstd` sum over rows), fast enough, not worth writing into the kernel

### Current Status
**Enabled**: `_make_norm()` directly returns `TritonRMSNorm`, replacing `nn.RMSNorm`.

---

## 2. RoPE

**File:** `rope.py` (kernel defined in file, not independent under kernels/)
**Tests:** `kernels/tests/test_rope.py`

### Results
| | Speed (fwd+bwd) |
|---|---|
| Native PyTorch | Baseline |
| Triton fwd+bwd | 0.65x (Slower) |

### Key Decisions
- Forward kernel originally only had forward, no backward → forced to use native during training
- **Backward implementation**: The rotation matrix is an orthogonal matrix, inverse = transpose = run the same kernel with `-sin`. `dx = kernel(dy, cos, -sin)`, diff = 0 (exact)
- Root cause of Triton being slower: tensor is too small (B=128, H=6, T=64, D=64), kernel launch overhead > compute
- autotune **not worth adding**: cannot solve the root problem of launch overhead

### Current Status
**Not Enabled**: `apply_rotary_emb` uses the native path. `_RoPEFn` is kept in the code, correctness verified, waiting to be evaluated when migrating to MI355X or with longer sequences.

---

## 3. Fused Linear CE

Two versions, different design philosophies.

### v1 (Python chunking)
**File:** `kernels/fused_ce.py`
**Tests:** `kernels/tests/test_fused_ce.py`

- Chunk at the Python layer, each chunk calls torch matmul + Triton CE kernel (in-place grad_logits write)
- Avoids the full [BT, V] tensor falling to DRAM
- Benchmark is fwd-only compared to fwd+bwd: **0.38x** (unfair comparison, actually 3x matmul workload vs 1x)

### v2 (tl.dot true kernel fusion)
**File:** `kernels/fused_ce_v2.py`
**Tests:** `kernels/tests/test_fused_ce_v2.py`

matmul is written into the Triton kernel, the [BT, V] logits tensor does not exist anywhere.

#### Algorithm
```
pass 1: scan V, tl.dot compute logits tile → online softmax (m_new = max(m, m_tile), s_new = s*exp(m-m_new) + sum(exp(x-m_new))) + collect target logit
pass 2: recompute logits tile → softmax → grad_logits → grad_x (local) + grad_W (atomic_add)
```

logits are computed twice (the unavoidable cost of not storing [BT,V]).

#### Hardware Limitations (gfx1151)
- RDNA4 WMMA only accepts 16-bit input + fp32 accumulation, **no fp32×fp32 matmul path**
  → dot inputs must be cast to bf16 (range = fp32 8-bit exponent, won't overflow)
- LDS = 64KB, w_tile [BLOCK_V, BLOCK_H] bf16 occupies `BLOCK_V × 512 × 2` bytes
  → BLOCK_V=64 maxes out 64KB, actual limit BLOCK_V ≤ 32, autotune selected **BLOCK_V=16**
- grad_x accumulator [16, 512] fp32 = 256 VGPRs/thread (RDNA4 limit), num_stages=1

#### Autotune Results
| | fwd+bwd Speed | Peak Memory |
|---|---|---|
| Standard CE | 198ms (1.0x) | 990MB |
| v2 Manual BLOCK_V=32 | 618ms (0.33x) | 235MB |
| v2 autotuned | **313ms (0.63x)** | **235MB** |

autotune selected `BLOCK_M=16, BLOCK_V=16, num_warps=4, num_stages=1`.

#### Why is it still slower than standard?
- Standard fwd+bwd = 3 large GEMMs (logits, grad_x, grad_W)
- v2 = 3 full GEMMs + 1 cache-hot partial GEMM (pass 2 "recomputes logits" but only reads x and W without writing out the full [BT, V] tensor).
- If v2 is 313ms vs 198ms (1.58x), the gap to the theoretical ~1.33x is likely RDNA WMMA utilization limits rather than pure instruction counts. (maybe?)
- **atomic_add vs occupancy**: We previously suspected 512 programs competing on `atomic_add` was the bottleneck. However, atomic conflicts are cache-line based, not program based. It is highly probable that the 256 VGPR requirement for the `grad_x` accumulator is crushing occupancy down to 1, acting as the true bottleneck.

#### 3-kernel design (abandoned after discussion)
Proposal: Kernel1 (loss) + Kernel2 (grad_x) + Kernel3 (grad_W) to eliminate atomic.
**Reason for abandonment**: logits would be computed 3 times instead of 2, one extra full W scan, net negative return.

### Current Status
**Optionally Enabled**: `use_fused_ce=True` in `Hyperparameters` (Default False).
Primary value is in **Memory** (4.21x), speed-wise gfx1151 doesn't have an advantage.

---

## 4. MI355X Migration Checklist

| Operation | File | Description |
|---|---|---|
| Uncomment CDNA3 configs | `kernels/fused_ce_v2.py` L17-21 | BLOCK_V=64/128, num_stages=2 |
| Remove bf16 cast | `kernels/fused_ce_v2.py` | MFMA supports fp32 dot, but throughput is often 1/4 to 1/2 of bf16/fp16. **Recommendation:** Keep both configs and let autotune decide. |
| Re-evaluate RMSNorm baseline | `kernels/rms_norm.py` | CDNA3 PyTorch baseline (via hipBLASLt/CK) will be much stronger. The current 4x speedup may drop to 1.5-2x. |
| Evaluate RoPE Triton | `rope.py` | HBM3 bandwidth 18x, might turn the tables at large T |
| Rerun autotune | All kernels | Run automatic search on new hardware for the first time |

---

## 5. Kernel Precision Details

| Kernel | Source of Error | Max Error |
|---|---|---|
| RMSNorm fwd (fp32) | Triton tl.sum accumulation order | ~1e-6 |
| RMSNorm fwd (bf16) | bf16 precision | ~8e-3 |
| RoPE fwd (fp32) | None (exact) | ~2e-7 |
| RoPE bwd | None (exact, same kernel) | 0.00 |
| Fused CE fwd | bf16 matmul vs fp32 ref | ~5e-3 |
| Fused CE bwd dx/dW | bf16 accumulation order across tiles | ~2e-3 |

---

## 6. TODOs / Next Steps

- **Fused CE "Tokens/Sec at same budget" test**: Run an experiment measuring actual throughput. If the 4.2x memory savings allows us to increase batch size and ultimately wins on `tokens/sec`, flip `use_fused_ce` default to `True`.
- **Profile Fused CE on gfx1151**: Run `rocprof` to check the `SQ_INSTS_VALU` vs `SQ_INSTS_LDS` ratio to definitively confirm if the bottleneck is `atomic_add` conflicts or VGPR-induced low occupancy.
