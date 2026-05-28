# Kernel Development Decision Record

Hardware: AMD Ryzen AI MAX+ 395 (gfx1151, RDNA4, 40 CU, 64KB LDS, ~300 GB/s unified)
Migration Target: MI355X (gfx942, CDNA3, 228 CU, 64KB LDS, ~5.3 TB/s HBM3)

---

## 1. RMSNorm

**File:** `kernels/rms_norm.py`
**Tests:** `kernels/tests/test_rms_norm.py`

### Results

Benchmark shape: `(128, 64, 384)` bfloat16

| | Speed |
|---|---|
| `nn.RMSNorm` (91.3 μs) | 1.0x |
| Triton + autotune (49.7 μs) | **1.84x** |

### Key Decisions
- `BLOCK = next_power_of_2(N)`, must cover the entire row for reduction, cannot be chunked
- `num_warps` is the only meaningful tuning dimension; autotune selected fwd `num_warps=2`, bwd `num_warps=1`
- dw gradient reduction left to PyTorch — fast enough, not worth writing into kernel

### Current Status
**Kernel available but not used in final config.** Ablation (g3_01) showed RMSNorm vs LayerNorm is neutral at this scale; final architecture uses `norm: layernorm`. Re-evaluate if switching to RMSNorm.

---

## 2. RoPE

**File:** `rope.py`
**Tests:** `kernels/tests/test_rope.py`

### Results

Benchmark shape: `B=64, H=4, T=128, D=64` (actual training config, bfloat16 fwd+bwd)

| | Speed (fwd+bwd) |
|---|---|
| Native PyTorch (283.4 μs) | 1.0x |
| Triton (246.9 μs) | **1.15x** |

Note: earlier result of **0.65x** was measured at wrong shape (B=128, H=6, T=64, D=64). At the correct training shape, Triton is faster.

### Key Decisions
- Backward: rotation matrix is orthogonal → inverse = transpose = same kernel with `-sin`. Exact, diff = 0.
- At correct training shape, Triton wins by 1.15x — kernel launch overhead is less dominant with T=128 vs T=64.

### Current Status
**Not Enabled** — discovered faster only after correcting benchmark shape. Time budget didn't allow re-integration and re-validation. `_RoPEFn` is in code and correct; enabling requires swapping one line in `rope.py`.

---

## 3. Fused Linear CE

Two versions, different design philosophies.

### v1 (Python chunking)
**File:** `kernels/fused_ce.py`

- Python-level chunking; matmul done by torch, CE kernel overwrites logits with grad_logits
- `[chunk, V]` still hits DRAM; benchmark fwd-only vs fwd+bwd: **0.38x** (unfair comparison)

### v2 (tl.dot true kernel fusion)
**File:** `kernels/fused_ce_v2.py`
**Tests:** `kernels/tests/test_fused_ce_v2.py`

matmul lives inside the Triton kernel. The `[BT, V]` logit tensor never exists anywhere.

#### Algorithm
```
pass 1: tl.dot → logits tile → online softmax + collect target logit
pass 2: tl.dot → logits tile → softmax → grad_x (local) + grad_W (atomic_add)
```
logits recomputed in pass 2 (cheaper than spilling `[BLOCK_M, V]` to DRAM).

#### Hardware Limitations (gfx1151)
- RDNA4 WMMA: 16-bit input only → dot inputs cast to bf16
- LDS 64KB → BLOCK_V ≤ 32; autotune selected **BLOCK_V=16**
- grad_x accumulator `[16, H]` fp32 = 256 VGPRs/thread → occupancy=1

#### Results

Full training step benchmark (B=64, T=128, V=10240, d=256, L=28):

| | ms/step | Peak Memory | Speedup |
|---|---|---|---|
| Standard CE | 1619 ms | 7109 MB | 1.0x |
| Fused CE v2 | 1559 ms | 5815 MB | **1.04x** |
| Memory savings | | | **1.22x** |

Note: measured without `torch.compile` or operator fusion — reference only. CE kernel contribution is diluted by all other ops; real compiled training impact requires rocprof comparison.

#### Why is CE kernel still slower in isolation?
- 256 VGPRs for grad_x accumulator crushes occupancy to 1 on RDNA4
- Standard CE = 3 large GEMMs via rocBLAS (optimal tile selection); v2 = custom tiled loop

### Current Status
**Optionally enabled** (`use_fused_ce=True`, default False). Primary value is **memory (1.22x)**, which could allow larger batch size. Speed benefit at full-step level is marginal (1.04x).

---

## 4. MI355X Migration Checklist

| Operation | File | Description |
|---|---|---|
| Uncomment CDNA3 configs | `kernels/fused_ce_v2.py` L34-39 | BLOCK_V=64/128, num_stages=2 |
| Keep bf16 cast or autotune | `kernels/fused_ce_v2.py` | MFMA supports fp32 but bf16 throughput often 2-4x higher; let autotune decide |
| Re-evaluate RMSNorm | `kernels/rms_norm.py` | CDNA3 baseline much stronger; 1.84x may drop |
| Enable RoPE Triton | `rope.py` | Already 1.15x on gfx1151; HBM3 bandwidth likely widens gap |
| Rerun autotune | All kernels | Cold run on new hardware |

---

## 5. Kernel Precision Details

| Kernel | Source of Error | Max Error |
|---|---|---|
| RMSNorm fwd (fp32) | tl.sum accumulation order | ~1e-6 |
| RMSNorm fwd (bf16) | bf16 precision | ~8e-3 |
| RoPE fwd (fp32) | None (exact) | ~2e-7 |
| RoPE bwd | None (exact, same kernel with -sin) | 0.00 |
| Fused CE fwd | bf16 matmul vs fp32 ref | ~5e-3 |
| Fused CE bwd dx/dW | bf16 accumulation order across tiles | ~2e-3 |

---

## 6. TODOs

- **Enable RoPE Triton**: swap `_apply_rope_native` → `_RoPEFn.apply` in `rope.py`. Correctness verified, 1.15x faster at training shape.
- **Fused CE batch-size scaling test**: 1.22x memory savings → can increase batch size → measure net tok/s gain.
- **Profile Fused CE with rocprof**: run `task rocprof-remote` with `use_fused_ce=True` to confirm VGPR occupancy bottleneck via `SQ_INSTS_VALU` ratio.
