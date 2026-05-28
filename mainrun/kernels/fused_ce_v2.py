"""
True kernel-fused Linear + Cross-Entropy.

Difference from fused_ce.py:
    v1  — Python-level chunking; matmul done by torch (16 small GEMMs), CE kernel
          only overwrites logits with grad_logits. [chunk, V] still hits DRAM.
    v2  — matmul lives INSIDE the Triton kernel via tl.dot. logits are tiled over
          V and consumed on the fly (online softmax). The [BT, V] tensor never
          exists anywhere — not in DRAM, not even fully in registers.

One program handles BLOCK_M rows. It walks V twice:
    pass 1: tl.dot → logits tile → online (max, sum_exp) + gather target logit
    pass 2: tl.dot → logits tile → softmax → grad_logits
            grad_x  += grad_logits   @ W_tile        (per-row, local)
            grad_W  += grad_logits.T @ x_block       (shared → atomic_add)

logits are recomputed in pass 2 (cheaper than spilling [BLOCK_M, V] to DRAM).
BLOCK_H is fixed to next_power_of_2(H) — not autotuned since it is determined by H.
BLOCK_M / BLOCK_V are autotuned per (H, V).  RDNA4 configs stay ≤ BLOCK_V=32 to
fit within 64 KB LDS.  MI355X / CDNA3 configs (commented out) allow larger tiles.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # ── RDNA4 (gfx1151) — LDS 64 KB, WMMA 16×16, wave32 ──
        triton.Config({"BLOCK_M": 16, "BLOCK_V": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_V": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_V": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_V": 32}, num_warps=8, num_stages=1),
        # ── CDNA3 (gfx942, MI300X / MI355X) — larger tiles, pipeline ──
        # triton.Config({"BLOCK_M": 16, "BLOCK_V": 64},  num_warps=4,  num_stages=2),
        # triton.Config({"BLOCK_M": 16, "BLOCK_V": 64},  num_warps=8,  num_stages=2),
        # triton.Config({"BLOCK_M": 16, "BLOCK_V": 128}, num_warps=8,  num_stages=2),
        # triton.Config({"BLOCK_M": 32, "BLOCK_V": 64},  num_warps=8,  num_stages=2),
        # triton.Config({"BLOCK_M": 32, "BLOCK_V": 128}, num_warps=16, num_stages=2),
    ],
    key=["H", "V"],
    reset_to_zero=["GradW"],   # atomic_add accumulates — zero between timing runs
)
@triton.jit
def _fused_ce_kernel(
    X, W, Targets, Loss, GradX, GradW,
    stride_xm, stride_xh,
    stride_wv, stride_wh,
    stride_gxm, stride_gxh,
    stride_gwv, stride_gwh,
    n_tokens, ignore_index,
    BT, H, V,
    logit_softcap,             # float, 0.0 = disabled
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid      = tl.program_id(0)
    rows     = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < BT

    h_idx  = tl.arange(0, BLOCK_H)
    h_mask = h_idx < H

    x_ptrs  = X + rows[:, None] * stride_xm + h_idx[None, :] * stride_xh
    x_block = tl.load(x_ptrs, mask=row_mask[:, None] & h_mask[None, :], other=0.0)
    xbf16   = x_block.to(tl.bfloat16)

    targets = tl.load(Targets + rows, mask=row_mask, other=ignore_index)
    ignored = targets == ignore_index

    # ---------------- pass 1: softmax statistics ----------------
    m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
    s_i = tl.zeros([BLOCK_M], tl.float32)
    tgt = tl.zeros([BLOCK_M], tl.float32)

    for v0 in range(0, V, BLOCK_V):
        v_idx  = v0 + tl.arange(0, BLOCK_V)
        v_mask = v_idx < V
        w_ptrs = W + v_idx[:, None] * stride_wv + h_idx[None, :] * stride_wh
        wbf16  = tl.load(w_ptrs, mask=v_mask[:, None] & h_mask[None, :], other=0.0).to(tl.bfloat16)

        logits = tl.dot(xbf16, tl.trans(wbf16))
        if logit_softcap != 0.0:
            logits = logit_softcap * 2.0 * tl.sigmoid(2.0 * (logits / logit_softcap)) - 1.0
        logits = tl.where(v_mask[None, :], logits, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        s_i   = s_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(logits - m_new[:, None]), axis=1)
        m_i   = m_new

        is_tgt = v_idx[None, :] == targets[:, None]
        tgt   += tl.sum(tl.where(is_tgt, logits, 0.0), axis=1)

    loss = tl.where(ignored, 0.0, tl.log(s_i) + m_i - tgt)
    tl.store(Loss + rows, loss, mask=row_mask)

    # ---------------- pass 2: gradients ----------------
    inv_n  = 1.0 / n_tokens
    grad_x = tl.zeros([BLOCK_M, BLOCK_H], tl.float32)

    for v0 in range(0, V, BLOCK_V):
        v_idx  = v0 + tl.arange(0, BLOCK_V)
        v_mask = v_idx < V
        w_ptrs = W + v_idx[:, None] * stride_wv + h_idx[None, :] * stride_wh
        wbf16  = tl.load(w_ptrs, mask=v_mask[:, None] & h_mask[None, :], other=0.0).to(tl.bfloat16)

        logits = tl.dot(xbf16, tl.trans(wbf16))
        if logit_softcap != 0.0:
            z      = logit_softcap * 2.0 * tl.sigmoid(2.0 * (logits / logit_softcap)) - 1.0
            p      = tl.exp(z - m_i[:, None]) / s_i[:, None]
            is_tgt = v_idx[None, :] == targets[:, None]
            g_z    = (p - tl.where(is_tgt, 1.0, 0.0)) * inv_n
            # chain rule through tanh: d(cap*tanh(x/cap))/dx = 1 - tanh²(x/cap) = 1 - (z/cap)²
            g      = g_z * (1.0 - (z / logit_softcap) * (z / logit_softcap))
        else:
            p      = tl.exp(logits - m_i[:, None]) / s_i[:, None]
            is_tgt = v_idx[None, :] == targets[:, None]
            g      = (p - tl.where(is_tgt, 1.0, 0.0)) * inv_n
        g      = tl.where(v_mask[None, :] & ~ignored[:, None], g, 0.0)

        grad_x += tl.dot(g.to(tl.bfloat16), wbf16)

        gw      = tl.dot(tl.trans(g).to(tl.bfloat16), xbf16)
        gw_ptrs = GradW + v_idx[:, None] * stride_gwv + h_idx[None, :] * stride_gwh
        tl.atomic_add(gw_ptrs, gw, mask=v_mask[:, None] & h_mask[None, :])

    gx_ptrs = GradX + rows[:, None] * stride_gxm + h_idx[None, :] * stride_gxh
    tl.store(gx_ptrs, grad_x.to(GradX.dtype.element_ty),
             mask=row_mask[:, None] & h_mask[None, :])


class FusedLinearCEv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, targets, ignore_index, logit_softcap):
        BT, H = x.shape
        V     = W.shape[0]
        BLOCK_H = triton.next_power_of_2(H)

        n_tokens = max(1, (targets != ignore_index).sum().item())

        loss   = torch.zeros(BT, device=x.device, dtype=torch.float32)
        grad_x = torch.empty_like(x, dtype=torch.float32)
        grad_W = torch.zeros_like(W, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(BT, meta["BLOCK_M"]),)
        _fused_ce_kernel[grid](
            x, W, targets, loss, grad_x, grad_W,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            grad_W.stride(0), grad_W.stride(1),
            n_tokens, ignore_index,
            BT, H, V,
            float(logit_softcap),
            BLOCK_H=BLOCK_H,
        )

        ctx.save_for_backward(grad_x.to(x.dtype), grad_W.to(W.dtype))
        return loss.sum() / n_tokens

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_W = ctx.saved_tensors
        return grad_x * grad_output, grad_W * grad_output, None, None, None


def fused_linear_ce_v2(x: torch.Tensor, W: torch.Tensor,
                       targets: torch.Tensor,
                       ignore_index: int = -100,
                       logit_softcap: float = 0.0) -> torch.Tensor:
    return FusedLinearCEv2.apply(x, W, targets, ignore_index, logit_softcap)
