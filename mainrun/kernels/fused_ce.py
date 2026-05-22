"""
Fused Linear + Cross-Entropy kernel.

Standard path:  logits = x @ W.T  →  store [BT, V]  →  CE  →  grad_logits  →  grad_x
Fused path:     chunk over BT, compute logits per chunk, CE kernel overwrites logits
                with grad_logits in-place. [BT, V] never lands in DRAM.

Memory: O(chunk_size * V) instead of O(BT * V).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ce_kernel(
    Logits,      # [n_rows, V] float32 — overwritten with grad_logits
    Targets,     # [n_rows] int64
    Loss,        # [n_rows] float32 — per-token CE loss (unnormalized)
    stride_l,    # stride of Logits (= V for contiguous)
    V,           # vocab size
    n_tokens,    # non-ignored tokens (for mean reduction)
    ignore_index,
    BLOCK: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < V
    off  = row * stride_l

    target = tl.load(Targets + row).to(tl.int32)

    # Ignored token: zero out grad and loss, return
    if target == ignore_index:
        tl.store(Loss + row, 0.0)
        tl.store(Logits + off + cols, 0.0, mask=mask)
        return

    # Load logits
    x = tl.load(Logits + off + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # Numerically stable softmax
    m     = tl.max(x, axis=0)
    exp_x = tl.where(mask, tl.exp(x - m), 0.0)
    s     = tl.sum(exp_x)

    # Per-token loss
    target_logit = tl.load(Logits + off + target).to(tl.float32)
    tl.store(Loss + row, tl.log(s) + m - target_logit)

    # Grad logits = (softmax - one_hot(target)) / n_tokens
    inv_n = 1.0 / n_tokens
    grad  = (exp_x / s - tl.where(cols == target, 1.0, 0.0)) * inv_n
    tl.store(Logits + off + cols, grad, mask=mask)


class FusedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, targets, ignore_index=-100):
        BT, H = x.shape
        V     = W.shape[0]
        BLOCK = triton.next_power_of_2(V)

        # Chunk size: keep peak logits tensor ~ same memory as input
        inc_factor = max(1, triton.cdiv(V, H))
        chunk_size = triton.next_power_of_2(max(1, triton.cdiv(BT, inc_factor)))

        n_tokens = max(1, (targets != ignore_index).sum().item())

        grad_x   = torch.zeros_like(x)
        grad_W   = torch.zeros_like(W, dtype=torch.float32)
        loss_1d  = torch.zeros(BT, device=x.device, dtype=torch.float32)

        x_f = x.float()
        W_f = W.float()

        for start in range(0, BT, chunk_size):
            end    = min(start + chunk_size, BT)
            n_rows = end - start
            chunk  = slice(start, end)

            logits = (x_f[chunk] @ W_f.T).contiguous()  # [n_rows, V]

            _ce_kernel[(n_rows,)](
                logits, targets[chunk], loss_1d[chunk],
                logits.stride(0), V, n_tokens, ignore_index, BLOCK,
            )

            # logits is now grad_logits
            grad_x[chunk] = (logits @ W_f).to(x.dtype)
            grad_W       += logits.T @ x_f[chunk]

        ctx.save_for_backward(grad_x, grad_W.to(W.dtype))
        return loss_1d.sum() / n_tokens

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_W = ctx.saved_tensors
        return grad_x * grad_output, grad_W * grad_output, None, None


def fused_linear_ce(x: torch.Tensor, W: torch.Tensor,
                    targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return FusedLinearCEFunction.apply(x, W, targets, ignore_index)
