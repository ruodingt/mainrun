"""
Triton RMSNorm kernel — forward + backward.
Drop-in replacement for nn.RMSNorm.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _fwd_kernel(
    X, W, Y, Rstd,
    stride,
    N:    tl.constexpr,
    eps:  tl.constexpr,
    BLOCK: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off  = row * stride

    x_orig = tl.load(X + off + cols, mask=mask, other=0.0)
    x      = x_orig.to(tl.float32)
    w      = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x) / N + eps)
    tl.store(Rstd + row, rstd)
    tl.store(Y + off + cols, (x * rstd * w).to(x_orig.dtype), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _bwd_dx_kernel(
    DX, DY, X, W, Rstd,
    stride,
    N:    tl.constexpr,
    BLOCK: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off  = row * stride

    dy_orig = tl.load(DY + off + cols, mask=mask, other=0.0)
    dy   = dy_orig.to(tl.float32)
    x    = tl.load(X  + off + cols, mask=mask, other=0.0).to(tl.float32)
    w    = tl.load(W  + cols,       mask=mask, other=1.0).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)

    xhat = x * rstd
    wdy  = w * dy
    # d/dx RMSNorm: (w*dy - xhat * dot(w*dy, xhat)/N) * rstd
    dx   = (wdy - xhat * tl.sum(wdy * xhat) / N) * rstd
    tl.store(DX + off + cols, dx.to(dy_orig.dtype), mask=mask)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        x2d   = x.reshape(-1, shape[-1]).contiguous()
        M, N  = x2d.shape
        BLOCK = triton.next_power_of_2(N)

        y    = torch.empty_like(x2d)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        _fwd_kernel[(M,)](x2d, w, y, rstd, x2d.stride(0), N, eps, BLOCK)

        ctx.save_for_backward(x2d, w, rstd)
        ctx.shape, ctx.N, ctx.BLOCK = shape, N, BLOCK
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, w, rstd = ctx.saved_tensors
        N, BLOCK = ctx.N, ctx.BLOCK
        dy2d = dy.reshape(-1, N).contiguous()
        M    = x2d.shape[0]

        dx = torch.empty_like(x2d)
        _bwd_dx_kernel[(M,)](dx, dy2d, x2d, w, rstd, x2d.stride(0), N, BLOCK)

        # dw: accumulate over rows — reduction is cheap, keep in torch
        dw = (dy2d.float() * x2d.float() * rstd.unsqueeze(1)).sum(0).to(w.dtype)
        return dx.reshape(ctx.shape), dw, None


class TritonRMSNorm(nn.Module):
    """RMSNorm backed by Triton kernels. Same interface as nn.RMSNorm."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNormFn.apply(x, self.weight, self.eps)
