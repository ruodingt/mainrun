"""
RMSNorm kernel tests: correctness + benchmark.

Run:
    pytest kernels/tests/test_rms_norm.py -v
    python kernels/tests/test_rms_norm.py        # runs benchmark too
"""
import torch
import pytest
from torch.utils.benchmark import Timer

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from kernels.rms_norm import TritonRMSNorm


def _ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rstd = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * rstd * w.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,D", [
    ((128, 64, 384), 384),   # training shape (B, T, D)
    ((8192,    384), 384),   # flattened
    ((4,  32,  512), 512),   # different D
    ((1,   1,  256), 256),   # edge case
])
def test_forward(shape, D):
    """Correctness in float32 — precise comparison."""
    device = "cuda"
    x    = torch.randn(*shape, device=device, dtype=torch.float32)
    norm = TritonRMSNorm(D).to(device=device)  # float32 weights

    ref = _ref(x, norm.weight)
    out = norm(x)

    diff = (ref - out).abs()
    print(f"  [fwd fp32] shape={shape}  max={diff.max():.2e}  mean={diff.mean():.2e}")
    torch.testing.assert_close(ref, out, atol=1e-5, rtol=0,
                                msg=f"forward mismatch for shape={shape}")


@pytest.mark.parametrize("D", [256, 384, 512])
def test_forward_bf16(D):
    """bf16 smoke test — relaxed tolerance due to 7-bit mantissa."""
    device = "cuda"
    x    = torch.randn(128, 64, D, device=device, dtype=torch.bfloat16)
    norm = TritonRMSNorm(D).to(device=device, dtype=torch.bfloat16)

    ref = _ref(x, norm.weight)
    out = norm(x)

    diff = (ref - out).abs()
    print(f"  [fwd bf16] D={D}  max={diff.max():.2e}  mean={diff.mean():.2e}")
    # bf16 relative error ~0.4%, atol=0.02 is generous but intentional
    torch.testing.assert_close(ref, out, atol=2e-2, rtol=0)


def test_backward():
    device = "cuda"
    B, T, D = 4, 16, 384
    # Use float32 for precise grad comparison
    x    = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    norm = TritonRMSNorm(D).to(device=device)

    # Triton backward
    norm(x).sum().backward()
    dx_triton = x.grad.clone()
    dw_triton = norm.weight.grad.clone()

    # Reference backward
    x.grad = None
    norm.weight.grad = None
    _ref(x, norm.weight).sum().backward()

    dx_diff = (dx_triton - x.grad).abs()
    dw_diff = (dw_triton - norm.weight.grad).abs()
    print(f"  [bwd dx]  max={dx_diff.max():.2e}  mean={dx_diff.mean():.2e}")
    print(f"  [bwd dw]  max={dw_diff.max():.2e}  mean={dw_diff.mean():.2e}")
    torch.testing.assert_close(dx_triton, x.grad,           atol=1e-4, rtol=0)
    torch.testing.assert_close(dw_triton, norm.weight.grad,  atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark():
    device = "cuda"
    B, T, D = 128, 64, 384
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

    triton_norm = TritonRMSNorm(D).to(device=device, dtype=torch.bfloat16)
    torch_norm  = torch.nn.RMSNorm(D).to(device=device, dtype=torch.bfloat16)

    # Warm up
    for _ in range(10):
        triton_norm(x); torch_norm(x)
    torch.cuda.synchronize()

    t_triton = Timer("norm(x); torch.cuda.synchronize()",
                     globals={"norm": triton_norm, "x": x, "torch": torch}).blocked_autorange()
    t_torch  = Timer("norm(x); torch.cuda.synchronize()",
                     globals={"norm": torch_norm,  "x": x, "torch": torch}).blocked_autorange()

    print(f"\n{'─'*40}")
    print(f"Shape: {(B, T, D)}  dtype: bfloat16")
    print(f"TritonRMSNorm : {t_triton.median * 1e6:.1f} μs")
    print(f"nn.RMSNorm    : {t_torch.median  * 1e6:.1f} μs")
    print(f"Speedup       : {t_torch.median / t_triton.median:.2f}x")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    test_forward((128, 64, 384), 384)
    test_forward((8192, 384), 384)
    test_forward_bf16(384)
    test_backward()
    print("✓ all correctness tests passed")
    benchmark()
