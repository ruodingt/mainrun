"""
RoPE kernel tests: correctness (fwd + bwd) + benchmark.

Run:
    pytest kernels/tests/test_rope.py -v
    python kernels/tests/test_rope.py
"""
import torch
import pytest
from torch.utils.benchmark import Timer

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from rope import _apply_rope_triton, _apply_rope_native, _RoPEFn, RotaryEmbedding


def _make_cos_sin(B, H, T, D, device, dtype):
    # D is the full head_dim; RotaryEmbedding(D) produces cos/sin of shape (1,1,T,D//2)
    rope = RotaryEmbedding(D, max_seq_len=T).to(device)
    x_dummy = torch.empty(B, H, T, D, device=device, dtype=dtype)
    cos, sin = rope(x_dummy, T)          # (1, 1, T, D//2)
    return cos.to(dtype), sin.to(dtype)


# ---------------------------------------------------------------------------
# Correctness — forward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,H,T,D", [
    (4, 6, 64, 64),    # training shape
    (1, 1, 64, 64),    # minimal
    (8, 6, 64, 64),    # larger batch
])
def test_forward(B, H, T, D):
    device = "cuda"
    x   = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    cos, sin = _make_cos_sin(B, H, T, D, device, torch.float32)

    ref    = _apply_rope_native(x, cos, sin)
    triton = _apply_rope_triton(x, cos, sin)

    diff = (ref - triton).abs()
    print(f"  [fwd] B={B} H={H} T={T} D={D}  max={diff.max():.2e}  mean={diff.mean():.2e}")
    torch.testing.assert_close(ref, triton, atol=1e-5, rtol=0)


@pytest.mark.parametrize("D", [32, 64])
def test_forward_bf16(D):
    device = "cuda"
    B, H, T = 4, 6, 64
    x   = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    cos, sin = _make_cos_sin(B, H, T, D, device, torch.bfloat16)

    ref    = _apply_rope_native(x, cos, sin)
    triton = _apply_rope_triton(x, cos, sin)

    diff = (ref.float() - triton.float()).abs()
    print(f"  [fwd bf16] D={D}  max={diff.max():.2e}  mean={diff.mean():.2e}")
    torch.testing.assert_close(ref, triton, atol=4e-2, rtol=0)


# ---------------------------------------------------------------------------
# Correctness — backward
# ---------------------------------------------------------------------------

def test_backward():
    device = "cuda"
    B, H, T, D = 4, 6, 64, 64

    x_ref    = torch.randn(B, H, T, D, device=device, dtype=torch.float32, requires_grad=True)
    cos, sin = _make_cos_sin(B, H, T, D, device, torch.float32)

    x_triton = x_ref.detach().clone().requires_grad_(True)

    # reference backward via torch autograd on native ops
    _apply_rope_native(x_ref, cos, sin).sum().backward()

    # triton backward via _RoPEFn
    _RoPEFn.apply(x_triton, cos, sin).sum().backward()

    diff = (x_ref.grad - x_triton.grad).abs()
    print(f"  [bwd] max={diff.max():.2e}  mean={diff.mean():.2e}")
    torch.testing.assert_close(x_ref.grad, x_triton.grad, atol=1e-5, rtol=0)


def test_backward_bf16():
    device = "cuda"
    B, H, T, D = 4, 6, 64, 64

    x_ref    = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    cos, sin = _make_cos_sin(B, H, T, D, device, torch.bfloat16)

    x_triton = x_ref.detach().clone().requires_grad_(True)

    _apply_rope_native(x_ref, cos, sin).sum().backward()
    _RoPEFn.apply(x_triton, cos, sin).sum().backward()

    diff = (x_ref.grad.float() - x_triton.grad.float()).abs()
    print(f"  [bwd bf16] max={diff.max():.2e}  mean={diff.mean():.2e}")
    torch.testing.assert_close(x_ref.grad, x_triton.grad, atol=4e-2, rtol=0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark():
    device = "cuda"
    B, H, T, D = 128, 6, 64, 64
    x   = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    cos, sin = _make_cos_sin(B, H, T, D, device, torch.bfloat16)

    def native_fb():
        x.grad = None
        _apply_rope_native(x, cos, sin).sum().backward()

    def triton_fb():
        x.grad = None
        _RoPEFn.apply(x, cos, sin).sum().backward()

    for _ in range(5):
        native_fb(); triton_fb()
    torch.cuda.synchronize()

    t_native = Timer("f(); torch.cuda.synchronize()",
                     globals={"f": native_fb, "torch": torch}).blocked_autorange()
    t_triton = Timer("f(); torch.cuda.synchronize()",
                     globals={"f": triton_fb, "torch": torch}).blocked_autorange()

    print(f"\n{'─'*45}")
    print(f"B={B}  H={H}  T={T}  D={D}  dtype=bfloat16  (fwd+bwd)")
    print(f"Native PyTorch : {t_native.median * 1e6:.1f} μs")
    print(f"Triton         : {t_triton.median * 1e6:.1f} μs")
    print(f"Speedup        : {t_native.median / t_triton.median:.2f}x")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    test_forward(4, 6, 64, 64)
    test_forward_bf16(64)
    test_backward()
    test_backward_bf16()
    print("✓ all correctness tests passed")
    benchmark()
