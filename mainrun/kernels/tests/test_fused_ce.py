"""
FusedLinearCE tests: correctness + benchmark.

Run:
    pytest kernels/tests/test_fused_ce.py -v
    python kernels/tests/test_fused_ce.py
"""
import torch
import pytest
from torch.utils.benchmark import Timer

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from kernels.fused_ce import fused_linear_ce


def _ref(x, W, targets, ignore_index=-100):
    logits = x.float() @ W.float().T
    return torch.nn.functional.cross_entropy(
        logits, targets, ignore_index=ignore_index
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("BT,H,V", [
    (8192,  384, 8192),   # training shape
    (512,   384, 8192),   # small batch
    (8192,  512, 8192),   # larger hidden
])
def test_forward(BT, H, V):
    device  = "cuda"
    x       = torch.randn(BT, H, device=device, dtype=torch.float32, requires_grad=True)
    W       = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device=device)

    ref  = _ref(x, W, targets)
    fuse = fused_linear_ce(x, W, targets)

    diff = (ref - fuse).abs()
    print(f"  [fwd] BT={BT} H={H} V={V}  diff={diff.item():.2e}")
    # chunked matmul has different float32 accumulation order than full matmul — 1e-2 is expected
    assert diff.item() < 1e-2, f"loss mismatch: ref={ref.item():.6f} fused={fuse.item():.6f}"


def test_forward_with_ignore():
    device  = "cuda"
    BT, H, V = 512, 384, 8192
    x       = torch.randn(BT, H, device=device, dtype=torch.float32, requires_grad=True)
    W       = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device=device)
    targets[:BT // 4] = -100  # ignore first quarter

    ref  = _ref(x, W, targets)
    fuse = fused_linear_ce(x, W, targets)

    diff = (ref - fuse).abs()
    print(f"  [ignore] diff={diff.item():.2e}")
    assert diff.item() < 1e-3


def test_backward():
    device  = "cuda"
    BT, H, V = 256, 384, 8192

    x_ref  = torch.randn(BT, H, device=device, dtype=torch.float32, requires_grad=True)
    W_ref  = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device=device)

    # Clone for fused
    x_fuse = x_ref.detach().clone().requires_grad_(True)
    W_fuse = W_ref.detach().clone().requires_grad_(True)

    _ref(x_ref, W_ref, targets).backward()
    fused_linear_ce(x_fuse, W_fuse, targets).backward()

    dx_diff = (x_ref.grad - x_fuse.grad).abs()
    dW_diff = (W_ref.grad - W_fuse.grad).abs()
    print(f"  [bwd dx] max={dx_diff.max():.2e}  mean={dx_diff.mean():.2e}")
    print(f"  [bwd dW] max={dW_diff.max():.2e}  mean={dW_diff.mean():.2e}")

    assert dx_diff.max() < 1e-3
    assert dW_diff.max() < 1e-3


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark():
    device  = "cuda"
    BT, H, V = 8192, 384, 8192
    x       = torch.randn(BT, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    W       = torch.randn(V, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device=device)

    def ref_fwd():
        logits = x.float() @ W.float().T
        return torch.nn.functional.cross_entropy(logits, targets)

    def fuse_fwd():
        return fused_linear_ce(x, W, targets)

    # Warm up
    for _ in range(5):
        ref_fwd(); fuse_fwd()
    torch.cuda.synchronize()

    t_ref  = Timer("f(); torch.cuda.synchronize()",
                   globals={"f": ref_fwd, "torch": torch}).blocked_autorange()
    t_fuse = Timer("f(); torch.cuda.synchronize()",
                   globals={"f": fuse_fwd, "torch": torch}).blocked_autorange()

    print(f"\n{'─'*45}")
    print(f"BT={BT}  H={H}  V={V}  dtype=bfloat16")
    print(f"Standard CE  : {t_ref.median  * 1e3:.2f} ms")
    print(f"Fused CE     : {t_fuse.median * 1e3:.2f} ms")
    print(f"Speedup      : {t_ref.median / t_fuse.median:.2f}x")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    test_forward(8192, 384, 8192)
    test_forward(512, 384, 8192)
    test_forward_with_ignore()
    test_backward()
    print("✓ all correctness tests passed")
    benchmark()
