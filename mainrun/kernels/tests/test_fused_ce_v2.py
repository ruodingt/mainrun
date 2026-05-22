"""
FusedLinearCEv2 (tl.dot inside kernel) tests: correctness + fwd/bwd benchmark.

Run:
    pytest kernels/tests/test_fused_ce_v2.py -v
    python kernels/tests/test_fused_ce_v2.py
"""
import torch
import pytest
from torch.utils.benchmark import Timer

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from kernels.fused_ce_v2 import fused_linear_ce_v2


def _ref(x, W, targets, ignore_index=-100):
    """fp32 reference — used for loose forward tolerance checks."""
    logits = x.float() @ W.float().T
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)


def _ref_bf16(x, W, targets, ignore_index=-100):
    """bf16-matmul reference — matches kernel compute dtype, used for precise grad checks."""
    logits = (x.bfloat16() @ W.bfloat16().T).float()
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("BT,H,V", [
    (8192, 384, 8192),   # training shape
    (512,  384, 8192),   # small batch
    (256,  384, 8192),   # tiny
    (8192, 512, 8192),   # larger hidden
])
def test_forward(BT, H, V):
    device  = "cuda"
    x       = torch.randn(BT, H, device=device, dtype=torch.float32)
    W       = torch.randn(V, H, device=device, dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device=device)

    ref  = _ref(x, W, targets)
    fuse = fused_linear_ce_v2(x, W, targets)

    diff = (ref - fuse).abs()
    print(f"  [fwd] BT={BT} H={H} V={V}  diff={diff.item():.2e}")
    assert diff.item() < 1e-2, f"loss mismatch: ref={ref.item():.6f} fused={fuse.item():.6f}"


def test_forward_with_ignore():
    device  = "cuda"
    BT, H, V = 512, 384, 8192
    x       = torch.randn(BT, H, device=device, dtype=torch.float32)
    W       = torch.randn(V, H, device=device, dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device=device)
    targets[:BT // 4] = -100

    ref  = _ref(x, W, targets)
    fuse = fused_linear_ce_v2(x, W, targets)

    diff = (ref - fuse).abs()
    print(f"  [ignore] diff={diff.item():.2e}")
    assert diff.item() < 1e-2


def test_backward():
    device  = "cuda"
    BT, H, V = 256, 384, 8192

    x_ref  = torch.randn(BT, H, device=device, dtype=torch.float32, requires_grad=True)
    W_ref  = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device=device)

    x_fuse = x_ref.detach().clone().requires_grad_(True)
    W_fuse = W_ref.detach().clone().requires_grad_(True)

    _ref_bf16(x_ref, W_ref, targets).backward()
    fused_linear_ce_v2(x_fuse, W_fuse, targets).backward()

    dx_diff = (x_ref.grad - x_fuse.grad).abs()
    dW_diff = (W_ref.grad - W_fuse.grad).abs()
    print(f"  [bwd dx] max={dx_diff.max():.2e}  mean={dx_diff.mean():.2e}")
    print(f"  [bwd dW] max={dW_diff.max():.2e}  mean={dW_diff.mean():.2e}")

    # bf16 matmul in kernel vs bf16 matmul reference: different K-tile accumulation order → ~2e-3 max
    assert dx_diff.max() < 5e-3
    assert dW_diff.max() < 5e-3


# ---------------------------------------------------------------------------
# Benchmark — FAIR: forward+backward on both sides
# ---------------------------------------------------------------------------

def benchmark():
    device  = "cuda"
    BT, H, V = 8192, 384, 8192
    x       = torch.randn(BT, H, device=device, dtype=torch.float32, requires_grad=True)
    W       = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device=device)

    def ref_fb():
        x.grad = None; W.grad = None
        loss = _ref(x, W, targets)
        loss.backward()

    def fuse_fb():
        x.grad = None; W.grad = None
        loss = fused_linear_ce_v2(x, W, targets)
        loss.backward()

    for _ in range(3):
        ref_fb(); fuse_fb()
    torch.cuda.synchronize()

    # peak memory: standard path materializes [BT, V] logits
    torch.cuda.reset_peak_memory_stats()
    ref_fb(); torch.cuda.synchronize()
    mem_ref = torch.cuda.max_memory_allocated() / 1e6

    torch.cuda.reset_peak_memory_stats()
    fuse_fb(); torch.cuda.synchronize()
    mem_fuse = torch.cuda.max_memory_allocated() / 1e6

    t_ref  = Timer("f(); torch.cuda.synchronize()",
                   globals={"f": ref_fb, "torch": torch}).blocked_autorange()
    t_fuse = Timer("f(); torch.cuda.synchronize()",
                   globals={"f": fuse_fb, "torch": torch}).blocked_autorange()

    print(f"\n{'─'*50}")
    print(f"BT={BT}  H={H}  V={V}  (forward+backward, fp32)")
    print(f"Standard CE  : {t_ref.median  * 1e3:7.2f} ms   peak {mem_ref:8.1f} MB")
    print(f"Fused v2     : {t_fuse.median * 1e3:7.2f} ms   peak {mem_fuse:8.1f} MB")
    print(f"Speedup      : {t_ref.median / t_fuse.median:.2f}x")
    print(f"Mem saving   : {mem_ref / mem_fuse:.2f}x")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    test_forward(8192, 384, 8192)
    test_forward(512, 384, 8192)
    test_forward_with_ignore()
    test_backward()
    print("✓ all correctness tests passed")
    benchmark()
