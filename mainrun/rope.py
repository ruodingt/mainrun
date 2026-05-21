import math
import torch
import torch.nn as nn

# ==========================================
# 0. Triton Environment Detection & Safeguard
# ==========================================
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ==========================================
# 1. Triton GPU-Accelerated Kernel (Forward Only)
# ==========================================
# Trade-off: The Triton kernel fuses split-rotate-concat into a single DRAM pass,
# which matters at long sequence lengths or large batch sizes. At block_size=128,
# the native path (below) is fast enough — torch.compile would achieve similar
# fusion. Keep the Triton path for correctness on GPU; remove if Triton is unavailable.
if HAS_TRITON:
    @triton.jit
    def _rope_fwd_kernel(
            X_ptr, COS_ptr, SIN_ptr, Y_ptr,
            stride_xb, stride_xh, stride_xt, stride_xd,
            stride_ct, stride_cd,
            stride_yb, stride_yh, stride_yt, stride_yd,
            H, T, D,
            BLOCK_D_HALF: tl.constexpr
    ):
        """
        Triton-fused RoPE forward kernel.
        Loads [B, H, T, D] tensor into SRAM, performs rotation at register level,
        and completely eliminates DRAM roundtrips from physical cat/stack operations.
        """
        # Grid dimension 0: Handles Batch and Head unfolding
        pid_bh = tl.program_id(0)
        # Grid dimension 1: Handles time step T (Sequence Length)
        pid_t = tl.program_id(1)

        batch_idx = pid_bh // H
        head_idx = pid_bh % H

        # Locate the starting pointers of X and Y in DRAM corresponding to the current token
        x_base = X_ptr + batch_idx * stride_xb + head_idx * stride_xh + pid_t * stride_xt
        y_base = Y_ptr + batch_idx * stride_yb + head_idx * stride_yh + pid_t * stride_yt

        # Locate the starting pointers of Cos and Sin in DRAM corresponding to the current time step
        cos_base = COS_ptr + pid_t * stride_ct
        sin_base = SIN_ptr + pid_t * stride_ct

        # Compute the index offset for the first half of the head dimension (0 to D//2 - 1)
        d_half = D // 2
        offsets = tl.arange(0, BLOCK_D_HALF)
        mask = offsets < d_half

        # Load data from external DRAM into SM registers in a single memory transaction
        x1 = tl.load(x_base + offsets * stride_xd, mask=mask, other=0.0)
        x2 = tl.load(x_base + (offsets + d_half) * stride_xd, mask=mask, other=0.0)

        cos = tl.load(cos_base + offsets * stride_cd, mask=mask, other=0.0)
        sin = tl.load(sin_base + offsets * stride_cd, mask=mask, other=0.0)

        # Register-level complex rotation, eliminating DRAM roundtrips from physical cat/stack
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        # Store the results directly back to the target DRAM memory
        tl.store(y_base + offsets * stride_yd, y1, mask=mask)
        tl.store(y_base + (offsets + d_half) * stride_yd, y2, mask=mask)


    def _apply_rope_gpu_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE using high-performance Triton kernel.
        x: (B, H, T, D)
        cos/sin: (1, 1, T, D // 2)
        """
        B, H, T, D = x.shape
        # Squeeze cos/sin to 2D (T, D // 2) to simplify pointer arithmetic inside Triton
        cos_2d = cos.squeeze(0).squeeze(0)
        sin_2d = sin.squeeze(0).squeeze(0)

        y = torch.empty_like(x)
        grid = (B * H, T)
        # Select the next power of 2 as the block size
        BLOCK_D_HALF = triton.next_power_of_2(D // 2)

        _rope_fwd_kernel[grid](
            x, cos_2d, sin_2d, y,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            cos_2d.stride(0), cos_2d.stride(1),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            H=H, T=T, D=D,
            BLOCK_D_HALF=BLOCK_D_HALF
        )
        return y


# ==========================================
# 2. CPU / Fallback Memory-Friendly Operator (Inductor Friendly)
# ==========================================
def _apply_rope_native(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Native PyTorch implementation optimized for CPU and non-CUDA devices.
    Minimizes unnecessary memory allocation via views and remains fully compatible with torch.compile (Inductor).
    x: (B, H, T, D)
    cos/sin: (1, 1, T, D // 2)
    """
    d_half = x.shape[-1] // 2
    # Compute cheap sliced views of the input tensor
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]

    # Memory-aligned element-wise multiply-add operations, leveraging CPU SIMD (AVX-512) or compilation-level fusion
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


# ==========================================
# 3. Unified Hardware-Adaptive Routing API
# ==========================================
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Unified routing entry point for RoPE.
    Dynamically detects runtime hardware status and routes execution to the optimal backend.
    """
    if x.is_cuda and HAS_TRITON:
        # GPU Scenario: Run high-performance Triton fused-kernel path
        return _apply_rope_gpu_triton(x, cos, sin)
    else:
        # CPU/Fallback Scenario: Run memory-friendly native path compatible with compiler fusion
        return _apply_rope_native(x, cos, sin)


# ==========================================
# 4. RoPE Cache Manager (Rotary Embedding Module)
# ==========================================
# Trade-off: RotaryEmbedding handles device migration, dtype casting, and dynamic
# cache expansion — useful for inference pipelines or multi-GPU setups where tensors
# move between devices. For fixed-length training with a known block_size, a plain
# register_buffer in the model would suffice. We use this module for correctness and
# portability without changing the training interface.
class RotaryEmbedding(nn.Module):
    """
    RoPE cache manager with lazy device/dtype alignment and dynamic length expansion.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency scale factors (Theta band)
        # shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Initialize cache on CPU
        self._update_cos_sin_cache(max_seq_len, device=torch.device("cpu"))

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device):
        """Build or update high-precision trigonometric cos/sin cache in-place"""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Outer product to generate sequential phase matrix of shape (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq.to(device))

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Register as non-persistent buffers
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically slice and return cos and sin matched to shape [1, 1, seq_len, D//2] based on the current batch tensor status.
        """
        # Align devices and handle cold start
        if self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self.inv_freq = self.inv_freq.to(x.device)
            self._update_cos_sin_cache(max(self.max_seq_len, seq_len), device=x.device)
            self.cos_cached = self.cos_cached.to(x.dtype)
            self.sin_cached = self.sin_cached.to(x.dtype)

        # Automatically expand cache when sequence length exceeds current capacity
        elif seq_len > self.cos_cached.shape[0]:
            self._update_cos_sin_cache(seq_len, device=x.device)
            self.cos_cached = self.cos_cached.to(x.dtype)
            self.sin_cached = self.sin_cached.to(x.dtype)

        # Slice cos/sin for the current sequence length and unsqueeze to 4D to support head and batch broadcasting
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(1)
        return cos, sin