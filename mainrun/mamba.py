"""
Mamba-2: a selective state-space model (SSM) language model.

This is a faithful, dependency-free implementation of the Mamba-2 architecture
(Dao & Gu, 2024, "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality").

Why a pure-PyTorch SSD scan instead of the official `mamba_ssm` CUDA kernels:
  - The official selective_scan / chunk_scan kernels are CUDA-only. This repo
    trains on AMD ROCm (RDNA 3.5 UMA), where those kernels do not build.
  - The State Space Duality (SSD) algorithm expresses the selective scan as a
    sequence of matmuls + a segment-sum, which is exactly what torch.compile /
    Inductor fuses well. At block_size=64 the chunked scan is cheap.

Mamba2Mixer is a drop-in token mixer for HybridLM. It accepts a duck-typed cfg
namespace with fields: d_model, d_inner, n_heads, d_head, d_state, n_groups,
d_conv, chunk_len, conv_bias, proj_bias, dropout, norm.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F





# ---------------------------------------------------------------------------
# State Space Duality (SSD) — the chunked scan that is the heart of Mamba-2.
# ---------------------------------------------------------------------------
def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segmented cumulative sum over the last dim.

    Given log-decays x[..., t], returns L[..., i, j] = sum_{k=j+1}^{i} x[..., k]
    for i >= j, and -inf for i < j. exp(L) is then the causal decay matrix:
    how much the input at step j survives to the output at step i.

    Doing this as a masked cumsum (instead of a Python loop) is what lets the
    whole scan stay a handful of fused tensor ops.
    """
    T = x.size(-1)
    # x[..., i, j] := x[..., i]  (broadcast the value across a new last axis)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, float("-inf"))
    return x_segsum


def ssd(x: torch.Tensor, a: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
        chunk_len: int) -> torch.Tensor:
    """Selective SSM scan via state-space duality (chunked).

    Computes  h_t = exp(a_t) * h_{t-1} + B_t x_t ,  y_t = C_t . h_t
    where the dt-discretization has already been folded into `x` (x := dt*x)
    and `a` (a := dt * A, the per-step log-decay).

    Shapes:
        x: (b, L, h, p)   per-head input channels
        a: (b, L, h)      per-step log-decay (negative)
        B: (b, L, h, n)   input->state projection (already expanded to n_heads)
        C: (b, L, h, n)   state->output projection
    Returns:
        y: (b, L, h, p)

    The trick (SSD): split the length into chunks. Within a chunk, the scan is a
    masked quadratic attention-like matmul (the "diagonal blocks"). Across
    chunks, only one summarized state per chunk is carried forward — a cheap
    chunk-level recurrence — and then broadcast back into each chunk's outputs.
    """
    b, L, h, p = x.shape
    n = B.shape[-1]
    assert L % chunk_len == 0, f"block_size {L} must be divisible by chunk_len {chunk_len}"
    c = L // chunk_len  # number of chunks
    q = chunk_len

    # Run the scan in fp32: cumsum/exp over decays is precision-sensitive, and
    # this is exactly the kind of accumulation autocast leaves in fp32 anyway.
    x, a, B, C = (t.float() for t in (x, a, B, C))

    # reshape (b, L, ...) -> (b, c, q, ...)
    x = x.reshape(b, c, q, h, p)
    a = a.reshape(b, c, q, h)
    B = B.reshape(b, c, q, h, n)
    C = C.reshape(b, c, q, h, n)

    a = a.permute(0, 3, 1, 2)            # (b, h, c, q)
    a_cumsum = torch.cumsum(a, dim=-1)   # (b, h, c, q)

    # 1. Intra-chunk (diagonal blocks): masked quadratic form.
    Lmat = torch.exp(_segsum(a))         # (b, h, c, q, q)
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, Lmat, x)

    # 2. Each chunk's end-state contribution (right factor of the off-diagonal).
    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)          # (b, h, c, q)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, x)  # (b, c, h, p, n)

    # 3. Inter-chunk recurrence: carry one state per chunk boundary.
    init = torch.zeros_like(states[:, :1])
    states = torch.cat([init, states], dim=1)                       # (b, c+1, h, p, n)
    decay_chunk = torch.exp(_segsum(F.pad(a_cumsum[..., -1], (1, 0))))  # (b, h, c+1, c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]                                     # state entering each chunk

    # 4. Off-diagonal: broadcast each chunk's entry-state across its timesteps.
    state_decay_out = torch.exp(a_cumsum)                           # (b, h, c, q)
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    y = (y_diag + y_off).reshape(b, L, h, p)
    return y


# ---------------------------------------------------------------------------
# Gated RMSNorm (Mamba-2 output norm): normalize x * silu(z).
# ---------------------------------------------------------------------------
class RMSNormGated(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = x * F.silu(z)
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).type_as(self.weight)


# ---------------------------------------------------------------------------
# Mamba-2 mixer block.
# ---------------------------------------------------------------------------
class Mamba2Mixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_inner = cfg.d_inner
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_state = cfg.d_state
        self.n_groups = cfg.n_groups
        self.chunk_len = cfg.chunk_len
        assert self.n_heads % self.n_groups == 0, "n_heads must be divisible by n_groups"

        self.conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        d_in_proj = self.d_inner + self.conv_dim + self.n_heads  # [z, xBC, dt]

        self.in_proj = nn.Linear(cfg.d_model, d_in_proj, bias=cfg.proj_bias)

        # Causal depthwise conv over (x, B, C). padding=d_conv-1 + truncate -> causal.
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim,
            kernel_size=cfg.d_conv, groups=self.conv_dim,
            padding=cfg.d_conv - 1, bias=cfg.conv_bias,
        )

        # A is a scalar per head (Mamba-2 simplification of Mamba-1's diagonal A).
        # Stored in log-space and negated so A < 0 (stable decay) by construction.
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))      # skip connection, per head
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))

        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=cfg.proj_bias)

        self._reset_ssm_parameters()

    @torch.no_grad()
    def _reset_ssm_parameters(self):
        # Canonical Mamba SSM-parameter init — physical and optimizer-agnostic, so it
        # lives on the mixer itself (any model reusing Mamba2Mixer gets it for free).
        # A ~ Uniform(1, 16): a spread of decay timescales across heads.
        A = torch.empty(self.n_heads).uniform_(1.0, 16.0)
        self.A_log.copy_(torch.log(A))
        self.D.fill_(1.0)
        # dt ~ log-uniform(dt_min, dt_max); dt_bias = softplus^{-1}(dt).
        dt_min, dt_max = 1e-3, 1e-1
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp_min(1e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B_, L, _ = u.shape
        ng, ds = self.n_groups, self.d_state

        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.n_heads], dim=-1
        )

        # Causal depthwise conv + SiLU. Conv1d wants (B, C, L).
        xBC = xBC.transpose(1, 2)
        xBC = self.conv1d(xBC)[..., :L]      # truncate the left-pad tail -> causal
        xBC = F.silu(xBC.transpose(1, 2))

        x, Bm, Cm = torch.split(xBC, [self.d_inner, ng * ds, ng * ds], dim=-1)

        x = x.reshape(B_, L, self.n_heads, self.d_head)
        Bm = Bm.reshape(B_, L, ng, ds)
        Cm = Cm.reshape(B_, L, ng, ds)
        # Share B/C across heads within a group (n_groups=1 -> broadcast to all heads).
        heads_per_group = self.n_heads // ng
        Bm = Bm.repeat_interleave(heads_per_group, dim=2)   # (B, L, n_heads, ds)
        Cm = Cm.repeat_interleave(heads_per_group, dim=2)

        # Discretize: dt = softplus(dt + bias); fold dt into x and into the log-decay.
        A = -torch.exp(self.A_log.float())                  # (n_heads,)
        dt = F.softplus(dt.float() + self.dt_bias.float())  # (B, L, n_heads)
        x_dt = x * dt.unsqueeze(-1)                          # (B, L, n_heads, d_head)
        a = dt * A                                           # (B, L, n_heads) per-step log-decay

        y = ssd(x_dt, a, Bm, Cm, self.chunk_len)             # (B, L, n_heads, d_head)
        y = y + x * self.D.float().view(1, 1, -1, 1)         # per-head skip on raw x
        y = y.reshape(B_, L, self.d_inner).type_as(u)

        y = self.norm(y, z)                                  # gated RMSNorm
        return self.out_proj(y)


