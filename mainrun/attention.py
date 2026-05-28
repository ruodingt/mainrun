"""
CausalSelfAttention and MLP — shared transformer building blocks.

Copied from train.py so hybrid.py can import them without pulling in GPT/GPTConfig.
train.py continues to use its own local copies unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import make_norm
from rope import apply_rotary_emb


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


class CausalSelfAttention(nn.Module):
    """
    cfg duck-type requirements:
        d_model, n_q_head, dropout, block_size
        n_kv_heads  (optional, defaults to n_q_head)
        use_fa2     (optional, defaults to True)
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_q_head == 0
        self.n_q_head = cfg.n_q_head
        self.head_dim = cfg.d_model // cfg.n_q_head
        self.n_kv_heads = cfg.n_kv_heads
        self.use_sdpa = cfg.use_fa2
        self.dropout_p = cfg.dropout
        self.use_value_residual = cfg.use_value_residual
        self.use_value_residual_x0 = cfg.use_value_residual_x0
        self.use_value_carry = cfg.use_value_carry
        if self.use_value_residual or self.use_value_residual_x0:
            assert self.n_kv_heads * self.head_dim == cfg.d_model, \
                "use_value_residual requires n_kv_heads * head_dim == d_model (MHA)"
        if self.use_value_carry:
            self.v_lambda = nn.Parameter(torch.zeros(1))

        self.has_ve = cfg.has_ve
        if self.has_ve:
            self.ve_gate_channels = cfg.ve_gate_channels
            self.ve_gate = nn.Linear(cfg.ve_gate_channels, self.n_kv_heads, bias=False)

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.separate_kv = cfg.separate_kv
        kv_dim = self.n_kv_heads * self.head_dim
        if self.separate_kv:
            self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        else:
            self.kv_proj = nn.Linear(cfg.d_model, 2 * kv_dim, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        if not self.use_sdpa:
            tril = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
            self.register_buffer("tril", tril, persistent=False)

    def forward(self, x: torch.Tensor, cos_sin, x0: torch.Tensor | None = None,
                v_prev: torch.Tensor | None = None,
                ve: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C = x.size()
        # contiguous() after transpose: fixes stride layout before RoPE or SDPA.
        # Without it, torch.compile generates FA2 backward kernels assuming transposed strides,
        # which breaks when RoPE's torch.cat changes q/k to contiguous layout.
        q = self.q_proj(x).view(B, T, self.n_q_head, self.head_dim).transpose(1, 2).contiguous()
        if self.separate_kv:
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        else:
            kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).transpose(1, 3).contiguous()
            k, v = kv[..., 0, :, :], kv[..., 1, :, :]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        if self.use_value_residual:
            v = v + x.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        if self.use_value_residual_x0 and x0 is not None:
            v = v + x0.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        if self.use_value_carry and v_prev is not None:
            v = v + self.v_lambda * v_prev
        if self.has_ve and ve is not None:
            # ve: (B, T, kv_dim) → (B, T, n_kv_heads, head_dim)
            ve = ve.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_heads)
            v = v + gate.transpose(1, 2).unsqueeze(-1) * ve

        v_out = v  # save before GQA expand, same shape as v_prev next layer

        if self.n_q_head != self.n_kv_heads:
            groups = self.n_q_head // self.n_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        v_carry = v_out if self.use_value_carry else None
        return self.resid_drop(self.proj(y)), v_carry


class MLP(nn.Module):
    """
    cfg duck-type requirements: d_model, mlp_act, dropout, mlp_expand
    """
    def __init__(self, cfg):
        super().__init__()
        self.mlp_act = cfg.mlp_act
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.mlp_act == "swiglu":
            expand = cfg.mlp_expand
            hidden = round(expand * cfg.d_model * 2 / 3 / 64) * 64
            self.up   = nn.Linear(cfg.d_model, hidden, bias=False)
            self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
            self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        else:
            _actv = {'relu_sq': ReLUSquared(), 'gelu': nn.GELU()}
            self.net = nn.Sequential(
                nn.Linear(cfg.d_model, 4 * cfg.d_model),
                _actv[cfg.mlp_act],
                nn.Linear(4 * cfg.d_model, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )

    def forward(self, x):
        if self.mlp_act == "swiglu":
            return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
        return self.net(x)
