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
        self.n_kv_heads = getattr(cfg, "n_kv_heads", cfg.n_q_head)
        self.use_sdpa = getattr(cfg, "use_fa2", True)
        self.dropout_p = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        if not self.use_sdpa:
            tril = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
            self.register_buffer("tril", tril, persistent=False)

    def forward(self, x: torch.Tensor, cos_sin) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_q_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).transpose(1, 3)
        k, v = kv[..., 0, :, :], kv[..., 1, :, :]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            if self.n_q_head != self.n_kv_heads:
                num_queries_per_kv = self.n_q_head // self.n_kv_heads
                k = k.repeat_interleave(num_queries_per_kv, dim=1)
                v = v.repeat_interleave(num_queries_per_kv, dim=1)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """
    cfg duck-type requirements: d_model, mlp_act, dropout
    """
    def __init__(self, cfg):
        super().__init__()
        self.mlp_act = cfg.mlp_act
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.mlp_act == "swiglu":
            # hidden = 8d/3 rounded to nearest multiple of 64 — iso-param vs 2-matrix 4d MLP
            hidden = round(8 * cfg.d_model / 3 / 64) * 64
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
