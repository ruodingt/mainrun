"""
Hybrid Mamba/Attention language model — one config, any layer recipe.

The whole point: a single `layer_pattern` string decides what each layer is.
  "AAAAAAAAAAAA"  -> pure Transformer (== train.py's GPT, the control)
  "MMMMMMMMMMMM"  -> pure Mamba-2 + MLP
  "MAMAMAMAMAMA"  -> 1:1 interleave (Samba-style)
  "MMMAMMMAMMMA"  -> 1:3, attention spread through the middle (Jamba-ish)

This makes the hybrid design space a hyperparameter you can sweep, and lets the
two pure architectures fall out as degenerate patterns for clean comparison.

Structure of every block (decoupled token-mixer / channel-mixer, like Samba/Jamba):
    x = x + scale_mix * token_mixer(norm(x))   # token_mixer = Attention OR Mamba-2
    x = x + scale_mlp * MLP(norm(x))            # channel mixer, identical everywhere

We reuse train.py's CausalSelfAttention and MLP verbatim so the "A" layers are
byte-for-byte the same operator as the existing GPT — any difference in results
is attributable to the architecture mix, not to an incidental reimplementation.

Positional info: only the attention layers consume RoPE. Mamba layers carry
order through their conv + recurrence and ignore it.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryEmbedding
from train import CausalSelfAttention, MLP
from mamba import Mamba2Mixer, _make_norm


@dataclass
class HybridConfig:
    vocab_size: int
    block_size: int
    d_model: int
    layer_pattern: str = "MMMAMMMAMMMA"  # 'M' = Mamba-2, 'A' = Attention; len == n_layer
    dropout: float = 0.1

    # --- attention ('A') layers ---
    n_q_head: int = 6
    n_kv_heads: int = 1            # 1 = MQA, n_q_head = MHA, in between = GQA
    use_fa2: bool = True

    # --- mamba ('M') layers ---
    expand: int = 2               # d_inner = expand * d_model
    d_head: int = 64              # mamba per-head dim P; n_heads = d_inner // d_head
    d_state: int = 128            # SSM state dim N
    n_groups: int = 1             # B/C shared across heads within a group
    d_conv: int = 4
    chunk_len: int = 64           # SSD chunk; must divide block_size
    conv_bias: bool = True
    proj_bias: bool = False

    # --- shared ---
    use_rmsnorm: bool = True
    use_rezero: bool = True        # per-layer learnable residual scalars (init 0 -> identity at step 0)
    tie_weights: bool = True
    norm_emb: bool = False
    logit_softcap: float = 15.0
    mlp_act: str = "relu_sq"

    @property
    def n_layer(self) -> int:
        return len(self.layer_pattern)

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_heads(self) -> int:
        return self.d_inner // self.d_head


class HybridBlock(nn.Module):
    def __init__(self, cfg: HybridConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.ln1 = _make_norm(cfg.d_model, cfg.use_rmsnorm)
        self.ln2 = _make_norm(cfg.d_model, cfg.use_rmsnorm)
        self.mixer = CausalSelfAttention(cfg) if layer_type == "A" else Mamba2Mixer(cfg)
        self.mlp = MLP(cfg)
        self.use_rezero = cfg.use_rezero
        if self.use_rezero:
            self.mixer_scale = nn.Parameter(torch.zeros(1))
            self.mlp_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cos_sin) -> torch.Tensor:
        h = self.ln1(x)
        # Only attention needs RoPE; Mamba's signature is mixer(x).
        mix = self.mixer(h, cos_sin) if self.layer_type == "A" else self.mixer(h)
        if self.use_rezero:
            x = x + self.mixer_scale * mix
            x = x + self.mlp_scale * self.mlp(self.ln2(x))
        else:
            x = x + mix
            x = x + self.mlp(self.ln2(x))
        return x


class HybridLM(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        assert set(cfg.layer_pattern) <= {"M", "A"}, f"layer_pattern must be M/A only: {cfg.layer_pattern!r}"
        assert cfg.d_model % cfg.n_q_head == 0, "d_model must be divisible by n_q_head"
        assert cfg.d_inner % cfg.d_head == 0, "d_inner must be divisible by d_head"
        assert cfg.n_heads % cfg.n_groups == 0, "n_heads must be divisible by n_groups"
        assert cfg.block_size % cfg.chunk_len == 0, "block_size must be divisible by chunk_len"

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        head_dim = cfg.d_model // cfg.n_q_head
        self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.block_size)
        self.blocks = nn.ModuleList([HybridBlock(cfg, t) for t in cfg.layer_pattern])
        self.ln_f = _make_norm(cfg.d_model, cfg.use_rmsnorm)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()
        if cfg.tie_weights:
            self.head.weight = self.token_emb.weight

    @torch.no_grad()
    def _init_weights(self):
        # Identity-start strategy mirrors train.py's GPT:
        #   - with ReZero, the per-layer scale (init 0) already gives identity at step 0,
        #     so residual-exit weights get a normal init.
        #   - without ReZero, zero the residual exits instead (std=0 -> all zeros).
        exit_std = 0.02 if self.cfg.use_rezero else 0.0

        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if not self.cfg.tie_weights:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

        for blk in self.blocks:
            nn.init.normal_(blk.mlp.net[0].weight, mean=0.0, std=0.02)   # MLP up
            nn.init.normal_(blk.mlp.net[2].weight, mean=0.0, std=exit_std)  # MLP down (exit)
            if blk.layer_type == "A":
                nn.init.normal_(blk.mixer.q_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(blk.mixer.kv_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(blk.mixer.proj.weight, mean=0.0, std=exit_std)  # attn exit
            else:
                nn.init.normal_(blk.mixer.in_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(blk.mixer.out_proj.weight, mean=0.0, std=exit_std)  # mamba exit
                # A_log/D/dt_bias are canonically initialized inside Mamba2Mixer.

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        x = self.drop(self.token_emb(idx))
        if self.cfg.norm_emb:
            x = self.ln_f(x)
        cos_sin = self.rope(x, T)
        for blk in self.blocks:
            x = blk(x, cos_sin)
        x = self.ln_f(x)
        logits = self.head(x).float()
        if self.cfg.logit_softcap > 0:
            logits = self.cfg.logit_softcap * torch.tanh(logits / self.cfg.logit_softcap)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean")
        return logits, loss
