"""
Hybrid Mamba/Attention language model — one config, any layer recipe.

The whole point: a single `layer_pattern` string decides what each layer is.
  "AAAAAAAAAAAA"  -> pure Transformer
  "MMMMMMMMMMMM"  -> pure Mamba-2 + MLP
  "MAMAMAMAMAMA"  -> 1:1 interleave (Samba-style)
  "MMMAMMMAMMMA"  -> 1:3, attention spread through the middle (Jamba-ish)

Structure of every block (decoupled token-mixer / channel-mixer):
    x = x + token_mixer(norm(x))   # token_mixer = Attention OR Mamba-2
    x = x + MLP(norm(x))

Positional info: only the attention layers consume RoPE.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention, MLP
from misc import make_norm
from hparams import Hyperparameters
from mamba import Mamba2Mixer
from rope import RotaryEmbedding


class HybridBlock(nn.Module):
    def __init__(self, cfg: SimpleNamespace, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.ln1 = make_norm(cfg.d_model, cfg.norm)
        self.ln2 = make_norm(cfg.d_model, cfg.norm)
        self.mixer = CausalSelfAttention(cfg) if layer_type == "A" else Mamba2Mixer(cfg)
        self.mlp = MLP(cfg)
        self.use_rezero = cfg.use_rezero
        if self.use_rezero:
            self.mixer_scale = nn.Parameter(torch.zeros(1))
            self.mlp_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cos_sin) -> torch.Tensor:
        h = self.ln1(x)
        mix = self.mixer(h, cos_sin) if self.layer_type == "A" else self.mixer(h)
        if self.use_rezero:
            x = x + self.mixer_scale * mix
            x = x + self.mlp_scale * self.mlp(self.ln2(x))
        else:
            x = x + mix
            x = x + self.mlp(self.ln2(x))
        return x


class HybridLM(nn.Module):
    def __init__(self, hp: Hyperparameters, vocab_size: int):
        super().__init__()
        assert set(hp.arch.layer_pattern) <= {"M", "A"}, \
            f"layer_pattern must be M/A only: {hp.arch.layer_pattern!r}"

        d_inner = hp.mamba.expand * hp.arch.d_model
        # Flat namespace for submodules (CausalSelfAttention, Mamba2Mixer, MLP).
        cfg = SimpleNamespace(
            vocab_size=vocab_size,
            block_size=hp.train.block_size,
            d_model=hp.arch.d_model,
            dropout=hp.arch.dropout,
            norm=hp.arch.norm,
            use_rezero=hp.arch.use_rezero,
            mlp_act=hp.arch.mlp_act,
            # attention
            n_q_head=hp.attention.n_q_head,
            n_kv_heads=hp.attention.n_kv_heads,
            use_fa2=hp.runtime.use_fa2,
            # mamba
            d_inner=d_inner,
            n_heads=d_inner // hp.mamba.d_head,
            d_head=hp.mamba.d_head,
            d_state=hp.mamba.d_state,
            n_groups=hp.mamba.n_groups,
            d_conv=hp.mamba.d_conv,
            chunk_len=hp.mamba.chunk_len,
            conv_bias=True,
            proj_bias=False,
        )
        self.hp = hp
        self.cfg = cfg

        assert cfg.d_model % cfg.n_q_head == 0, "d_model must be divisible by n_q_head"
        assert d_inner % hp.mamba.d_head == 0, "d_inner must be divisible by d_head"
        assert cfg.n_heads % cfg.n_groups == 0, "n_heads must be divisible by n_groups"
        assert cfg.block_size % cfg.chunk_len == 0, "block_size must be divisible by chunk_len"

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        if hp.arch.pos_emb == "rope":
            head_dim = cfg.d_model // cfg.n_q_head
            self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.block_size)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.blocks = nn.ModuleList([HybridBlock(cfg, t) for t in hp.arch.layer_pattern])
        self.ln_f = make_norm(cfg.d_model, cfg.norm)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        if hp.arch.use_token_anchor:
            self.x0_lambdas = nn.Parameter(torch.linspace(0.20, 0.05, hp.arch.n_layer))
        if hp.arch.use_resid_scale:
            assert hp.arch.use_token_anchor, "use_resid_scale requires use_token_anchor=True"
            self.resid_lambdas = nn.Parameter(torch.linspace(1.15, 1.05, hp.arch.n_layer))

        self._init_weights()
        if hp.arch.tie_weights:
            self.head.weight = self.token_emb.weight

    def _init_weights(self):
        _plans = {"gpt2": self._init_gpt2, "muon_uniform": self._init_muon_uniform}
        assert self.hp.arch.weight_init in _plans, f"unknown weight_init: {self.hp.arch.weight_init!r}"
        _plans[self.hp.arch.weight_init]()

    @torch.no_grad()
    def _init_gpt2(self):
        exit_std = 0.02 if self.hp.arch.use_rezero else 0.0
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if not self.hp.arch.tie_weights:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        for blk in self.blocks:
            if self.hp.arch.mlp_act == "swiglu":
                nn.init.normal_(blk.mlp.up.weight,   mean=0.0, std=0.02)
                nn.init.normal_(blk.mlp.gate.weight,  mean=0.0, std=0.02)
                nn.init.normal_(blk.mlp.down.weight,  mean=0.0, std=exit_std)
            else:
                nn.init.normal_(blk.mlp.net[0].weight, mean=0.0, std=0.02)
                nn.init.normal_(blk.mlp.net[2].weight, mean=0.0, std=exit_std)
            if blk.layer_type == "A":
                nn.init.normal_(blk.mixer.q_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(blk.mixer.kv_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(blk.mixer.proj.weight, mean=0.0, std=exit_std)
            else:
                nn.init.normal_(blk.mixer.in_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(blk.mixer.out_proj.weight, mean=0.0, std=exit_std)

    @torch.no_grad()
    def _init_muon_uniform(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if not self.hp.arch.tie_weights:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        for blk in self.blocks:
            d = self.cfg.d_model
            s = 3 ** 0.5 * d ** -0.5
            if self.hp.arch.mlp_act == "swiglu":
                nn.init.uniform_(blk.mlp.up.weight,   -s * 0.4, s * 0.4)
                nn.init.uniform_(blk.mlp.gate.weight,  -s * 0.4, s * 0.4)
                nn.init.zeros_(blk.mlp.down.weight)
            else:
                nn.init.uniform_(blk.mlp.net[0].weight, -s * 0.4, s * 0.4)
                nn.init.zeros_(blk.mlp.net[2].weight)
            if blk.layer_type == "A":
                nn.init.uniform_(blk.mixer.q_proj.weight, -s, s)
                nn.init.uniform_(blk.mixer.kv_proj.weight, -s, s)
                nn.init.zeros_(blk.mixer.proj.weight)
            else:
                nn.init.uniform_(blk.mixer.in_proj.weight, -s, s)
                nn.init.zeros_(blk.mixer.out_proj.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        x0 = self.drop(self.token_emb(idx))
        if self.hp.arch.norm_emb:
            x0 = self.ln_f(x0)
        x = x0
        if self.hp.arch.pos_emb == "rope":
            cos_sin = self.rope(x, T)
        else:
            # Note: dropout is applied to token_emb only, not to pos_emb.
            # train_old.py applies dropout to (tok + pos) together — a minor difference
            # that likely explains any loss delta between the two baselines.
            x = x + self.pos_emb[:, :T]
            cos_sin = None
        for i, blk in enumerate(self.blocks):
            if self.hp.arch.use_token_anchor:
                r = self.resid_lambdas[i] if self.hp.arch.use_resid_scale else 1.0
                x = r * x + self.x0_lambdas[i] * x0
            x = blk(x, cos_sin)
        x = self.ln_f(x)

        if self.hp.runtime.use_fused_ce and targets is not None and torch.is_grad_enabled():
            from kernels.fused_ce_v2 import fused_linear_ce_v2
            loss = fused_linear_ce_v2(x.view(-1, x.size(-1)), self.head.weight,
                                      targets.view(-1), logit_softcap=self.hp.arch.logit_softcap)
            return None, loss

        logits = self.head(x).float()
        if self.hp.arch.logit_softcap > 0:
            logits = self.hp.arch.logit_softcap * torch.tanh(logits / self.hp.arch.logit_softcap)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean")
        return logits, loss
