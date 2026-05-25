"""
Structured hyperparameters for train_hybrid.py.

Groups:
  AttentionHparams  — attention-specific architecture
  MambaHparams      — Mamba-specific architecture (only active when layer_pattern has 'M')
  ModelArchHparams  — shared architecture (layer pattern, d_model, norms, init, …)
  OptimizerHparams  — learning rates + schedule
  RunHparams        — training operation (batch/block size, precision, compile flags, …)
  FixedHparams      — locked; never swept (seed, dataset size, epochs)

CLI: pass a flat JSON dict; dotted keys ("optimizer.muon_lr") or bare keys both work.
Bare keys are resolved by scanning groups in order — fail loudly on ambiguity-free miss.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class AttentionHparams:
    n_q_head: int = 6
    n_kv_heads: int = 1    # 1=MQA, n_q_head=MHA, anything between=GQA


@dataclass
class MambaHparams:
    expand: int = 2        # d_inner = expand * d_model
    d_head: int = 64       # SSM per-head dim; n_heads = d_inner // d_head
    d_state: int = 128     # SSM state dim N
    n_groups: int = 1      # SSD groups — B/C shared across heads within a group
    d_conv: int = 4        # depthwise conv kernel size
    chunk_len: int = 64    # SSD chunk length; must evenly divide block_size


@dataclass
class ModelArchHparams:
    # "A"=Attention, "M"=Mamba; length defines n_layer.
    # Pure transformer: "AAAAAAAAAAAA". Jamba-ish: "MMMAMMMAMMMA". Samba: "MAMAMAMAMAMA".
    layer_pattern: str = "A" * 12

    vocab_size: int = 8192
    d_model: int = 384
    dropout: float = 0.1

    norm: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    pos_emb: Literal["rope", "learned"] = "rope"
    # use_rezero - False (default): muon_uniform init already zeros residual exits, giving identity-at-init
    # for free. Stacking ReZero (scale init=0) on top would multiply exit by 0×0 and kill
    # gradients entirely. Only set True when using weight_init="gpt2".
    use_rezero: bool = False
    use_token_anchor: bool = True   # per-layer learnable skip from original token embedding
    use_resid_scale: bool = False   # per-layer residual stream scaling; requires use_token_anchor=True
    weight_init: Literal["gpt2", "muon_uniform"] = "muon_uniform"
    tie_weights: bool = True        # tie lm_head to token_emb
    norm_emb: bool = False          # RMSNorm after embedding; required when token_emb std=0.8
    logit_softcap: float = 15.0     # tanh softcap on logits; 0.0 = disabled
    mlp_act: Literal["relu_sq", "gelu", "swiglu"] = "gelu"

    @property
    def n_layer(self) -> int:
        return len(self.layer_pattern)


@dataclass
class OptimizerHparams:
    optimizer_type: Literal["muon_adamw", "sgd"] = "muon_adamw"
    muon_lr: float = 0.02       # Muon: 2D weight matrices (attn, mlp)
    adamw_lr: float = 3e-4      # AdamW: norms, biases, other 1D params
    emb_lr: float = 3e-3        # AdamW: token_emb (sparse updates → slightly higher LR)
    scalar_lr: float = 1e-3     # AdamW: ReZero scalars, x0_lambdas
    adamw_wd: float = 0.1
    sgd_lr: float = 6e-3        # SGD baseline LR (matches train_old.py)
    sgd_wd: float = 0.0         # SGD weight decay

    lr_schedule: Literal["wsd", "cosine"] = "wsd"
    warmup_frac: float = 0.05
    decay_frac: float = 0.60    # WSD only: fraction of steps for final cosine decay
    min_lr_frac: float = 0.05    # decay floor; 0.0 = decay all the way to zero


@dataclass
class TrainHparams:
    """Training params that affect model quality — all included in fingerprint."""
    batch_size: int = 128
    block_size: int = 64
    clip_norm_mode: Literal["all", "adamw"] = "all"  # "adamw": skip Muon params (self-normalizing)


@dataclass
class RuntimeHparams:
    """Execution config — doesn't affect loss numerics, excluded from fingerprint."""
    device: str = "cuda"            # "cuda", "cpu"
    use_fa2: bool = True            # FlashAttention-2 kernel; same math, different execution path
    use_fused_ce: bool = False      # fused linear+CE kernel (~0.6x speed on RDNA4)
    use_compile: bool = True
    use_bf16: bool = True
    evals_per_epoch: int = 3
    experiments_dir: str = "./experiments"
    log_file: str = "./logs/mainrun.log"


@dataclass
class FixedHparams:
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10


@dataclass
class Hyperparameters:
    arch: ModelArchHparams = field(default_factory=ModelArchHparams)
    attention: AttentionHparams = field(default_factory=AttentionHparams)
    mamba: MambaHparams = field(default_factory=MambaHparams)
    optimizer: OptimizerHparams = field(default_factory=OptimizerHparams)
    train: TrainHparams = field(default_factory=TrainHparams)
    runtime: RuntimeHparams = field(default_factory=RuntimeHparams)
    fixed: FixedHparams = field(default_factory=FixedHparams)

    def update_from_flat(self, flat: dict) -> None:
        """
        Update from a flat dict. Supports:
          - dotted keys:  {"optimizer.muon_lr": 0.02}
          - bare keys:    {"muon_lr": 0.02}  (scanned in group order; fails loudly if not found)
        """
        groups: dict[str, object] = {
            "arch": self.arch, "attention": self.attention,
            "mamba": self.mamba, "optimizer": self.optimizer,
            "train": self.train, "runtime": self.runtime, "fixed": self.fixed,
        }
        for key, value in flat.items():
            if "." in key:
                group_name, attr = key.split(".", 1)
                assert group_name in groups, f"unknown group: {group_name!r}"
                obj = groups[group_name]
                assert hasattr(obj, attr), f"unknown param: {key!r}"
                setattr(obj, attr, value)
            else:
                found = False
                for obj in groups.values():
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                        found = True
                        break
                assert found, f"unknown hyperparam: {key!r}"

    def flat_dict(self) -> dict:
        """All params as a flat dict (for logging). Includes n_layer as a convenience."""
        result: dict = {}
        for g in [self.arch, self.attention, self.mamba, self.optimizer, self.train, self.runtime, self.fixed]:
            result.update(asdict(g))
        result["n_layer"] = self.arch.n_layer
        return result

    def get_fingerprint(self) -> dict:
        """
        Params that affect model quality — used for experiment deduplication.
        Excludes: runtime (use_compile, use_bf16, …), fixed (seed, epochs, …),
                  and params that are inactive given the current config.
        """
        fp: dict = {}
        fp.update(asdict(self.arch))
        fp.update(asdict(self.attention))
        fp.update(asdict(self.optimizer))
        fp.update(asdict(self.train))

        # Mamba params only matter when there are Mamba layers
        if "M" in self.arch.layer_pattern:
            fp.update(asdict(self.mamba))

        # Params whose effect is gated by another param
        if not self.arch.use_token_anchor:
            fp.pop("use_resid_scale", None)
        if self.optimizer.lr_schedule != "wsd":
            fp.pop("decay_frac", None)

        return dict(sorted(fp.items()))
