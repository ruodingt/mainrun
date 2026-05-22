

# Arch search



```aiignore

block_size: int = 64
batch_size: int = 128
vocab_size: int = 8192

dropout: float = 0.1
muon_lr: float = 0.02  # Muon: 2D weight matrices (attn, mlp)
adamw_lr: float = 3e-4  # AdamW: norms, biases, other 1D params
emb_lr: float = 3e-3  # AdamW: token_emb — sparse updates justify slightly higher LR than base
scalar_lr: float = 1e-3  # AdamW: ReZero scalars, x0_lambdas
adamw_wd: float = 0.1  # weight decay for AdamW group only
lr_schedule: str = "wsd"  # "wsd" (warmup→stable→decay) | "cosine" (warmup→cosine decay)
warmup_frac: float = 0.05  # fraction of total steps for linear warmup
decay_frac: float = 0.30  # WSD only: fraction of total steps for final cosine decay
min_lr_frac: float = 0.0  # decay floor; 0.0 = decay all the way to zero

use_rezero: bool = False  # learnable per-layer residual scalars (ReZero); init=0 → identity at step 0
use_rmsnorm: bool = True  # RMSNorm instead of LayerNorm: faster, fewer params, no re-centering
use_token_anchor: bool = True  # per-layer learnable skip from original token embedding; prevents token identity dilution with depth
use_resid_scale: bool = False  # per-layer residual stream scaling (nanochat resid_lambdas); init 1.15→1.05; requires use_token_anchor=True
weight_init: str = "muon_uniform"  # weight init: "gpt2" (Normal 0.02) | "muon_uniform" (Uniform + zero exits; incompatible with use_rezero=True)
tie_weights: bool = True          # tie lm_head to token_emb; saves vocab*d_model params but prevents independent init
norm_emb: bool = False            # apply RMSNorm after embedding (nanochat style); required for token_emb std=0.8 to work safely
logit_softcap: float = 15.0       # tanh softcap on logits; 0.0 = disabled
n_kv_heads: int = 1  # n_kv_heads can be 1, 2, 4 to enable MQA
mlp_act: str = "relu_sq"  # gelu
```

With `python3 hypertune.py arch_sweep` with `ARCH_CANDIDATES = ARCH_CANDIDATES_V1`

============================================================


Arch sweep results (best first):
  [1] 1.2056  d=384 n=12 heads=6
  [2] 1.2067  d=384 n=14 heads=6
  [3] 1.2080  d=384 n=10 heads=6
  [4] 1.2159  d=512 n=8 heads=8
  [5] 1.2161  d=512 n=6 heads=8
  [6] 1.2178  d=512 n=7 heads=8
  [7] 1.2194  d=512 n=5 heads=8
============================================================


With `python3 hypertune.py arch_sweep` with `ARCH_CANDIDATES = ARCH_CANDIDATES_V2`



============================================================

Arch sweep results (best first):
  rank   val_loss   d_model   n_layer   heads   params_M
  [1]    1.2059    d=384    n=12     h=6     21.4M
  [2]    1.2061    d=384    n=14     h=6     24.5M
  [3]    1.2066    d=384    n=16     h=6     27.5M
  [4]    1.2111    d=448    n=9      h=7     22.3M
  [5]    1.2111    d=448    n=10     h=7     24.3M
  [6]    1.2125    d=448    n=8      h=7     20.2M
============================================================