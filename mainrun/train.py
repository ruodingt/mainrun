# import utils
import math, random, time
import os
from dataclasses import dataclass
import expt_util
from rope import apply_rotary_emb, RotaryEmbedding
from optim import MuonAdamW

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm

@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 6
    n_q_head: int = 8 # number of query head
    d_model: int = 512
    dropout: float = 0.1
    muon_lr: float = 0.02      # Muon: 2D weight matrices (attn, mlp)
    adamw_lr: float = 3e-4     # AdamW: embeddings, norms, biases
    adamw_wd: float = 0.1      # weight decay for AdamW group only
    evals_per_epoch: int = 3

    use_fa2: bool = True
    use_bf16: bool = True
    n_kv_heads: int = 8 # n_kv_heads can be 1, 2, 4 to enable MQA
    
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

    def get_fingerprint(self, ignore: list[str] | None = None) -> dict:
        if ignore is None:
            ignore = ['seed']
        return {k: v for k, v in vars(self).items() if k not in ignore}



logger = None
tb_writer = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_q_head: int
    d_model: int
    dropout: float
    # Explicit control for MQA/GQA; defaults to n_q_head (standard MHA mode)
    n_kv_heads: int = 8
    # Enable SDPA (FlashAttention-2) by default to leverage hardware-level acceleration on RDNA 3.5 UMA
    use_fa2: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_q_head == 0
        self.n_q_head = cfg.n_q_head
        self.head_dim = cfg.d_model // cfg.n_q_head

        # Dynamic support: n_kv_heads = n_q_head (MHA), n_kv_heads = 1 (MQA), 1 < n_kv_heads < n_q_head (GQA)
        self.n_kv_heads = getattr(cfg, "n_kv_heads", cfg.n_q_head)
        self.use_sdpa = getattr(cfg, "use_fa2", True)  # Static switch for SDPA
        self.dropout_p = cfg.dropout

        # [Muon Overclocking Core Design: Decoupled Projections]
        # Isolate q_proj so its shape strictly equals (d_model, d_model) -> e.g., 512x512.
        # This allows q_proj to be perfectly stacked with the final proj (512x512) into a single Muon optimizer group!
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # kv_proj handles variable KV heads. Under MQA, its shape is only (d_model, 2 * head_dim) -> e.g., 512x128.
        # Since this matrix is very small, it can be assigned to the AdamW group or a separate Muon group,
        # preserving the stack structure of the main Muon group.
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)

        # Output projection layer: Shape strictly equals (d_model, d_model) -> perfectly aligned with q_proj.
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Prevent state contamination: register tril buffer only in manual debugging mode.
        if not self.use_sdpa:
            tril = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
            self.register_buffer("tril", tril, persistent=False)

    def forward(self, x: torch.Tensor, cos_sin) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Project Q and KV
        # q shape: (B, T, n_q_head, head_dim) -> transpose to (B, n_q_head, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_q_head, self.head_dim).transpose(1, 2)

        # kv shape: (B, T, 2, n_kv_heads, head_dim) -> transpose to (B, n_kv_heads, T, head_dim)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).transpose(1, 3)
        k, v = kv[..., 0, :, :], kv[..., 1, :, :]

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # 2. Static conditional routing
        if self.use_sdpa:
            # PyTorch SDPA natively supports GQA/MQA broadcasting (when n_q_head % n_kv_heads == 0).
            # Automatically activates hardware-level Causal FlashAttention acceleration under the hood (e.g., RDNA 3.5).
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Backup/debug path manually supporting GQA/MQA broadcasting.
            if self.n_q_head != self.n_kv_heads:
                # Broadcast along the head dimension to align with Q's head count.
                num_queries_per_kv = self.n_q_head // self.n_kv_heads
                k = k.repeat_interleave(num_queries_per_kv, dim=1)
                v = v.repeat_interleave(num_queries_per_kv, dim=1)

            # Classic white-box dot-product attention calculation.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        # 3. Restore dimensions and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
    def forward(self, x, cos_sin):
        x = x + self.attn(self.ln1(x), cos_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)

        head_dim = cfg.d_model // cfg.n_q_head
        self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.block_size)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight



    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        x = self.drop(self.token_emb(idx))
        cos_sin = self.rope(x, T)
        for block in self.blocks: x = block(x, cos_sin)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Resolve experiment and run directories
    exp_dir = expt_util.get_or_create_experiment_dir(args)
    run_dir = expt_util.create_run_dir(exp_dir, args.seed)
    args.log_file = str(run_dir / "log.txt")
    
    global logger, tb_writer
    logger = expt_util.configure_logging(args.log_file)
    tb_writer = SummaryWriter(log_dir=str(run_dir))
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    print("vocab:", tok.vocab_size)
    # exit()
    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_q_head   = args.n_q_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        use_fa2    = args.use_fa2
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    # Save a comprehensive model summary to the experiment and run directories
    expt_util.save_model_summary(model, exp_dir, run_dir)
    
    # MuonAdamW: Muon for 2D weight matrices, AdamW for embeddings/norms/biases.
    # Muon should not be used for embedding or head layers (see optim.py docstring).
    # head.weight is tied to token_emb.weight, so excluding one excludes both.
    skip_ids = {id(model.token_emb.weight), id(model.head.weight)}
    muon_groups: dict[tuple, list] = {}
    adamw_params: list = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in skip_ids or p.ndim < 2:
            adamw_params.append(p)
        else:
            muon_groups.setdefault(tuple(p.shape), []).append(p)

    param_groups = [
        *[{'kind': 'muon', 'params': ps, 'lr': args.muon_lr,
           'momentum': 0.95, 'ns_steps': 5, 'beta2': 0.999, 'weight_decay': 0.0}
          for ps in muon_groups.values()],
        {'kind': 'adamw', 'params': adamw_params, 'lr': args.adamw_lr,
         'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': args.adamw_wd},
    ]
    compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    opt = MuonAdamW(param_groups, compute_dtype=compute_dtype)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    val_loss = 0.0
    t0 = time.time()
    tokens_per_step = args.block_size * args.batch_size
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step_start = time.time()
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            t1 = time.time()
            step_time = t1 - step_start
            elapsed = t1 - t0
            
            # Calculate performance metrics
            local_tokens_per_sec = tokens_per_step / step_time if step_time > 0 else 0.0
            avg_tokens_per_sec = (step * tokens_per_step) / elapsed if elapsed > 0 else 0.0
            
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                max_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
            else:
                allocated_gb = 0.0
                reserved_gb = 0.0
                max_allocated_gb = 0.0

            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if tb_writer:
                tb_writer.add_scalar("Loss/train", loss.item(), step)
                tb_writer.add_scalar("Charts/muon_lr", opt.param_groups[0]['lr'], step)
                tb_writer.add_scalar("Charts/adamw_lr", opt.param_groups[-1]['lr'], step)
                tb_writer.add_scalar("Throughput/local_tokens_per_sec", local_tokens_per_sec, step)
                tb_writer.add_scalar("Throughput/avg_tokens_per_sec", avg_tokens_per_sec, step)
                tb_writer.add_scalar("Memory/allocated_GB", allocated_gb, step)
                tb_writer.add_scalar("Memory/reserved_GB", reserved_gb, step)
                tb_writer.add_scalar("Memory/max_allocated_GB", max_allocated_gb, step)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)
                
                if tb_writer:
                    tb_writer.add_scalar("Loss/val", val_loss, step)

    # Record final hyperparameters and metric
    if tb_writer:
        tb_writer.add_hparams(
            hparam_dict=args.get_fingerprint(ignore=['log_file']),
            metric_dict={"Loss/val_final": val_loss}
        )

if __name__ == "__main__":
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
        if tb_writer:
            tb_writer.close()
