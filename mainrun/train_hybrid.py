# import utils
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import expt_util
from optim import MuonAdamW
from tokenizer import BPETokenizer, train_tokenizer
from train import GPT, GPTConfig


@dataclass
class Hyperparameters:
    block_size: int = 64
    batch_size: int = 128
    vocab_size: int = 8192
    n_layer: int = 12
    n_q_head: int = 6  # number of query head
    d_model: int = 384
    dropout: float = 0.1
    muon_lr: float = 0.02  # Muon: 2D weight matrices (attn, mlp)
    adamw_lr: float = 3e-4  # AdamW: norms, biases, other 1D params
    emb_lr: float = 3e-3  # AdamW: token_emb — sparse updates justify slightly higher LR than base
    scalar_lr: float = 1e-3  # AdamW: ReZero scalars, x0_lambdas
    adamw_wd: float = 0.1  # weight decay for AdamW group only
    lr_schedule: str = "wsd"  # "wsd" (warmup→stable→decay) | "cosine" (warmup→cosine decay)
    warmup_frac: float = 0.05  # fraction of total steps for linear warmup
    decay_frac: float = 0.50  # WSD only: fraction of total steps for final cosine decay
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

    # Hybrid Mamba/Attention (only used when layer_pattern contains 'M')
    # "AAAAAAAAAAAA" = pure Transformer; "MMMAMMMAMMMA" = Jamba-ish; "MAMAMAMAMAMA" = Samba-style
    layer_pattern: str = "A" * 12
    expand: int = 2      # Mamba d_inner = expand * d_model
    d_head: int = 64     # Mamba SSM per-head dim
    d_state: int = 128   # Mamba SSM state dim
    n_groups: int = 1    # SSD groups (B/C shared across heads per group)
    d_conv: int = 4      # Mamba depthwise conv kernel size
    chunk_len: int = 64  # SSD chunk length; must divide block_size

    # irrelevant/minor to training / loss
    use_fa2: bool = True
    evals_per_epoch: int = 3
    use_fused_ce: bool = False  # fused linear+CE kernel (memory efficient; ~0.6x speed on RDNA4, faster on CDNA3)
    clip_norm_mode: str = "all"  # gradient clip scope: "all" (all params) | "adamw" (AdamW group only; Muon self-normalizes)
    use_compile: bool = True  # torch.compile with reduce-overhead (auto CUDA graphs); skip on CPU
    use_bf16: bool = True
    experiments_dir: str = "./experiments"

    # fixed
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

    # Fields that don't affect model quality — excluded from experiment fingerprint.
    _INFRA = frozenset(['use_fa2', 'use_compile', 'use_bf16', 'evals_per_epoch', 'experiments_dir', 'log_file'])
    _FIXED = frozenset(['epochs', 'seed', 'num_titles', 'val_frac'])

    def get_fingerprint(self) -> dict:
        ignore = self._INFRA | self._FIXED
        return {k: v for k, v in vars(self).items() if k not in ignore}


import contextlib


def make_autocast(device: str, dtype: torch.dtype | None):
    if dtype is not None:
        return torch.amp.autocast(device_type=device, dtype=dtype)
    return contextlib.nullcontext()


logger = None
tb_writer = None


def disk_cache(cache_dir: str = "./data"):
    import hashlib, json, pickle
    from pathlib import Path
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = hashlib.md5(
                json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
            path = Path(cache_dir) / f"{fn.__name__}_{key}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
            result = fn(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator


@disk_cache("./data")
def get_titles(num_titles: int, seed: int, val_frac: float) -> tuple[list[Any], list[Any]]:
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


def main():
    import json, sys
    args = Hyperparameters()
    if len(sys.argv) > 1:
        for k, v in json.loads(sys.argv[1]).items():
            assert hasattr(args, k), f"unknown hyperparam: {k!r}"
            setattr(args, k, v)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # TF32 is NVIDIA-only; no-op on AMD ROCm. Skip to avoid triggering hipBLASLt paths.

    # Resolve experiment and run directories
    exp_dir = expt_util.get_or_create_experiment_dir(args, base_dir=args.experiments_dir)
    run_dir = expt_util.create_run_dir(exp_dir, args)
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
    tok = BPETokenizer(train_tokenizer(train_titles + val_titles, args.vocab_size, eos_token=eos_token))
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
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_q_head=args.n_q_head,
        n_kv_heads=args.n_kv_heads,
        d_model=args.d_model,
        dropout=args.dropout,
        use_fa2=args.use_fa2,
        use_rezero=args.use_rezero,
        use_rmsnorm=args.use_rmsnorm,
        use_token_anchor=args.use_token_anchor,
        use_resid_scale=args.use_resid_scale,
        weight_init=args.weight_init,
        tie_weights=args.tie_weights,
        norm_emb=args.norm_emb,
        logit_softcap=args.logit_softcap,
        mlp_act=args.mlp_act,
        use_fused_ce=args.use_fused_ce,
    )
    if 'M' in args.layer_pattern:
        from hybrid import HybridLM, HybridConfig
        hybrid_cfg = HybridConfig(
            vocab_size=tok.vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            layer_pattern=args.layer_pattern,
            dropout=args.dropout,
            n_q_head=args.n_q_head,
            n_kv_heads=args.n_kv_heads,
            use_fa2=args.use_fa2,
            expand=args.expand,
            d_head=args.d_head,
            d_state=args.d_state,
            n_groups=args.n_groups,
            d_conv=args.d_conv,
            chunk_len=args.chunk_len,
            use_rmsnorm=args.use_rmsnorm,
            use_rezero=args.use_rezero,
            tie_weights=args.tie_weights,
            norm_emb=args.norm_emb,
            logit_softcap=args.logit_softcap,
            mlp_act=args.mlp_act,
        )
        model = HybridLM(hybrid_cfg).to(device)
    else:
        model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    # Save a comprehensive model summary to the experiment and run directories
    expt_util.save_model_summary(model, exp_dir, run_dir)

    # Mixed precision: autocast handles bf16 matmuls automatically (weight storage stays fp32,
    # actual matmuls run in bf16). Optimizer states remain fp32 (handled in optim.py).
    # This is the industrial standard — autocast is smarter than manual casting because it
    # only downcasts ops that are safe in reduced precision (matmul, conv), and leaves
    # accumulations and softmax in fp32.
    amp_dtype = torch.bfloat16 if args.use_bf16 else None  # None = no autocast

    # MuonAdamW: Muon for 2D weight matrices, AdamW split into three groups by LR:
    #   emb_group   : token_emb — large LR, embeddings are slow to converge with small LR
    #   scalar_group: ReZero scalars + x0_lambdas — fast-moving, benefit from high LR
    #   other_group : norms, biases, remaining 1D params — standard LR
    # head.weight excluded from Muon regardless of tying (output proj → AdamW).
    scalar_ids = set()
    if hasattr(model, 'x0_lambdas'):
        scalar_ids.add(id(model.x0_lambdas))
    if hasattr(model, 'resid_lambdas'):
        scalar_ids.add(id(model.resid_lambdas))

    for block in model.blocks:
        if hasattr(block, 'attn_scale'):
            scalar_ids.add(id(block.attn_scale))
            scalar_ids.add(id(block.mlp_scale))

    skip_ids = {id(model.token_emb.weight), id(model.head.weight)}
    muon_groups: dict[tuple, list] = {}
    emb_params, scalar_params, other_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in skip_ids:
            continue  # handled separately below
        if p.ndim >= 2:
            if p.ndim == 2:
                # Muon handles square-ish 2D weight matrices (attn projections, MLP weights, Mamba in/out proj).
                muon_groups.setdefault(tuple(p.shape), []).append(p)
            else:
                # 3D+ params (e.g. Mamba conv1d.weight shape [d_inner, 1, d_conv]):
                # orthogonalizing a depthwise conv kernel is undefined/harmful → AdamW.
                other_params.append(p)
        elif id(p) in scalar_ids:
            scalar_params.append(p)
        else:
            other_params.append(p)

    emb_params.append(model.token_emb.weight)

    adamw_shared = {'kind': 'adamw', 'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': args.adamw_wd}
    compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    param_groups = [
        *[{'kind': 'muon', 'params': ps, 'lr': args.muon_lr,
           'momentum': 0.95, 'ns_steps': 5, 'beta2': 0.999, 'weight_decay': 0.0}
          for ps in muon_groups.values()],
        {**adamw_shared, 'params': emb_params, 'lr': args.emb_lr},
        {**adamw_shared, 'params': scalar_params, 'lr': args.scalar_lr},
        {**adamw_shared, 'params': other_params, 'lr': args.adamw_lr},
    ]
    if not args.tie_weights:
        # lm_head is independent — use adamw_lr (output proj, not an embedding)
        param_groups.append({**adamw_shared, 'params': [model.head.weight], 'lr': args.adamw_lr})
    opt = MuonAdamW(param_groups, compute_dtype=compute_dtype)

    # torch.compile: fuses kernels and (with reduce-overhead) captures CUDA graphs.
    # Must happen AFTER param groups are collected — optimizer holds refs to original params.
    # Skip on CPU (compile gives no benefit and slows startup).
    if args.use_compile and device != "cpu":
        model = torch.compile(model, mode="reduce-overhead")

    assert args.lr_schedule in ("wsd", "cosine"), f"unknown lr_schedule: {args.lr_schedule!r}"
    warmup_steps = int(max_steps * args.warmup_frac)

    if args.lr_schedule == "wsd":
        decay_steps = int(max_steps * args.decay_frac)
        stable_steps = max_steps - warmup_steps - decay_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if step < warmup_steps + stable_steps:
                return 1.0
            progress = (step - warmup_steps - stable_steps) / max(decay_steps, 1)
            return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    else:  # cosine
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, torch.device(device)):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        # here we use length of char because we want to normalize across tokenizers
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
            t_data = time.time()

            with make_autocast(device, amp_dtype):
                _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_norm_mode == "adamw":
                clip_params = [p for g in opt.param_groups if g['kind'] == 'adamw' for p in g['params']]
            else:
                clip_params = list(model.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            scheduler.step()
            if device != "cpu":
                torch.cuda.synchronize()  # wait for GPU before stopping the clock
            t1 = time.time()

            step_time = t1 - step_start
            data_time = t_data - step_start  # CPU: data fetch + H2D transfer
            compute_time = t1 - t_data  # GPU: forward + backward + optimizer
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
                tb_writer.add_scalar("Charts/emb_lr", opt.param_groups[-3]['lr'], step)
                tb_writer.add_scalar("Charts/scalar_lr", opt.param_groups[-2]['lr'], step)
                tb_writer.add_scalar("Charts/adamw_lr", opt.param_groups[-1]['lr'], step)
                tb_writer.add_scalar("Throughput/local_tokens_per_sec", local_tokens_per_sec, step)
                tb_writer.add_scalar("Throughput/avg_tokens_per_sec", avg_tokens_per_sec, step)
                tb_writer.add_scalar("Bottleneck/data_ms", data_time * 1000, step)
                tb_writer.add_scalar("Bottleneck/compute_ms", compute_time * 1000, step)
                tb_writer.add_scalar("Bottleneck/data_pct", data_time / step_time * 100, step)
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

    # Record final hyperparameters and metrics.
    # In TensorBoard's HParams tab this becomes an interactive scatter-plot matrix:
    # pick any two axes (hparam or metric) and each run plots as a single point.
    # total_time_min + param_count let you see the loss/speed/size trade-off across runs.
    #
    # We write directly into the existing FileWriter instead of calling add_hparams(),
    # because add_hparams() creates an ugly timestamped subdirectory (known TB bug).
    if tb_writer:
        from torch.utils.tensorboard.summary import hparams as tb_hparams
        total_time_min = (time.time() - t0) / 60
        hparam_dict = args.get_fingerprint()
        metric_dict = {
            "Loss/val_final": val_loss,
            "Perf/total_time_min": total_time_min,
            "Model/param_count": float(model_params),
        }
        exp, ssi, sei = tb_hparams(hparam_dict, metric_dict)
        tb_writer.file_writer.add_summary(exp)
        tb_writer.file_writer.add_summary(ssi)
        tb_writer.file_writer.add_summary(sei)

    expt_util.save_run_results(run_dir, val_loss, time.time() - t0)


if __name__ == "__main__":
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    # hipBLASLt rejects certain small/non-standard matrix shapes common in this model
    # (e.g. bmm with head_dim=64, MQA kv shapes). Fall back to standard hipBLAS which
    # supports all shapes. No accuracy loss; avoids noisy HIPBLAS_STATUS_NOT_SUPPORTED warnings.
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
        if tb_writer:
            tb_writer.close()
