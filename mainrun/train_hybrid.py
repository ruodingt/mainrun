# import utils
import math
import os
import random
import time
from typing import Any

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import expt_util
from hparams import Hyperparameters
from optim import MuonAdamW
from tokenizer import BPETokenizer, train_tokenizer
from hybrid import HybridLM

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
        args.update_from_flat(json.loads(sys.argv[1]))

    torch.manual_seed(args.fixed.seed)
    random.seed(args.fixed.seed)

    # Resolve experiment and run directories
    exp_dir = expt_util.get_or_create_experiment_dir(args, base_dir=args.runtime.experiments_dir)
    run_dir = expt_util.create_run_dir(exp_dir, args)
    args.runtime.log_file = str(run_dir / "log.txt")

    global logger, tb_writer
    logger = expt_util.configure_logging(args.runtime.log_file)
    tb_writer = SummaryWriter(log_dir=str(run_dir))

    logger.log("hyperparameters_configured", **args.flat_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.fixed.num_titles, args.fixed.seed, args.fixed.val_frac)

    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles + val_titles, args.arch.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    batches = len(train_ids) // (args.train.block_size * args.train.batch_size)
    max_steps = args.fixed.epochs * batches
    eval_interval = batches // args.runtime.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.fixed.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    print("vocab:", tok.vocab_size)
    args.arch.vocab_size = tok.vocab_size  # sync actual vocab size back (tokenizer may round)

    model = HybridLM(args, tok.vocab_size).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    expt_util.save_model_summary(model, exp_dir, run_dir)

    amp_dtype = torch.bfloat16 if args.runtime.use_bf16 else None  # None = no autocast

    # MuonAdamW: Muon for 2D weight matrices, AdamW split into three groups by LR:
    #   emb_group   : token_emb — large LR, embeddings are slow to converge with small LR
    #   scalar_group: rezero scales + x0_lambdas — fast-moving, benefit from high LR
    #   other_group : norms, biases, remaining 1D/3D params — standard LR
    _SCALAR_NAMES = ('x0_lambdas', 'resid_lambdas', 'mixer_scale', 'mlp_scale', 'attn_scale')

    muon_groups: dict[tuple, list] = {}
    emb_params, scalar_params, other_params = [], [], []

    seen_ids: set[int] = set()  # dedup tied weights (token_emb.weight == head.weight when tied)
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        if any(k in name for k in _SCALAR_NAMES):  # rezero scales, token anchor lambdas
            scalar_params.append(p)
        elif 'token_emb' in name:                   # sparse updates → own high-LR group
            emb_params.append(p)
        elif p.ndim == 2 and 'conv' not in name and 'head' not in name:
            muon_groups.setdefault(tuple(p.shape), []).append(p)
        else:
            # 1D (norms, biases), 3D conv (Muon ortho degenerates on (d,1,k)), head.weight
            other_params.append(p)

    adamw_shared = {'kind': 'adamw', 'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': args.optimizer.adamw_wd}
    compute_dtype = torch.bfloat16 if args.runtime.use_bf16 else torch.float32
    pg_emb    = {**adamw_shared, 'params': emb_params,    'lr': args.optimizer.emb_lr}
    pg_scalar = {**adamw_shared, 'params': scalar_params, 'lr': args.optimizer.scalar_lr}
    pg_other  = {**adamw_shared, 'params': other_params,  'lr': args.optimizer.adamw_lr}
    param_groups = [
        *[{'kind': 'muon', 'params': ps, 'lr': args.optimizer.muon_lr,
           'momentum': 0.95, 'ns_steps': 5, 'beta2': 0.999, 'weight_decay': 0.0}
          for ps in muon_groups.values()],
        pg_emb, pg_scalar, pg_other,
    ]
    opt = MuonAdamW(param_groups, compute_dtype=compute_dtype)

    # torch.compile must happen AFTER param groups are collected (optimizer holds refs to original params).
    if args.runtime.use_compile and device != "cpu":
        model = torch.compile(model, mode="reduce-overhead")

    warmup_steps = int(max_steps * args.optimizer.warmup_frac)

    if args.optimizer.lr_schedule == "wsd":
        decay_steps = int(max_steps * args.optimizer.decay_frac)
        stable_steps = max_steps - warmup_steps - decay_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if step < warmup_steps + stable_steps:
                return 1.0
            progress = (step - warmup_steps - stable_steps) / max(decay_steps, 1)
            return args.optimizer.min_lr_frac + (1 - args.optimizer.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    else:  # cosine
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return args.optimizer.min_lr_frac + (1 - args.optimizer.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.train.block_size, args.train.batch_size, torch.device(device)):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        # Normalize by char count (not token count) for tokenizer-agnostic nats/char metric.
        return losses / len(val_text)

    ptr = 0
    step = 0
    val_loss = 0.0
    t0 = time.time()
    tokens_per_step = args.train.block_size * args.train.batch_size
    for epoch in range(1, args.fixed.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.fixed.epochs}"):
            step_start = time.time()
            step += 1

            xb, yb, ptr = get_batch(train_ids, ptr, args.train.block_size, args.train.batch_size, device)
            t_data = time.time()

            with make_autocast(device, amp_dtype):
                _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.train.clip_norm_mode == "adamw":
                clip_params = [p for g in opt.param_groups if g['kind'] == 'adamw' for p in g['params']]
            else:
                clip_params = list(model.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            scheduler.step()
            if device != "cpu":
                torch.cuda.synchronize()
            t1 = time.time()

            step_time = t1 - step_start
            data_time = t_data - step_start
            compute_time = t1 - t_data
            elapsed = t1 - t0

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
                tb_writer.add_scalar("Charts/emb_lr", pg_emb['lr'], step)
                tb_writer.add_scalar("Charts/scalar_lr", pg_scalar['lr'], step)
                tb_writer.add_scalar("Charts/adamw_lr", pg_other['lr'], step)
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

    # TensorBoard HParams tab: interactive scatter-plot matrix across runs.
    # Written directly into the existing FileWriter to avoid add_hparams()'s
    # timestamped-subdirectory bug.
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
    # hipBLASLt rejects certain small/non-standard matrix shapes common in this model.
    # Fall back to standard hipBLAS. No accuracy loss.
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
        if tb_writer:
            tb_writer.close()
