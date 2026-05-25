import utils
import contextlib
import math
import os
import random
import sys
import time
from typing import Any

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import expt_util
from hparams import Hyperparameters
from hybrid import HybridLM
from optim import MuonAdamW
from tokenizer import BPETokenizer, train_tokenizer


# ---------------------------------------------------------------------------
# Pure helpers (no state)
# ---------------------------------------------------------------------------

def make_autocast(device: str, dtype: torch.dtype | None):
    if dtype is not None:
        return torch.amp.autocast(device_type=device, dtype=dtype)
    return contextlib.nullcontext()


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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, args: Hyperparameters):
        self.args = args
        self.device = args.runtime.device
        self.logger = None
        self.tb_writer = None
        self.train_ids = self.val_ids = self.val_text = self.tok = None
        self.model = self.opt = self.scheduler = None
        self.batches = self.max_steps = self.eval_interval = 0
        self.model_params = 0
        # named AdamW param groups — kept as refs for TensorBoard LR logging
        self.pg_emb = self.pg_scalar = self.pg_other = None

    # ------------------------------------------------------------------

    def _setup_experiment(self):
        args = self.args
        exp_dir = expt_util.get_or_create_experiment_dir(args, base_dir=args.runtime.experiments_dir)
        run_dir = expt_util.create_run_dir(exp_dir, args)
        # Write to the fixed path that checkpoint.mjs expects to find and rotate.
        self.logger = expt_util.configure_logging(args.runtime.log_file)
        self.tb_writer = SummaryWriter(log_dir=str(run_dir))
        self.exp_dir = exp_dir
        self.run_dir = run_dir
        self.logger.log("hyperparameters_configured", **args.flat_dict())
        self.logger.log("device_info", device=self.device)

    def _setup_data(self):
        args = self.args
        train_titles, val_titles = get_titles(args.fixed.num_titles, args.fixed.seed, args.fixed.val_frac)
        eos_token = "<eos>"
        tok = BPETokenizer(train_tokenizer(train_titles + val_titles, args.arch.vocab_size, eos_token=eos_token))
        train_text = eos_token.join(train_titles) + eos_token
        val_text   = eos_token.join(val_titles)   + eos_token
        self.tok       = tok
        self.train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
        self.val_ids   = torch.tensor(tok.encode(val_text),   dtype=torch.long)
        self.val_text  = val_text

        self.batches       = len(self.train_ids) // (args.train.block_size * args.train.batch_size)
        self.max_steps     = args.fixed.epochs * self.batches
        self.eval_interval = self.batches // args.runtime.evals_per_epoch

        print("vocab:", tok.vocab_size)
        args.arch.vocab_size = tok.vocab_size
        self.logger.log("dataset_info",
                        titles_count=len(train_titles),
                        epochs=args.fixed.epochs,
                        batches_per_epoch=self.batches,
                        tokens_per_epoch=len(self.train_ids),
                        vocab_size=tok.vocab_size)

    def _build_model(self):
        args = self.args
        model = HybridLM(args, self.tok.vocab_size).to(self.device)
        self.model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.log("model_info", parameters_count=self.model_params)
        expt_util.save_model_summary(model, self.exp_dir, self.run_dir)
        self.model = model

    def _compile_model(self):
        # Must happen AFTER param groups are collected — optimizer holds refs to original params.
        if self.args.runtime.use_compile and self.device != "cpu":
            self.model = torch.compile(self.model, mode="default")

    # ------------------------------------------------------------------
    # Optimizer builders — add new ones here, register in _OPTIMIZERS below.
    # ------------------------------------------------------------------

    def _build_muon_adamw(self):
        args = self.args
        _SCALAR_NAMES = ('x0_lambdas', 'resid_lambdas', 'mixer_scale', 'mlp_scale', 'attn_scale')
        muon_groups: dict[tuple, list] = {}
        emb_params, scalar_params, other_params = [], [], []

        seen_ids: set[int] = set()
        for name, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            if any(k in name for k in _SCALAR_NAMES):
                scalar_params.append(p)
            elif 'token_emb' in name:
                emb_params.append(p)
            elif p.ndim == 2 and 'conv' not in name and 'head' not in name:
                muon_groups.setdefault(tuple(p.shape), []).append(p)
            else:
                other_params.append(p)

        adamw_shared = {'kind': 'adamw', 'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': args.optimizer.adamw_wd}
        compute_dtype = torch.bfloat16 if args.runtime.use_bf16 else torch.float32
        self.pg_emb    = {**adamw_shared, 'params': emb_params,    'lr': args.optimizer.emb_lr}
        self.pg_scalar = {**adamw_shared, 'params': scalar_params, 'lr': args.optimizer.scalar_lr}
        self.pg_other  = {**adamw_shared, 'params': other_params,  'lr': args.optimizer.adamw_lr}
        param_groups = [
            *[{'kind': 'muon', 'params': ps, 'lr': args.optimizer.muon_lr,
               'momentum': 0.95, 'ns_steps': 5, 'beta2': 0.999, 'weight_decay': 0.0}
              for ps in muon_groups.values()],
            self.pg_emb, self.pg_scalar, self.pg_other,
        ]
        self.opt = MuonAdamW(param_groups, compute_dtype=compute_dtype)
        self._build_lr_schedule()

    def _build_sgd(self):
        args = self.args
        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=args.optimizer.sgd_lr,
            weight_decay=args.optimizer.sgd_wd,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.max_steps)

    def _build_lr_schedule(self):
        """WSD or cosine LambdaLR — used by muon_adamw."""
        args = self.args
        warmup_steps = int(self.max_steps * args.optimizer.warmup_frac)
        if args.optimizer.lr_schedule == "wsd":
            decay_steps  = int(self.max_steps * args.optimizer.decay_frac)
            stable_steps = self.max_steps - warmup_steps - decay_steps
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                if step < warmup_steps + stable_steps:
                    return 1.0
                progress = (step - warmup_steps - stable_steps) / max(decay_steps, 1)
                return args.optimizer.min_lr_frac + (1 - args.optimizer.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(self.max_steps - warmup_steps, 1)
                return args.optimizer.min_lr_frac + (1 - args.optimizer.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    _OPTIMIZERS = {
        "muon_adamw": _build_muon_adamw,
        "sgd":        _build_sgd,
    }

    def _build_optimizer(self):
        build_fn = self._OPTIMIZERS[self.args.optimizer.optimizer_type]
        build_fn(self)

    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        args = self.args
        self.model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(self.val_ids, args.train.block_size, args.train.batch_size, torch.device(self.device)):
                logits, _ = self.model(xb, yb)
                B, T, V = logits.size()
                losses += F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum').item()
        self.model.train()
        return losses / len(self.val_text)

    def train(self):
        args = self.args
        amp_dtype = torch.bfloat16 if args.runtime.use_bf16 else None
        tokens_per_step = args.train.block_size * args.train.batch_size
        ptr = step = 0
        val_loss = 0.0
        t0 = time.time()
        total_compute_time = 0.0  # cumulative compute-only time (excludes data load + eval)

        for epoch in range(1, args.fixed.epochs + 1):
            for _ in tqdm(range(1, self.batches + 1), desc=f"Epoch {epoch}/{args.fixed.epochs}",
                          disable=not args.runtime.use_tqdm):
                step_start = time.time()
                step += 1

                xb, yb, ptr = get_batch(self.train_ids, ptr, args.train.block_size, args.train.batch_size, self.device)
                t_data = time.time()

                with make_autocast(self.device, amp_dtype):
                    _, loss = self.model(xb, yb)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()

                if args.train.clip_norm_mode == "adamw" and self.pg_emb is not None:
                    # muon_adamw only: skip Muon params (self-normalizing via Newton-Schulz)
                    clip_params = [p for g in self.opt.param_groups if g.get('kind') == 'adamw' for p in g['params']]
                else:
                    clip_params = list(self.model.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)

                self.opt.step()
                self.scheduler.step()
                if self.device != "cpu":
                    torch.cuda.synchronize()
                t1 = time.time()

                step_time    = t1 - step_start
                data_time    = t_data - step_start
                compute_time = t1 - t_data
                elapsed      = t1 - t0
                total_compute_time += compute_time
                local_tok_s = tokens_per_step / compute_time if compute_time > 0 else 0.0
                avg_tok_s   = (step * tokens_per_step) / total_compute_time if total_compute_time > 0 else 0.0

                if torch.cuda.is_available():
                    alloc_gb     = torch.cuda.memory_allocated() / 1e9
                    reserved_gb  = torch.cuda.memory_reserved() / 1e9
                    max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
                else:
                    alloc_gb = reserved_gb = max_alloc_gb = 0.0

                self.logger.log("training_step", step=step, max_steps=self.max_steps,
                                loss=loss.item(), elapsed_time=elapsed, prnt=False)

                if self.tb_writer:
                    tb = self.tb_writer
                    tb.add_scalar("Loss/train", loss.item(), step)
                    if self.pg_emb:  # muon_adamw path
                        tb.add_scalar("Charts/muon_lr",   self.opt.param_groups[0]['lr'], step)
                        tb.add_scalar("Charts/emb_lr",    self.pg_emb['lr'],    step)
                        tb.add_scalar("Charts/scalar_lr", self.pg_scalar['lr'], step)
                        tb.add_scalar("Charts/adamw_lr",  self.pg_other['lr'],  step)
                    else:            # sgd path
                        tb.add_scalar("Charts/lr", self.opt.param_groups[0]['lr'], step)
                    tb.add_scalar("Throughput/local_tokens_per_sec", local_tok_s, step)
                    tb.add_scalar("Throughput/avg_tokens_per_sec",   avg_tok_s,   step)
                    tb.add_scalar("Bottleneck/data_ms",    data_time    * 1000, step)
                    tb.add_scalar("Bottleneck/compute_ms", compute_time * 1000, step)
                    tb.add_scalar("Bottleneck/data_pct",   data_time / step_time * 100, step)
                    tb.add_scalar("Memory/allocated_GB",     alloc_gb,     step)
                    tb.add_scalar("Memory/reserved_GB",      reserved_gb,  step)
                    tb.add_scalar("Memory/max_allocated_GB", max_alloc_gb, step)

                if step == 1 or step % self.eval_interval == 0 or step == self.max_steps:
                    val_loss = self.evaluate()
                    self.logger.log("validation_step", step=step, max_steps=self.max_steps,
                                    loss=val_loss, elapsed_time=elapsed)
                    if self.tb_writer:
                        self.tb_writer.add_scalar("Loss/val", val_loss, step)

        if self.tb_writer:
            from torch.utils.tensorboard.summary import hparams as tb_hparams
            hparam_dict = args.get_fingerprint()
            metric_dict = {
                "Loss/val_final":     val_loss,
                "Perf/total_time_min": (time.time() - t0) / 60,
                "Model/param_count":  float(self.model_params),
            }
            exp, ssi, sei = tb_hparams(hparam_dict, metric_dict)
            self.tb_writer.file_writer.add_summary(exp)
            self.tb_writer.file_writer.add_summary(ssi)
            self.tb_writer.file_writer.add_summary(sei)

        expt_util.save_run_results(self.run_dir, val_loss, time.time() - t0,
                                   avg_tok_s=avg_tok_s, total_params=self.model_params)

        # Archive a copy of the log into the run dir for experiment record-keeping.
        import shutil
        shutil.copy(self.args.runtime.log_file, self.run_dir / "log.txt")

    # ------------------------------------------------------------------

    def run(self):
        self._setup_experiment()
        self._setup_data()
        self._build_model()
        self._build_optimizer()
        self._compile_model()
        self.train()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

    args = Hyperparameters()
    if len(sys.argv) > 1:
        args.update_from_flat(json.loads(sys.argv[1]))
    torch.manual_seed(args.fixed.seed)
    random.seed(args.fixed.seed)

    trainer = Trainer(args)
    try:
        trainer.run()
    finally:
        if trainer.logger and hasattr(trainer.logger, 'file_handler'):
            trainer.logger.file_handler.close()
        if trainer.tb_writer:
            trainer.tb_writer.close()
