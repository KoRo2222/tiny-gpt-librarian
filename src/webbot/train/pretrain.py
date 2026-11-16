from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from webbot.data import RandomBatches, sequential_batches
from webbot.model import GPT

from .checkpoint import load_checkpoint, save_checkpoint
from .precision import Precision, resolve_precision
from .schedule import d2z_scheduler
from .step import train_step


@dataclass
class PretrainConfig:
    total_steps: int
    batch_size: int = 16  # sequences per optimizer update
    grad_accum_steps: int = 1  # micro-batches the batch is split into
    seq_len: int = 256
    peak_lr: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    precision: str = "auto"
    seed: int = 0


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]
) -> torch.optim.AdamW:
    """AdamW with weight decay on weight matrices (incl. embeddings) only.

    1-D parameters -- the RMSNorm gains -- are excluded: decaying them would
    pull every layer's output scale toward zero for no benefit.
    """
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    precision: Precision,
    device: str | torch.device = "cpu",
) -> float:
    """Mean loss over batches (weighted by batch size), in eval mode."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for idx, targets in batches:
        with precision.autocast():
            _, loss = model(idx.to(device), targets.to(device))
        total += loss.item() * idx.shape[0]
        count += idx.shape[0]
    model.train(was_training)
    return total / count


def pretrain(
    model: GPT,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    config: PretrainConfig,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Next-token-prediction pretraining with AdamW + D2Z + mixed precision.

    Every eval_interval steps (and at the end) the validation loss is
    measured on a fixed set of windows and, if checkpoint_path is given, a
    checkpoint with the full training state is written. With resume=True
    training continues from that checkpoint and produces the same result as
    an uninterrupted run. Returns the evaluation history.

    Each update uses batch_size sequences; with grad_accum_steps > 1 they
    are processed as that many micro-batches (see train_step), which lowers
    activation memory without changing the update. The batches drawn do not
    depend on grad_accum_steps.
    """
    if config.seq_len > model.config.max_seq_len:
        raise ValueError(
            f"seq_len {config.seq_len} exceeds model max_seq_len {model.config.max_seq_len}"
        )
    torch.manual_seed(config.seed)
    model.to(device).train()
    precision = resolve_precision(device, config.precision)
    optimizer = make_optimizer(model, config.peak_lr, config.weight_decay, config.betas)
    scheduler = d2z_scheduler(optimizer, config.total_steps, config.warmup_steps)
    scaler = precision.make_grad_scaler()
    batches = RandomBatches(train_tokens, config.seq_len, config.batch_size, config.seed)

    step, history = 0, []
    if resume:
        ckpt = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        batches.load_state_dict(ckpt["batches"])
        step, history = ckpt["step"], ckpt["history"]
        log(f"resumed from {checkpoint_path} at step {step}")

    def run_eval() -> None:
        val_loss = evaluate(
            model,
            sequential_batches(val_tokens, config.seq_len, config.batch_size, config.eval_batches),
            precision,
            device,
        )
        history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                batches=batches,
                step=step,
                history=history,
                pretrain_config=vars(config),
            )
        # Logged after saving: once this line appears, the step is on disk.
        log(
            f"step {step:>6}/{config.total_steps} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} (ppl {math.exp(val_loss):.1f}) | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

    log(
        f"pretraining {model.num_params() / 1e6:.2f}M params, precision "
        f"{precision.name}, {config.total_steps} steps x {config.batch_size} x "
        f"{config.seq_len} tokens"
        + (f" ({config.grad_accum_steps} micro-batches)" if config.grad_accum_steps > 1 else "")
    )
    start = time.perf_counter()
    train_loss = float("nan")
    while step < config.total_steps:
        idx, targets = batches.next()
        train_loss = train_step(
            model,
            idx.to(device),
            targets.to(device),
            optimizer,
            precision,
            scaler,
            scheduler,
            config.max_grad_norm,
            config.grad_accum_steps,
        )
        step += 1
        elapsed = time.perf_counter() - start
        if step % config.eval_interval == 0 or step == config.total_steps:
            run_eval()
    return history
