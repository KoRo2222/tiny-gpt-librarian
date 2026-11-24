"""Supervised fine-tuning on search-read-answer exchanges.

Examples have different lengths: a batch is padded to its longest example
with <|pad|>, and every target the model should not learn (question,
paragraphs, padding) is -100, which cross-entropy ignores, so the loss is
the mean over the tokens the model writes (per micro-batch). Each step
draws its batch at random; the generator's state is checkpointed, so an
interrupted run resumes exactly. Otherwise the loop matches pretraining:
AdamW, D2Z, mixed precision, clipping, gradient accumulation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from webbot.train.checkpoint import load_checkpoint, save_checkpoint
from webbot.train.precision import resolve_precision
from webbot.train.pretrain import make_optimizer
from webbot.train.schedule import d2z_scheduler
from webbot.train.step import train_step

IGNORE = -100
Encoded = tuple[list[int], list[bool]]


def make_batch(
    examples: Sequence[Encoded], pad_id: int, pad_multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """(input ids, targets), both (batch, width - 1): targets are the next
    tokens where the model writes them, IGNORE elsewhere and on padding.
    width is the longest example, rounded up to pad_multiple (on XLA devices
    every new width compiles a new graph; rounding keeps them few)."""
    width = max(len(ids) for ids, _ in examples)
    width = -(-(width - 1) // pad_multiple) * pad_multiple + 1
    idx = torch.full((len(examples), width), pad_id, dtype=torch.long)
    tgt = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    for row, (ids, mask) in enumerate(examples):
        t = torch.tensor(ids)
        idx[row, : len(ids)] = t
        tgt[row, : len(ids)] = torch.where(torch.tensor(mask), t, torch.full_like(t, IGNORE))
    return idx[:, :-1], tgt[:, 1:]


class SFTBatches:
    """Endless random batches of encoded examples, resumable."""

    def __init__(self, examples: Sequence[Encoded], batch_size: int, pad_id: int, seed: int = 0,
                 pad_multiple: int = 1):
        if not examples:
            raise ValueError("no examples")
        self.examples = examples
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.pad_multiple = pad_multiple
        self.rng = np.random.default_rng(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.rng.integers(0, len(self.examples), size=self.batch_size)
        return make_batch([self.examples[i] for i in rows], self.pad_id, self.pad_multiple)

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]


@torch.no_grad()
def evaluate_sft(
    model: nn.Module, examples: Sequence[Encoded], pad_id: int, batch_size: int,
    precision, device: str | torch.device = "cpu", pad_multiple: int = 1,
) -> float:
    """Mean loss over every written token of the examples (token-weighted,
    in order), in eval mode."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        idx, tgt = make_batch(examples[start : start + batch_size], pad_id, pad_multiple)
        with precision.autocast():
            logits, _ = model(idx.to(device))
        logits = logits.float()
        tgt = tgt.to(device)
        total += nn.functional.cross_entropy(
            logits.flatten(0, 1), tgt.flatten(), ignore_index=IGNORE, reduction="sum"
        ).item()
        count += int((tgt != IGNORE).sum())
    model.train(was_training)
    return total / count if count else float("nan")  # no examples: nothing to measure


@dataclass
class SFTConfig:
    total_steps: int
    batch_size: int = 32
    peak_lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    eval_interval: int = 200
    precision: str = "auto"
    seed: int = 0
    pad_multiple: int = 1  # round batch widths up to this (use e.g. 128 on TPUs)


def train_sft(
    model: nn.Module,
    train_examples: Sequence[Encoded],
    val_examples: Sequence[Encoded],
    pad_id: int,
    config: SFTConfig,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Fine-tune model; returns the evaluation history. Every eval_interval
    steps (and at the end) the validation loss is measured and, with
    checkpoint_path, the full training state saved; resume=True continues
    from it as if never interrupted."""
    torch.manual_seed(config.seed)
    model.to(device).train()
    precision = resolve_precision(device, config.precision)
    optimizer = make_optimizer(model, config.peak_lr, config.weight_decay, config.betas)
    scheduler = d2z_scheduler(optimizer, config.total_steps, config.warmup_steps)
    scaler = precision.make_grad_scaler()
    batches = SFTBatches(train_examples, config.batch_size, pad_id, config.seed, config.pad_multiple)
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
    log(f"SFT: {len(train_examples):,} train / {len(val_examples):,} val examples, "
        f"{config.total_steps} steps x {config.batch_size}, precision {precision.name}")
    start, train_loss = time.perf_counter(), float("nan")
    while step < config.total_steps:
        idx, tgt = batches.next()
        train_loss = train_step(model, idx.to(device), tgt.to(device), optimizer, precision, scaler,
                                scheduler, config.max_grad_norm, config.grad_accum_steps)
        step += 1
        if step % config.eval_interval == 0 or step == config.total_steps:
            val_loss = evaluate_sft(model, val_examples, pad_id, config.batch_size, precision, device,
                                    config.pad_multiple)
            history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler,
                                scaler=scaler, batches=batches, step=step, history=history,
                                sft_config=vars(config))
            log(f"step {step:>6}/{config.total_steps} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} (ppl {math.exp(val_loss):.2f}) | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | {time.perf_counter() - start:.1f}s")
    return history
