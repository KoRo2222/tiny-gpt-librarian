from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .device import xla_step
from .precision import Precision


def train_step(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    precision: Precision,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = None,
    grad_accum_steps: int = 1,
    grad_sync: Callable[[], None] | None = None,
) -> float:
    """One optimizer update with mixed precision and gradient accumulation;
    returns the loss on the whole batch.

    With grad_accum_steps = k the batch is split along its first dim into k
    micro-batches (sizes differing by at most one), each forwarded and
    backpropagated in turn, so activations are held for one micro-batch at
    a time. Each micro-batch loss (a mean over its rows) is weighted by its
    share of the rows, so the accumulated gradient is the gradient of the
    whole batch's mean loss, as without accumulation. Clipping and the
    optimizer step happen once, after the last micro-batch.

    Forward (and therefore backward) runs under autocast; the loss is
    scaled for fp16 (scaler is a no-op otherwise) and gradients are
    unscaled before clipping, so max_grad_norm applies to the true
    gradients. If fp16 gradients overflowed, the scaler skips the update.

    grad_sync, if given, is called once the batch's gradients are complete
    and before clipping (data-parallel training averages them across
    processes there).
    """
    batch = idx.shape[0]
    if not 1 <= grad_accum_steps <= batch:
        raise ValueError(f"grad_accum_steps must be in 1..batch size ({batch})")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # The loss is read once, after the update: on XLA devices reading a value
    # forces execution, so the whole step stays one compiled graph.
    total = None
    for idx_mb, targets_mb in zip(
        idx.tensor_split(grad_accum_steps), targets.tensor_split(grad_accum_steps)
    ):
        weight = idx_mb.shape[0] / batch
        with precision.autocast():
            _, loss = model(idx_mb, targets_mb)
        scaler.scale(loss * weight).backward()
        part = loss.detach() * weight  # the loss is already >= float32
        total = part if total is None else total + part
    if grad_sync is not None:
        grad_sync()
    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        # A sharded optimizer (ZeRO) holds the gradients and clips them itself.
        if hasattr(optimizer, "clip_grad_norm_"):
            optimizer.clip_grad_norm_(max_grad_norm)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()
    xla_step(idx.device)
    return total.item()
