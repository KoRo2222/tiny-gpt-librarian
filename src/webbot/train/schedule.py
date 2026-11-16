from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def d2z_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate multiplier (relative to peak) for the update at `step`.

    D2Z (linear decay-to-zero; Bergsma et al. 2025, "Straight to Zero"):
    linear warmup to the peak over warmup_steps, then linear decay that
    reaches exactly 0 at total_steps. Unlike cosine-to-10%, nothing is left
    at the end, which averages out more of the gradient noise from the late,
    small updates.

    Steps are 0-indexed: updates 0 .. warmup_steps-1 ramp up as
    1/W, 2/W, ..., 1; update total_steps-1 is the last, smallest non-zero
    one, and anything at or past total_steps gets 0.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"warmup_steps must be in [0, total_steps), got {warmup_steps}"
        )
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))


def d2z_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> LambdaLR:
    """Scale each param group's lr (its peak) by d2z_multiplier.

    Call scheduler.step() once after every optimizer.step().
    """
    # Validate eagerly rather than on the first step.
    d2z_multiplier(0, total_steps, warmup_steps)
    return LambdaLR(
        optimizer, lambda step: d2z_multiplier(step, total_steps, warmup_steps)
    )
