"""ZeRO (Rajbhandari et al. 2020), stages 1 and 2, around AdamW.

Plain data parallelism keeps a full copy of everything on every rank. ZeRO
splits the redundant training state instead. All parameters are laid out
as one flat vector (the weight-decayed matrices first, then the rest),
padded and cut into world_size equal, contiguous shards; rank r owns shard
r:

  stage 1: the optimizer state (AdamW's two moments, twice the parameter
           count) exists only for the owned shard. Gradients are all-reduced
           as in plain data parallelism; each rank updates its shard and the
           updated shards are all-gathered back into every replica.
  stage 2: gradients are also sharded: a reduce-scatter (half the traffic
           of an all-reduce) leaves each rank only the summed gradient of
           its own shard, and the full local gradients are released.

AdamW is elementwise, so updating a flat shard equals updating the
parameters one by one. Each shard is held as at most two flat tensors (its
part of the decayed and of the non-decayed parameters), so weight decay
still applies to the matrices only.

Unlike per-parameter AdamW, a parameter that gets no gradient anywhere (an
MoE expert no token was routed to) is updated with a zero gradient (its
moments decay, and weight decay applies) instead of being skipped.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from .distributed import is_distributed, rank, world_size


class ZeroOptimizer:
    """Sharded AdamW (ZeRO stage 1 or 2) for a model replicated on every
    rank. Use like an optimizer: zero_grad(), backward, reduce_gradients(
    share), optionally clip_grad_norm_(), step(). Works without a process
    group too (a single shard)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        stage: int = 1,
    ):
        if stage not in (1, 2):
            raise ValueError(f"ZeRO stage must be 1 or 2, got {stage}")
        self.stage = stage
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        self.params = decay + no_decay
        self.numels = [p.numel() for p in self.params]
        self.total = sum(self.numels)
        n_decay = sum(p.numel() for p in decay)
        self.world, self.rank = world_size(), rank()
        self.shard_size = math.ceil(self.total / self.world)
        self.lo = min(self.rank * self.shard_size, self.total)
        self.hi = min(self.lo + self.shard_size, self.total)

        flat = self._flat_params()
        # This rank's piece of each param group, as [start, stop) in the flat vector.
        self.pieces = [
            (self.lo, max(self.lo, min(self.hi, n_decay))),
            (max(self.lo, min(self.hi, n_decay)), self.hi),
        ]
        self.shard_params = [nn.Parameter(flat[a:b].clone()) for a, b in self.pieces]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in self.shard_params[:1] if p.numel()], "weight_decay": weight_decay},
                {"params": [p for p in self.shard_params[1:] if p.numel()], "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )

    # --- flat views -------------------------------------------------------

    def _padded(self, flat: torch.Tensor) -> torch.Tensor:
        pad = self.shard_size * self.world - flat.numel()
        return torch.cat((flat, flat.new_zeros(pad))) if pad else flat

    def _flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self.params])

    def _flat_grads(self) -> torch.Tensor:
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in self.params
        ])

    @property
    def param_groups(self) -> list[dict]:
        return self.optimizer.param_groups

    # --- the training step ------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params + self.shard_params:
            p.grad = None

    @torch.no_grad()
    def reduce_gradients(self, share: float = 1.0) -> None:
        """Sum (share * local gradient) over ranks into this rank's shard.

        share is this rank's fraction of the batch rows, as in plain data
        parallelism. Stage 1 all-reduces the full gradient (and leaves it
        in the parameters' .grad); stage 2 reduce-scatters it and releases
        the full local gradients.
        """
        flat = self._padded(self._flat_grads() * share)
        if self.world == 1:
            shard = flat
        elif self.stage == 1:
            dist.all_reduce(flat)
            shard = flat[self.rank * self.shard_size : (self.rank + 1) * self.shard_size]
            for p, g in zip(self.params, flat[: self.total].split(self.numels)):
                p.grad = g.view_as(p).clone()
        else:
            shard = flat.new_empty(self.shard_size)
            dist.reduce_scatter_tensor(shard, flat)
            for p in self.params:
                p.grad = None
        for sp, (a, b) in zip(self.shard_params, self.pieces):
            sp.grad = shard[a - self.lo : b - self.lo].clone()

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """Clip the (sharded) gradient by its global norm, as
        nn.utils.clip_grad_norm_ would clip the full gradient."""
        sq = sum((sp.grad.double().pow(2).sum() for sp in self.shard_params), torch.zeros((), dtype=torch.float64))
        if self.world > 1:
            dist.all_reduce(sq)
        norm = sq.sqrt()
        coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for sp in self.shard_params:
            sp.grad.mul_(coef.to(sp.grad.dtype))
        return norm.float()

    @torch.no_grad()
    def step(self) -> None:
        """Update this rank's shard, then give every replica all shards.

        The shard is first refreshed from the model, which stays the source
        of truth: weights broadcast or loaded after this optimizer was
        built are picked up.
        """
        flat = self._flat_params()
        for sp, (a, b) in zip(self.shard_params, self.pieces):
            sp.copy_(flat[a:b])
        self.optimizer.step()
        shard = torch.cat([sp.detach() for sp in self.shard_params])
        shard = torch.cat((shard, shard.new_zeros(self.shard_size - shard.numel())))
        if self.world > 1:
            full = shard.new_empty(self.shard_size * self.world)
            dist.all_gather_into_tensor(full, shard)
        else:
            full = shard
        for p, v in zip(self.params, full[: self.total].split(self.numels)):
            p.copy_(v.view_as(p))

    # --- checkpoints --------------------------------------------------------

    def _gather_state(self, key: str) -> torch.Tensor:
        parts = []
        for sp in self.shard_params:
            st = self.optimizer.state.get(sp, {})
            parts.append(st[key] if key in st else torch.zeros_like(sp))
        shard = torch.cat(parts) if parts else torch.zeros(0)
        shard = torch.cat((shard, shard.new_zeros(self.shard_size - shard.numel())))
        if self.world > 1:
            full = shard.new_empty(self.shard_size * self.world)
            dist.all_gather_into_tensor(full, shard)
            shard = full
        return shard[: self.total].clone()

    def state_dict(self) -> dict:
        """The full (all shards) optimizer state, the same on every rank and
        independent of the number of ranks. A collective: every rank must
        call it."""
        steps = [
            float(self.optimizer.state[sp]["step"]) if sp in self.optimizer.state else 0.0
            for sp in self.shard_params
            if sp.numel()
        ]
        step = torch.tensor(max(steps, default=0.0), dtype=torch.float64)
        if self.world > 1:
            dist.all_reduce(step, op=dist.ReduceOp.MAX)
        return {
            "zero": True,
            "step": step.item(),
            "exp_avg": self._gather_state("exp_avg"),
            "exp_avg_sq": self._gather_state("exp_avg_sq"),
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"} for g in self.optimizer.param_groups
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from state_dict() saved with any number of ranks."""
        for group, saved in zip(self.optimizer.param_groups, state["param_groups"]):
            group.update(saved)
        for sp, (a, b) in zip(self.shard_params, self.pieces):
            if not sp.numel():
                continue
            self.optimizer.state[sp] = {
                "step": torch.tensor(state["step"]),
                "exp_avg": state["exp_avg"][a:b].clone(),
                "exp_avg_sq": state["exp_avg_sq"][a:b].clone(),
            }

    def state_numel(self) -> int:
        """Elements of optimizer state held by this rank."""
        return sum(
            t.numel() for st in self.optimizer.state.values() for t in st.values() if t.dim()
        )
