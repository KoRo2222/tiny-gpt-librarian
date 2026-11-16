"""Data-parallel training across processes (the DDP scheme, written out).

Every process (rank) holds a full replica of the model. Each optimizer
update, all ranks draw the same global batch and each takes its own share
of the rows; after backward, the gradients are summed across ranks (with
each rank's weighted by its share of the rows), so every rank holds the
gradient of the whole batch's mean loss and takes the identical optimizer
step. The replicas never diverge, and the result equals single-process
training on the whole batch.

Only collective primitives (broadcast, all_reduce) come from
torch.distributed; the "gloo" backend runs them between CPU processes.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

BUCKET_BYTES = 25 << 20  # gradients are all-reduced in flat buckets of ~25 MB


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def local_rows(batch: int, r: int | None = None, world: int | None = None) -> tuple[int, int]:
    """(start, stop) of rank r's rows (default: this process) in a global
    batch of `batch` rows: contiguous shares whose sizes differ by at most
    one."""
    r = rank() if r is None else r
    world = world_size() if world is None else world
    if batch < world:
        raise ValueError(f"batch of {batch} rows cannot be split over {world} ranks")
    base, extra = divmod(batch, world)
    start = r * base + min(r, extra)
    return start, start + base + (r < extra)


def _buckets(tensors: list[torch.Tensor]) -> list[list[torch.Tensor]]:
    out, cur, size = [], [], 0
    for t in tensors:
        if cur and size + t.numel() * t.element_size() > BUCKET_BYTES:
            out.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += t.numel() * t.element_size()
    if cur:
        out.append(cur)
    return out


def broadcast_parameters(model: nn.Module, src: int = 0) -> None:
    """Copy rank src's parameters to every rank (identical start)."""
    if not is_distributed():
        return
    with torch.no_grad():
        for bucket in _buckets([p.data for p in model.parameters()]):
            flat = torch.cat([t.flatten() for t in bucket])
            dist.broadcast(flat, src)
            for t, chunk in zip(bucket, flat.split([t.numel() for t in bucket])):
                t.copy_(chunk.view_as(t))


def all_reduce_gradients(model: nn.Module, weight: float) -> None:
    """Replace every gradient by sum over ranks of (weight_r * grad_r).

    weight is this rank's share of the batch rows, so the result is the
    gradient of the global mean loss. A parameter that got no gradient on
    this rank (an MoE expert no local token was routed to) counts as zero,
    so every rank reduces the same buckets in the same order; one that got
    none on any rank is left at None, as in single-process training (the
    optimizer then skips it rather than decaying it).
    """
    if not is_distributed():
        return
    params = [p for p in model.parameters() if p.requires_grad]
    has_grad = torch.tensor([p.grad is not None for p in params], dtype=torch.float32)
    dist.all_reduce(has_grad)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    with torch.no_grad():
        for bucket in _buckets([p.grad for p in params]):
            flat = torch.cat([g.flatten() for g in bucket]) * weight
            dist.all_reduce(flat)
            for g, chunk in zip(bucket, flat.split([g.numel() for g in bucket])):
                g.copy_(chunk.view_as(g))
    for p, any_grad in zip(params, has_grad.tolist()):
        if not any_grad:
            p.grad = None


def all_reduce_sum(*values: float) -> list[float]:
    """Sum Python floats over ranks (as float64)."""
    t = torch.tensor(values, dtype=torch.float64)
    if is_distributed():
        dist.all_reduce(t)
    return t.tolist()


def _worker(
    r: int, world: int, init_file: str, threads: int | None, fn: Callable, args: tuple
) -> None:
    if threads is not None:
        torch.set_num_threads(threads)
    dist.init_process_group(
        "gloo", init_method=Path(init_file).as_uri(), rank=r, world_size=world
    )
    try:
        fn(*args)
    finally:
        dist.destroy_process_group()


def launch(
    fn: Callable, world: int, *args, threads_per_rank: int | None = None
) -> None:
    """Run fn(*args) in `world` new processes joined into one gloo process
    group (rank() / world_size() tell them apart). fn and args must be
    picklable (a module-level function). Blocks until all ranks finish; an
    exception in any rank is raised here.

    threads_per_rank sets each process's intra-op thread count (default:
    PyTorch's, which would oversubscribe the cores with several ranks).
    """
    fd, init_file = tempfile.mkstemp(prefix="webbot_dist_")
    os.close(fd)
    os.remove(init_file)  # the file store creates it
    try:
        mp.spawn(_worker, args=(world, init_file, threads_per_rank, fn, args), nprocs=world)
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)
