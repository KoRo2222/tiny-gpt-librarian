"""Gradient (activation) checkpointing (Chen et al. 2016, "Training Deep
Nets with Sublinear Memory Cost").

checkpoint(fn, *inputs) runs fn without recording its graph, so none of
fn's intermediate activations are kept; only the inputs are saved. In
backward, fn is run again on the saved inputs with the graph recorded, and
the incoming gradients are backpropagated through that recomputation. The
cost is one extra forward pass of fn per backward.

Gradients of parameters used inside fn are accumulated into their .grad
during that inner backward (the "reentrant" scheme), so this works with
loss.backward() but not with torch.autograd.grad(loss, params).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.utils.checkpoint


class _Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *inputs):
        ctx.fn = fn
        device_type = inputs[0].device.type
        # Backward runs outside any autocast region: remember this one so the
        # recomputation uses the same precision as the original forward.
        ctx.device_type = device_type
        ctx.autocast_enabled = torch.is_autocast_enabled(device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(device_type)
        ctx.save_for_backward(*inputs)
        # Autograd already disables grad recording inside Function.forward.
        outputs = fn(*inputs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = [x.detach().requires_grad_(x.requires_grad) for x in ctx.saved_tensors]
        with torch.enable_grad(), torch.autocast(
            ctx.device_type, dtype=ctx.autocast_dtype, enabled=ctx.autocast_enabled
        ):
            outputs = ctx.fn(*inputs)
        pairs = [
            (out, grad)
            for out, grad in zip(outputs, grad_outputs)
            if out.requires_grad and grad is not None
        ]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        return (None, *(x.grad if x.requires_grad else None for x in inputs))


def checkpoint(
    fn: Callable[..., tuple[torch.Tensor, ...]], *inputs: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """fn(*inputs) (a tuple of tensors), keeping only inputs for backward
    and recomputing fn there. fn must be deterministic (same outputs when
    run again), which holds for this model (no dropout).

    Under torch.compile, Dynamo cannot trace the backward-inside-backward
    of this implementation (each call would break the graph), so there
    PyTorch's own non-reentrant torch.utils.checkpoint is used: Dynamo
    captures it as a checkpointed region of the graph with the same effect.
    """
    if torch.compiler.is_compiling():
        return torch.utils.checkpoint.checkpoint(fn, *inputs, use_reentrant=False)
    return _Checkpoint.apply(fn, *inputs)
