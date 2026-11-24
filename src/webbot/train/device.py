"""Choosing the device, including TPUs through PyTorch/XLA.

On XLA devices operations are recorded lazily and compiled into graphs;
xla_step() marks the end of a training step (the recorded graph is
compiled once and then replayed), every change of tensor shapes compiles
a new graph, and reading a value (.item()) forces execution. Elsewhere
these helpers do nothing, so the CPU/GPU paths are unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def xla_available() -> bool:
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        return False
    return True


def tpu_present() -> bool:
    """A TPU host, judged from the environment PyTorch/XLA itself reads."""
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    return any(k.startswith(("TPU_", "COLAB_TPU")) for k in os.environ)


def resolve_device(name: str = "auto") -> torch.device:
    """auto: CUDA if available, else a TPU (via torch_xla) if present, else CPU.
    "xla"/"tpu" ask for the XLA device explicitly."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if tpu_present() and xla_available():
            name = "xla"
        else:
            return torch.device("cpu")
    if name in ("xla", "tpu"):
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    return torch.device(name)


def is_xla(device: str | torch.device) -> bool:
    return torch.device(device).type == "xla"


def xla_step(device: str | torch.device) -> None:
    """End of a step on an XLA device (compile/run the recorded graph)."""
    if is_xla(device):
        import torch_xla.core.xla_model as xm

        xm.mark_step()


def to_cpu(obj: Any) -> Any:
    """obj with every tensor (in dicts, lists, tuples) moved to the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj
