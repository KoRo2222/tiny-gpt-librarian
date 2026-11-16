from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch

_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass(frozen=True)
class Precision:
    """Mixed-precision setting for training.

    Parameters and optimizer state always stay float32 ("master weights");
    only the forward/backward math runs in compute_dtype via autocast.
    fp16 additionally needs loss scaling, since its narrow exponent range
    flushes small gradients to zero; bf16 has float32's range and doesn't.
    """

    device_type: str
    compute_dtype: torch.dtype | None  # None = plain float32

    @property
    def name(self) -> str:
        return next(k for k, v in _DTYPES.items() if v == self.compute_dtype)

    def autocast(self) -> AbstractContextManager:
        return torch.autocast(
            self.device_type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype is not None,
        )

    def make_grad_scaler(self) -> torch.amp.GradScaler:
        # A disabled scaler is a no-op, so callers can use it unconditionally.
        return torch.amp.GradScaler(
            self.device_type, enabled=self.compute_dtype == torch.float16
        )


def resolve_precision(device: str | torch.device, name: str = "auto") -> Precision:
    """Pick a Precision for device. name is "auto", "fp32", "bf16" or "fp16".

    auto: bf16 on GPUs that support it, else fp16 on GPU; fp32 on CPU,
    where bf16 is only faster with native bf16 instructions (AMX /
    AVX512-BF16) and otherwise just costs accuracy.
    """
    device_type = torch.device(device).type
    if name == "auto":
        if device_type == "cuda":
            name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            name = "fp32"
    if name not in _DTYPES:
        raise ValueError(f"unknown precision {name!r}; expected auto/{'/'.join(_DTYPES)}")
    return Precision(device_type, _DTYPES[name])
