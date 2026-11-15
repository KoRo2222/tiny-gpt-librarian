from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich 2019).

        out = x / sqrt(mean(x^2) + eps) * weight

    Unlike LayerNorm it neither subtracts the mean nor adds a bias: it only
    rescales each vector to unit RMS, then applies a learned per-dimension
    gain. Cheaper, and in practice just as stable for Transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the statistic in float32 even for half-precision inputs:
        # squaring small values in fp16/bf16 loses precision.
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight.to(x.dtype)
