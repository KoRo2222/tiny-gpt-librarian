from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def default_hidden_dim(d_model: int, multiple_of: int = 64) -> int:
    """Hidden size that keeps SwiGLU's parameter count close to a standard
    4*d_model GELU FFN: SwiGLU has three weight matrices instead of two, so
    the hidden size shrinks to 2/3 * 4 * d_model, rounded up to multiple_of
    for efficient matmuls."""
    hidden = int(2 * 4 * d_model / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """Position-wise feed-forward block with a SwiGLU activation
    (Shazeer 2020, "GLU Variants Improve Transformer"; as used in LLaMA).

        out = W_down( SiLU(W_gate x) * (W_up x) )

    W_gate's branch passes through SiLU (x * sigmoid(x)) and multiplies the
    linear W_up branch element-wise, so the network learns per-dimension how
    much of each hidden feature to let through. No biases, as in LLaMA.
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or default_hidden_dim(d_model)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
