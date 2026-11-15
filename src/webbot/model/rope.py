from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE, Su et al. 2021).

    Instead of adding a position vector to the token embedding, each query /
    key vector is rotated by an angle proportional to its position. Dims are
    grouped into head_dim/2 pairs, and pair i at position m is rotated by
    m * base^(-2i/head_dim). Since rotations compose, the dot product
    <R(m)q, R(n)k> depends only on the offset m - n, which is what lets
    attention see relative position.

    Pairs follow the "rotate half" layout: dim i is paired with dim
    i + head_dim/2 (equivalent to pairing adjacent dims up to a fixed
    permutation of the weights).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), inv_freq)
        # Tables are derived from the config, so they are not saved in the
        # state dict (persistent=False).
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate x of shape (..., seq_len, head_dim).

        offset is the position of x's first token, so that during generation
        the newest token can be rotated alone at its true position.
        """
        seq_len = x.shape[-2]
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"positions up to {offset + seq_len} exceed max_seq_len "
                f"{self.max_seq_len}"
            )
        cos = self.cos[offset : offset + seq_len]
        sin = self.sin[offset : offset + seq_len]
        # Rotate in float32 even for half-precision inputs: bf16 cos/sin are
        # only accurate to ~3 digits, and the error would differ per position.
        x1, x2 = x.float().chunk(2, dim=-1)
        # 2D rotation of each (x1[i], x2[i]) pair.
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.to(x.dtype)
