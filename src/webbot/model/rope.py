from __future__ import annotations

import math

import torch
from torch import nn


ROPE_SCALINGS = (None, "pi", "ntk", "yarn")


def rope_frequencies(
    head_dim: int,
    base: float = 10000.0,
    scaling: str | None = None,
    factor: float = 1.0,
    original_max_seq_len: int | None = None,
    yarn_alpha: float = 1.0,
    yarn_beta: float = 32.0,
) -> tuple[torch.Tensor, float]:
    """Per-pair angular frequencies theta_i (float64) and the attention
    scale for RoPE, optionally stretched to a context factor times longer
    than the one trained on:

    None:  theta_i = base^(-2i/d)
    "pi":  Position Interpolation (Chen et al. 2023): theta_i / factor, i.e.
           position m is read as m / factor, inside the trained range.
    "ntk": NTK-aware scaling: base * factor^(d/(d-2)). The fastest pair
           is unchanged, the slowest is divided by exactly factor, and the
           ones between are stretched progressively.
    "yarn": YaRN (Peng et al. 2023), "NTK-by-parts" plus a temperature. A
           pair's wavelength 2*pi/theta_i fits r_i = L * theta_i / (2*pi)
           times into the trained length L. Pairs with r_i > beta (many
           turns: local, high-frequency) are kept; pairs with r_i < alpha
           (less than one turn) are interpolated as in PI; between, the two
           are blended linearly in r. q and k are both multiplied by
           mscale = 0.1 ln(factor) + 1, sharpening the softmax that a longer
           context flattens.
    """
    if scaling not in ROPE_SCALINGS:
        raise ValueError(f"unknown RoPE scaling {scaling!r}")
    if scaling is not None and factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
    if scaling == "ntk":
        base = base * factor ** (head_dim / (head_dim - 2))
    inv_freq = base ** (-exponents)
    mscale = 1.0
    if scaling == "pi":
        inv_freq = inv_freq / factor
    elif scaling == "yarn":
        if original_max_seq_len is None:
            raise ValueError("yarn needs original_max_seq_len")
        turns = original_max_seq_len * inv_freq / (2 * math.pi)
        keep = ((turns - yarn_alpha) / (yarn_beta - yarn_alpha)).clamp(0.0, 1.0)
        inv_freq = (1 - keep) * inv_freq / factor + keep * inv_freq
        mscale = 0.1 * math.log(factor) + 1.0
    return inv_freq, mscale


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

    scaling ("pi", "ntk", "yarn"; see rope_frequencies) stretches the
    frequencies so a model trained on original_max_seq_len tokens can
    attend over factor times as many.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        scaling: str | None = None,
        factor: float = 1.0,
        original_max_seq_len: int | None = None,
        yarn_alpha: float = 1.0,
        yarn_beta: float = 32.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.inv_freq, self.mscale = rope_frequencies(
            head_dim, base, scaling, factor, original_max_seq_len, yarn_alpha, yarn_beta
        )
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), self.inv_freq)
        # Tables are derived from the config, so they are not saved in the
        # state dict (persistent=False). YaRN's mscale multiplies both q and
        # k, so attention logits are scaled by mscale^2.
        self.register_buffer("cos", (angles.cos() * self.mscale).float(), persistent=False)
        self.register_buffer("sin", (angles.sin() * self.mscale).float(), persistent=False)

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
        # (float64 input stays float64.)
        x1, x2 = x.to(torch.promote_types(x.dtype, torch.float32)).chunk(2, dim=-1)
        # 2D rotation of each (x1[i], x2[i]) pair.
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.to(x.dtype)
