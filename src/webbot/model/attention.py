from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .flash_attention import flash_attention
from .rope import RotaryEmbedding

# Cached keys and values of one layer, each (batch, n_kv_heads, past_len, head_dim).
KVCache = tuple[torch.Tensor, torch.Tensor]


def attention_weights(scores: torch.Tensor) -> torch.Tensor:
    """Softmax over keys, computed in (at least) float32 even under bf16/fp16 autocast
    (CPU autocast would otherwise keep it in bf16, where exp() of
    low-precision scores loses most of the small attention weights)."""
    return scores.to(torch.promote_types(scores.dtype, torch.float32)).softmax(dim=-1)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with rotary position embedding and grouped
    key/value heads.

    Queries and keys are rotated by RoPE before the dot product, so scores
    depend on relative position and no learned position table is needed.
    An optional KV cache lets generation process only the newest token(s):
    the new tokens' positions start right after the cached ones.

    attn_impl "naive" builds the full masked score matrix; "flash" computes
    the same attention tile by tile (flash_attention), keeping memory linear
    in sequence length; "sdpa" calls PyTorch's fused
    scaled_dot_product_attention (FlashAttention / memory-efficient kernels
    on a GPU; the fastest choice on this CPU too).

    Grouped-query attention (GQA): there are n_kv_heads key/value heads
    (n_heads by default, i.e. ordinary multi-head attention; 1 is
    multi-query attention). Query heads come in n_kv_heads groups of
    n_heads // n_kv_heads consecutive heads, and query head h uses key/value
    head h // group. The K/V projection and the KV cache shrink by the group
    size; K/V are broadcast over the group, never copied per query head.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryEmbedding,
        attn_impl: str = "naive",
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        if n_kv_heads < 1 or n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}")
        if attn_impl not in ("naive", "flash", "sdpa"):
            raise ValueError(f"unknown attn_impl {attn_impl!r}")
        self.attn_impl = attn_impl
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        kv_dim = n_kv_heads * self.head_dim
        self.split_sizes = (d_model, kv_dim, kv_dim)
        self.qkv = nn.Linear(d_model, d_model + 2 * kv_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope

    def forward(
        self, x: torch.Tensor, kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        batch, seq_len, d_model = x.shape
        q, k, v = (
            t.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
            for t in self.qkv(x).split(self.split_sizes, dim=-1)
        )  # q (batch, n_heads, seq_len, head_dim), k/v (batch, n_kv_heads, ...)

        past_len = 0 if kv_cache is None else kv_cache[0].shape[2]
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)

        if self.attn_impl == "sdpa":
            y = self._sdpa(q, k, v, past_len)
            y = y.transpose(1, 2).reshape(batch, seq_len, d_model)
            return self.out(y), (k, v)

        # Group the query heads: q (batch, n_kv, group, seq, hd) against
        # k/v (batch, n_kv, 1, past+seq, hd), broadcast over the group.
        qg = q.unflatten(1, (self.n_kv_heads, -1))
        kg, vg = k[:, :, None], v[:, :, None]
        if self.attn_impl == "flash":
            # Queries are the last seq_len positions, after the cache.
            y = flash_attention(qg, kg, vg, causal=True)
        else:
            scores = qg @ kg.transpose(-2, -1) / math.sqrt(self.head_dim)
            # Query i sits at absolute position past_len + i and may attend
            # to keys at positions <= that, i.e. never to the future.
            q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)
            k_pos = torch.arange(past_len + seq_len, device=x.device)
            scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            y = attention_weights(scores).to(v.dtype) @ vg

        y = y.flatten(1, 2).transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out(y), (k, v)

    def _sdpa(self, q, k, v, past_len: int) -> torch.Tensor:
        """F.scaled_dot_product_attention with the cache-aware causal mask.

        is_causal=True aligns the mask at the top left (query i sees keys
        0..i), right only without a cache; after a cache, query i sits at
        position past_len + i, so the mask is passed explicitly (a single
        new token sees every key and needs none). enable_gqa lets query
        head h use key/value head h // group without copying k and v.
        """
        seq_len = q.shape[2]
        gqa = self.n_kv_heads != self.n_heads
        if past_len == 0:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        mask = None
        if seq_len > 1:
            q_pos = torch.arange(past_len, past_len + seq_len, device=q.device)
            k_pos = torch.arange(past_len + seq_len, device=q.device)
            mask = k_pos[None, :] <= q_pos[:, None]  # True = may attend
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
