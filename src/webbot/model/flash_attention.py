"""FlashAttention (Dao et al., 2022; the FlashAttention-2 loop order) in
plain PyTorch.

Attention is computed tile by tile: for a block of queries, blocks of keys
are visited in turn and the softmax is accumulated online (a running max m,
a running denominator l, and an output accumulator rescaled whenever m
grows). The (seq x seq) score matrix never exists in memory; only the
output and the per-query log-sum-exp are kept for backward, which recomputes
each tile's probabilities from them.

On a GPU the point is fewer slow-memory round trips. Here (CPU, PyTorch
ops per tile) the gain is memory: activation memory linear in sequence
length instead of quadratic. It is not expected to beat a fused kernel in
speed.

Leading (batch/head) dims broadcast, so grouped-query attention can pass
k and v with a size-1 group dim instead of copies per query head; the
gradients of broadcast inputs are summed back to their shapes.
"""

from __future__ import annotations

import math

import torch


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """At least float32 (bf16/fp16 inputs are accumulated in float32)."""
    return torch.promote_types(dtype, torch.float32)


def _tile_mask(
    q_start: int, q_end: int, k_start: int, k_end: int, offset: int, device
) -> torch.Tensor | None:
    """Causal mask (True = masked) for a tile, or None if no entry in it is
    masked. Query i sits at absolute position offset + i; key j at j."""
    if k_end - 1 <= offset + q_start:
        return None
    q_pos = torch.arange(offset + q_start, offset + q_end, device=device)
    k_pos = torch.arange(k_start, k_end, device=device)
    return k_pos[None, :] > q_pos[:, None]


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool, scale: float, block_q: int, block_k: int):
        # Autocast would run the tile matmuls in bf16; accumulate in the
        # compute dtype instead.
        with torch.autocast(q.device.type, enabled=False):
            return _FlashAttention._forward(ctx, q, k, v, causal, scale, block_q, block_k)

    @staticmethod
    def _forward(ctx, q, k, v, causal, scale, block_q, block_k):
        cdt = _compute_dtype(q.dtype)
        qc, kc, vc = q.to(cdt), k.to(cdt), v.to(cdt)
        len_q, len_k = q.shape[-2], k.shape[-2]
        offset = len_k - len_q
        batch = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
        out = torch.empty(*batch, len_q, v.shape[-1], dtype=cdt, device=q.device)
        lse = torch.empty(*batch, len_q, dtype=cdt, device=q.device)

        for qs in range(0, len_q, block_q):
            qe = min(qs + block_q, len_q)
            qb = qc[..., qs:qe, :]
            m = torch.full((*batch, qe - qs), -math.inf, dtype=cdt, device=q.device)
            l = torch.zeros_like(m)
            acc = torch.zeros(*batch, qe - qs, v.shape[-1], dtype=cdt, device=q.device)
            # Keys after the last query's position are never visible.
            k_stop = min(len_k, offset + qe) if causal else len_k
            for ks in range(0, k_stop, block_k):
                ke = min(ks + block_k, k_stop)
                s = (qb @ kc[..., ks:ke, :].transpose(-2, -1)) * scale
                if causal:
                    mask = _tile_mask(qs, qe, ks, ke, offset, q.device)
                    if mask is not None:
                        s = s.masked_fill(mask, -math.inf)
                # m_new is finite: every query sees key 0, which is in the
                # first key block (m = -inf before it gives rescale = 0).
                m_new = torch.maximum(m, s.amax(dim=-1))
                p = torch.exp(s - m_new[..., None])
                rescale = torch.exp(m - m_new)
                l = l * rescale + p.sum(dim=-1)
                acc = acc * rescale[..., None] + p @ vc[..., ks:ke, :]
                m = m_new
            out[..., qs:qe, :] = acc / l[..., None]
            lse[..., qs:qe] = m + torch.log(l)

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.scale = causal, scale
        ctx.block_q, ctx.block_k = block_q, block_k
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        with torch.autocast(grad_out.device.type, enabled=False):
            return _FlashAttention._backward(ctx, grad_out)

    @staticmethod
    def _backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        causal, scale = ctx.causal, ctx.scale
        block_q, block_k = ctx.block_q, ctx.block_k
        cdt = out.dtype
        qc, kc, vc, do = q.to(cdt), k.to(cdt), v.to(cdt), grad_out.to(cdt)
        len_q, len_k = q.shape[-2], k.shape[-2]
        offset = len_k - len_q

        # D_i = sum_j P_ij dP_ij = rowsum(dO * O)
        delta = (do * out).sum(dim=-1)
        batch = out.shape[:-2]
        # dq is accumulated at the broadcast shape and summed down at the
        # end; dk/dv per key block.
        dq = torch.zeros(*batch, len_q, q.shape[-1], dtype=cdt, device=q.device)
        dk = torch.zeros_like(kc)
        dv = torch.zeros_like(vc)

        for ks in range(0, len_k, block_k):
            ke = min(ks + block_k, len_k)
            kb, vb = kc[..., ks:ke, :], vc[..., ks:ke, :]
            dk_b = torch.zeros(*batch, ke - ks, k.shape[-1], dtype=cdt, device=q.device)
            dv_b = torch.zeros(*batch, ke - ks, v.shape[-1], dtype=cdt, device=q.device)
            # Only queries at positions >= ks can see this key block.
            q_first = max(0, ks - offset) if causal else 0
            for qs in range((q_first // block_q) * block_q, len_q, block_q):
                qe = min(qs + block_q, len_q)
                qb, dob = qc[..., qs:qe, :], do[..., qs:qe, :]
                s = (qb @ kb.transpose(-2, -1)) * scale
                if causal:
                    mask = _tile_mask(qs, qe, ks, ke, offset, q.device)
                    if mask is not None:
                        s = s.masked_fill(mask, -math.inf)
                p = torch.exp(s - lse[..., qs:qe, None])
                dv_b += p.transpose(-2, -1) @ dob
                dp = dob @ vb.transpose(-2, -1)
                ds = p * (dp - delta[..., qs:qe, None])
                dq[..., qs:qe, :] += (ds @ kb) * scale
                dk_b += (ds.transpose(-2, -1) @ qb) * scale
            dk[..., ks:ke, :] = dk_b.sum_to_size(kb.shape)
            dv[..., ks:ke, :] = dv_b.sum_to_size(vb.shape)

        dq = dq.sum_to_size(qc.shape)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """softmax(q k^T / sqrt(d)) v over the last two dims, tiled.

    q is (..., len_q, d), k and v are (..., len_k, d), with leading dims
    that broadcast against each other. With causal=True,
    query i is at position len_k - len_q + i (the queries are the last
    len_q positions, as with a KV cache) and sees keys at positions <= it.
    """
    if causal and q.shape[-2] > k.shape[-2]:
        raise ValueError("causal attention needs len_q <= len_k")
    scale = 1.0 / math.sqrt(q.shape[-1])
    return _FlashAttention.apply(q, k, v, causal, scale, block_q, block_k)
