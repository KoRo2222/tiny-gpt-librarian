from __future__ import annotations

import torch
from torch import nn

from .swiglu import SwiGLU


class MoE(nn.Module):
    """Sparse mixture-of-experts feed-forward layer (Shazeer et al. 2017;
    top-k routing as in Switch Transformer / Mixtral).

    n_experts SwiGLU experts replace the single FFN. A linear router scores
    every expert for each token; the token goes only through its top_k
    experts, and their outputs are summed, weighted by gates:

        top_k == 1: the router probability itself (Switch), so the router
                    also learns from the main loss
        top_k >= 2: the top-k probabilities renormalized to sum to 1
                    (Mixtral; equal to a softmax over the top-k logits)

    Compute per token is that of top_k experts while parameters grow with
    n_experts. There is no capacity limit: no token is dropped.

    After each forward, self.aux_loss holds the Switch Transformer
    load-balancing loss  n_experts * sum_i f_i * P_i,  where f_i is the
    fraction of routing slots (tokens * top_k) given to expert i and P_i the
    mean router probability of expert i. It is 1 when routing is uniform and
    grows as tokens pile onto few experts; its gradient reaches the router
    through P (f is a count).
    """

    def __init__(
        self, d_model: int, n_experts: int, top_k: int = 2, hidden_dim: int | None = None
    ):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k {top_k} must be in 1..n_experts ({n_experts})")
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, hidden_dim) for _ in range(n_experts))
        self.aux_loss = torch.zeros(())
        self.last_counts = torch.zeros(n_experts, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])  # (tokens, d_model)
        n_tokens = x.shape[0]

        # Routing in at least float32 (softmax of bf16 logits is coarse).
        logits = self.router(x)
        probs = logits.to(torch.promote_types(logits.dtype, torch.float32)).softmax(dim=-1)
        top_p, top_e = probs.topk(self.top_k, dim=-1)  # (tokens, top_k)
        gates = top_p if self.top_k == 1 else top_p / top_p.sum(dim=-1, keepdim=True)

        # Group the (token, slot) assignments by expert: one batched call
        # per expert instead of one per token.
        flat_e = top_e.flatten()
        order = flat_e.argsort(stable=True)
        token_of = order // self.top_k
        gate_of = gates.flatten()[order]
        counts = torch.bincount(flat_e, minlength=self.n_experts)

        out = torch.zeros_like(x)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            idx = token_of[start : start + n]
            y = self.experts[e](x[idx]) * gate_of[start : start + n, None]
            out.index_add_(0, idx, y.to(out.dtype))
            start += n

        frac = counts.to(probs.dtype) / (n_tokens * self.top_k)
        self.aux_loss = self.n_experts * (frac * probs.mean(dim=0)).sum()
        self.last_counts = counts.detach()
        return out.view(shape)
