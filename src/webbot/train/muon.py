"""Muon (Keller Jordan, 2024: "MomentUm Orthogonalized by Newton-Schulz").

For a hidden weight matrix, Muon takes the Nesterov momentum of the
gradient and replaces it by (approximately) the nearest semi-orthogonal
matrix, U V^T for the SVD U S V^T: every direction the gradient points in
gets the same step size, instead of a few dominant directions taking all
of it. The orthogonalization is not an SVD but five steps of a quintic
Newton-Schulz iteration, X <- aX + b(XX^T)X + c(XX^T)^2 X, whose
coefficients are tuned to push all singular values into roughly [0.7, 1.2]
quickly (exact convergence to 1 is not needed).

Muon is meant for the 2-D weights inside the Transformer blocks. The
embedding, the output head and the 1-D gains are updated with AdamW, in
the same optimizer (one object for the scheduler, checkpoints and
train_step), as in the reference implementation.
"""

from __future__ import annotations

import torch
from torch import nn

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a matrix: U f(S) V^T for g = U S V^T,
    with f(s) near 1 for every singular value s > 0.

    The iteration is an odd matrix polynomial, so it acts on the singular
    values alone: after scaling g to unit Frobenius norm (all s <= 1), each
    step maps s -> a s + b s^3 + c s^5. Computed in (at least) float32 and
    on the wide orientation (X X^T is the smaller Gram matrix).
    """
    a, b, c = NS_COEFFS
    x = g.to(torch.promote_types(g.dtype, torch.float32))
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.T if tall else x).to(g.dtype)


def muon_update(
    grad: torch.Tensor,
    buf: torch.Tensor,
    momentum: float,
    nesterov: bool = True,
    ns_steps: int = 5,
    split: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Update the momentum buffer in place and return the orthogonalized
    step direction for one matrix. split cuts the rows into blocks (a fused
    q/k/v projection) that are orthogonalized separately."""
    buf.lerp_(grad, 1 - momentum)  # buf = momentum * buf + (1 - momentum) * grad
    g = grad.lerp(buf, momentum) if nesterov else buf
    blocks = g.split(list(split)) if split is not None else (g,)
    out = []
    for blk in blocks:
        rows, cols = blk.shape
        out.append(newton_schulz(blk, ns_steps) * max(1.0, rows / cols) ** 0.5)
    return torch.cat(out)


class Muon(torch.optim.Optimizer):
    """Muon for groups with use_muon=True (2-D parameters), AdamW for the
    others. Muon groups take lr, momentum, nesterov, ns_steps,
    weight_decay and an optional split; AdamW groups take lr, betas, eps,
    weight_decay. Weight decay is decoupled in both (p *= 1 - lr * wd)."""

    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            if group.get("use_muon", False):
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("split", None)
                if any(p.dim() != 2 for p in group["params"]):
                    raise ValueError("Muon groups take 2-D parameters only")
            else:
                group["use_muon"] = False
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], group["momentum"],
                        group["nesterov"], group["ns_steps"], group["split"],
                    )
                    p.mul_(1 - lr * wd)
                    p.add_(update.to(p.dtype), alpha=-lr)
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    t = state["step"]
                    state["exp_avg"].lerp_(p.grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    m_hat = state["exp_avg"] / (1 - beta1**t)
                    v_hat = state["exp_avg_sq"] / (1 - beta2**t)
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-lr)
        return loss


def make_muon_optimizer(
    model: nn.Module,
    muon_lr: float,
    adam_lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
) -> Muon:
    """Muon for the 2-D weights inside model.blocks (fused q/k/v split into
    its three projections), AdamW for everything else: the embedding and
    output head with weight decay, the 1-D gains without."""
    qkv = {id(b.attn.qkv.weight): b.attn.split_sizes for b in model.blocks}
    in_blocks = {id(p) for p in model.blocks.parameters()}
    hidden = [p for p in model.blocks.parameters() if p.dim() == 2 and id(p) not in qkv]
    fused = [p for p in model.blocks.parameters() if id(p) in qkv]
    other = [p for p in model.parameters() if id(p) not in in_blocks or p.dim() != 2]
    groups = [
        {"params": hidden, "use_muon": True, "lr": muon_lr, "momentum": momentum,
         "weight_decay": muon_weight_decay},
    ]
    # One group per distinct split (all layers share it).
    for split in sorted({tuple(v) for v in qkv.values()}):
        groups.append({
            "params": [p for p in fused if tuple(qkv[id(p)]) == split],
            "use_muon": True, "lr": muon_lr, "momentum": momentum,
            "weight_decay": muon_weight_decay, "split": split,
        })
    groups += [
        {"params": [p for p in other if p.dim() >= 2], "lr": adam_lr, "betas": betas,
         "weight_decay": weight_decay},
        {"params": [p for p in other if p.dim() < 2], "lr": adam_lr, "betas": betas,
         "weight_decay": 0.0},
    ]
    return Muon(groups)
