import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from webbot.model import GPT, GPTConfig, MoE, SwiGLU

D = 16


def _moe(n_experts=4, top_k=2, seed=0):
    torch.manual_seed(seed)
    return MoE(D, n_experts, top_k, hidden_dim=32).double()


def _x(*shape, seed=1):
    return torch.randn(*shape, D, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def _reference(moe, x):
    """One token at a time: rank the experts by router probability, take
    the top k, weight their outputs by the gates and sum."""
    outs = []
    for t in x.reshape(-1, D):
        p = moe.router(t).softmax(dim=-1)
        top = sorted(range(moe.n_experts), key=lambda e: -p[e].item())[: moe.top_k]
        gates = [p[e] for e in top]
        if moe.top_k > 1:
            total = sum(gates)
            gates = [g / total for g in gates]
        outs.append(sum(g * moe.experts[e](t) for g, e in zip(gates, top)))
    return torch.stack(outs).view(x.shape)


def _reference_aux(moe, x):
    xs = x.reshape(-1, D)
    probs = moe.router(xs).softmax(dim=-1)
    frac = []
    for e in range(moe.n_experts):
        hits = sum(e in torch.topk(p, moe.top_k).indices.tolist() for p in probs)
        frac.append(hits / (len(xs) * moe.top_k))
    return moe.n_experts * sum(f * probs[:, e].mean() for e, f in enumerate(frac))


@pytest.mark.parametrize("n_experts,top_k", [(4, 1), (4, 2), (5, 3), (3, 3), (8, 1)])
def test_matches_per_token_reference(n_experts, top_k):
    moe = _moe(n_experts, top_k)
    x = _x(2, 7)  # 14 tokens: with 8 experts some get none
    out = moe(x)
    torch.testing.assert_close(out, _reference(moe, x), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(moe.aux_loss, _reference_aux(moe, x), rtol=1e-12, atol=1e-12)
    assert moe.last_counts.sum() == 14 * top_k

    grad = torch.randn_like(out)
    params = list(moe.parameters())
    got = torch.autograd.grad(out, params, grad, allow_unused=True)
    want = torch.autograd.grad(_reference(moe, x), params, grad, allow_unused=True)
    for (name, _), g, w in zip(moe.named_parameters(), got, want):
        if w is None:  # an expert no token was routed to
            assert g is None or not g.any(), name
        else:
            torch.testing.assert_close(g, w, rtol=1e-10, atol=1e-12, msg=name)


def test_router_learns_from_the_main_loss():
    # Gates are router probabilities, so the output depends on the router
    # (with top_k == 1 the gate is not renormalized to 1).
    for top_k in (1, 2):
        moe = _moe(4, top_k)
        moe(_x(10)).pow(2).sum().backward()
        assert moe.router.weight.grad.abs().sum() > 0


def test_identical_experts_reduce_to_a_dense_ffn():
    moe = _moe(4, 2)
    dense = SwiGLU(D, 32).double()
    for expert in moe.experts:
        expert.load_state_dict(dense.state_dict())
    x = _x(3, 5)
    torch.testing.assert_close(moe(x), dense(x), rtol=1e-12, atol=1e-12)


def test_aux_loss_is_one_for_uniform_routing_and_large_when_collapsed():
    moe = _moe(4, 1)
    with torch.no_grad():
        moe.router.weight.zero_()
    moe(_x(32))
    assert moe.aux_loss.item() == pytest.approx(1.0)

    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 1.0
    moe(_x(32) + 20.0)  # every token's expert-0 logit dominates
    assert moe.last_counts.tolist() == [32, 0, 0, 0]
    assert moe.aux_loss.item() > 3.9  # the maximum is n_experts = 4


def test_minimizing_the_aux_loss_spreads_the_tokens():
    moe = _moe(4, 2)
    shift = torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    x = _x(64) + shift
    with torch.no_grad():
        moe.router.weight[0] += shift / shift.norm()  # expert 0 is overloaded
    moe(x)
    before = moe.last_counts[0].item()
    aux_before = moe.aux_loss.item()
    opt = torch.optim.SGD(moe.router.parameters(), lr=0.5)
    for _ in range(30):
        moe(x)
        opt.zero_grad()
        moe.aux_loss.backward()
        opt.step()
    moe(x)
    assert before == 64  # every token had expert 0 in its top 2
    assert moe.last_counts[0].item() < before
    assert moe.aux_loss.item() < aux_before


def test_invalid_top_k_rejected():
    for top_k in (0, 5):
        with pytest.raises(ValueError):
            MoE(D, 4, top_k)


def test_bf16_autocast_runs_close_to_float32():
    torch.manual_seed(0)
    moe = MoE(D, 4, 2, hidden_dim=32)
    x = torch.randn(4, 9, D)
    ref = moe(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = moe(x)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=0.05, atol=0.02)


# --- whole model -----------------------------------------------------------


def _gpt(**kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, ffn_hidden_dim=64)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg))


def test_training_loss_adds_the_weighted_aux_loss_eval_does_not():
    model = _gpt(n_experts=4, moe_top_k=2, moe_aux_coef=0.1)
    idx = torch.randint(0, 50, (2, 16), generator=torch.Generator().manual_seed(1))
    targets = idx.roll(-1, dims=1)
    logits, loss = model(idx, targets)
    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    aux = model.moe_aux_loss()
    assert aux.item() >= 1.0 - 1e-6  # the minimum of the balancing loss
    assert torch.allclose(aux, sum(b.ffn.aux_loss for b in model.blocks) / 2)
    torch.testing.assert_close(loss, ce + 0.1 * aux)
    model.eval()
    _, eval_loss = model(idx, targets)
    torch.testing.assert_close(eval_loss, ce)


def test_active_params_count_top_k_experts():
    dense = _gpt()
    moe1 = _gpt(n_experts=4, moe_top_k=1)
    moe2 = _gpt(n_experts=4, moe_top_k=2)
    router = 2 * 32 * 4  # layers * d_model * n_experts
    ffn = 3 * 32 * 64  # one SwiGLU
    assert moe1.num_active_params() == dense.num_params() + router
    assert moe2.num_active_params() == dense.num_params() + router + 2 * ffn
    assert moe1.num_params() == dense.num_params() + router + 2 * 3 * ffn


def test_every_expert_gets_the_scaled_residual_init():
    model = _gpt(n_experts=8, n_layers=8)
    residual_std = 0.02 / math.sqrt(2 * 8)
    for block in model.blocks:
        for expert in block.ffn.experts:
            assert expert.w_down.weight.std().item() == pytest.approx(residual_std, rel=0.15)


def test_generation_with_kv_cache_matches_no_cache():
    model = _gpt(n_experts=4, moe_top_k=2)
    prompt = torch.randint(0, 50, (2, 5), generator=torch.Generator().manual_seed(2))
    out = model.generate(prompt, max_new_tokens=12, temperature=0)
    idx = prompt
    for _ in range(12):
        logits, _ = model(idx)
        idx = torch.cat((idx, logits[:, -1:].argmax(dim=-1)), dim=1)
    assert torch.equal(out, idx)


def test_moe_model_learns():
    model = _gpt(n_experts=4, moe_top_k=1)
    idx = torch.randint(0, 50, (1, 33), generator=torch.Generator().manual_seed(3))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    _, loss = model(idx[:, :-1], idx[:, 1:])
    assert loss.item() < 0.1
