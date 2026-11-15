import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from webbot.model import GPT, CausalSelfAttention, GPTConfig, RotaryEmbedding
from webbot.model.flash_attention import flash_attention

D, H, HD = 32, 4, 8  # d_model, n_heads, head_dim
IMPLS = ["naive", "flash"]


def _pair(n_kv, impl, block=None):
    """A GQA layer and an ordinary multi-head layer computing the same
    function: the MHA layer's K/V weights are the GQA ones copied to every
    query head of the group (query head h uses K/V head h // group)."""
    torch.manual_seed(0)
    rope = RotaryEmbedding(HD, max_seq_len=64)
    gqa = CausalSelfAttention(D, H, rope, impl, n_kv_heads=n_kv).double()
    mha = CausalSelfAttention(D, H, rope, impl).double()
    wq, wk, wv = gqa.qkv.weight.detach().split((D, n_kv * HD, n_kv * HD))
    expand = lambda w: w.view(n_kv, HD, D).repeat_interleave(H // n_kv, dim=0).reshape(D, D)  # noqa: E731
    with torch.no_grad():
        mha.qkv.weight.copy_(torch.cat((wq, expand(wk), expand(wv))))
        mha.out.weight.copy_(gqa.out.weight)
    return gqa, mha


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("n_kv", [1, 2])
def test_gqa_equals_mha_with_kv_heads_copied_across_the_group(impl, n_kv):
    gqa, mha = _pair(n_kv, impl)
    x = torch.randn(2, 20, D, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    xg, xm = x.clone().requires_grad_(), x.clone().requires_grad_()
    out_g, _ = gqa(xg)
    out_m, _ = mha(xm)
    torch.testing.assert_close(out_g, out_m, rtol=1e-12, atol=1e-12)

    grad = torch.randn_like(out_g)
    out_g.backward(grad)
    out_m.backward(grad)
    torch.testing.assert_close(xg.grad, xm.grad, rtol=1e-10, atol=1e-12)
    # A shared K/V row gets the summed gradient of all its copies.
    g = gqa.qkv.weight.grad.split((D, n_kv * HD, n_kv * HD))
    m = mha.qkv.weight.grad.split(D)
    fold = lambda w: w.view(n_kv, H // n_kv, HD, D).sum(dim=1).reshape(n_kv * HD, D)  # noqa: E731
    torch.testing.assert_close(g[0], m[0], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(g[1], fold(m[1]), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(g[2], fold(m[2]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("impl", IMPLS)
def test_kv_cache_holds_only_the_kv_heads(impl):
    gqa, mha = _pair(1, impl)
    x = torch.randn(2, 10, D, dtype=torch.float64)
    full, _ = mha(x)
    out, cache = gqa(x[:, :6])
    outs = [out]
    for t in range(6, 10):
        out, cache = gqa(x[:, t : t + 1], cache)
        outs.append(out)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-12, atol=1e-12)
    assert cache[0].shape == cache[1].shape == (2, 1, 10, HD)


def test_kv_projection_shrinks_by_the_group_size():
    cfg = dict(vocab_size=50, d_model=D, n_layers=3, n_heads=H)
    mha = GPT(GPTConfig(**cfg))
    for n_kv in (1, 2, 4):
        gpt = GPT(GPTConfig(**cfg, n_kv_heads=n_kv))
        assert gpt.blocks[0].attn.qkv.weight.shape == (D + 2 * n_kv * HD, D)
        saved = 3 * 2 * (H - n_kv) * HD * D  # K and V rows in each of 3 layers
        assert mha.num_params() - gpt.num_params() == saved


def test_invalid_n_kv_heads_rejected():
    rope = RotaryEmbedding(HD, max_seq_len=8)
    for n_kv in (0, 3, 8):
        with pytest.raises(ValueError):
            CausalSelfAttention(D, H, rope, n_kv_heads=n_kv)


# --- flash_attention with broadcast leading dims ---------------------------


def _naive(q, k, v, causal=True):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        len_q, len_k = q.shape[-2], k.shape[-2]
        future = torch.arange(len_k)[None, :] > torch.arange(len_k - len_q, len_k)[:, None]
        scores = scores.masked_fill(future, float("-inf"))
    return scores.softmax(dim=-1) @ v


# (q batch dims, k/v batch dims): K/V shared across a query group, and the
# reverse (one query set against several K/V sets).
SHAPES = [((2, 3, 4), (2, 3, 1)), ((2, 1, 1), (2, 3, 4)), ((3, 1), (1, 2))]


@pytest.mark.parametrize("qb,kb", SHAPES)
@pytest.mark.parametrize("len_q,len_k,causal", [(13, 13, True), (5, 17, True), (13, 9, False)])
def test_flash_broadcast_matches_naive(qb, kb, len_q, len_k, causal):
    g = torch.Generator().manual_seed(0)
    mk = lambda b, n: torch.randn(*b, n, 8, generator=g, dtype=torch.float64).requires_grad_()  # noqa: E731
    q, k, v = mk(qb, len_q), mk(kb, len_k), mk(kb, len_k)
    out = flash_attention(q, k, v, causal=causal, block_q=4, block_k=5)
    ref = _naive(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    grad = torch.randn(ref.shape, generator=g, dtype=torch.float64)
    got = torch.autograd.grad(out, (q, k, v), grad)
    want = torch.autograd.grad(ref, (q, k, v), grad)
    for a, b in zip(got, want):
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_flash_broadcast_gradcheck():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 2, 3, 6, 4, generator=g, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 1, 6, 4, generator=g, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 1, 6, 4, generator=g, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda q, k, v: flash_attention(q, k, v, block_q=2, block_k=4), (q, k, v)
    )


def _saved_numel(fn, *args):
    total = 0

    def pack(t):
        nonlocal total
        total += t.numel()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return total


def test_flash_saves_kv_unexpanded():
    # Saved for backward: q, k, v, out, lse. With one K/V head for 4 query
    # heads, k and v must be saved at 1/4 of the per-query-head size.
    b, n, d = 2, 32, 8
    q = torch.randn(b, 1, H, n, d, requires_grad=True)
    k = torch.randn(b, 1, 1, n, d, requires_grad=True)
    v = torch.randn(b, 1, 1, n, d, requires_grad=True)
    saved = _saved_numel(flash_attention, q, k, v)
    assert saved == 2 * b * H * n * d + 2 * b * n * d + b * H * n


# --- whole model -----------------------------------------------------------


def _models(n_kv, seq_len=160):
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, max_seq_len=seq_len, d_model=D, n_layers=2, n_heads=H, n_kv_heads=n_kv)
    naive = GPT(GPTConfig(**cfg, attn_impl="naive"))
    flash = GPT(GPTConfig(**cfg, attn_impl="flash"))
    flash.load_state_dict(naive.state_dict())
    return naive, flash


@pytest.mark.parametrize("n_kv", [1, 2])
def test_gqa_model_flash_matches_naive(n_kv):
    naive, flash = _models(n_kv)
    idx = torch.randint(0, 50, (2, 160), generator=torch.Generator().manual_seed(1))
    targets = idx.roll(-1, dims=1)
    logits_n, loss_n = naive(idx, targets)
    logits_f, loss_f = flash(idx, targets)
    torch.testing.assert_close(logits_f, logits_n, rtol=1e-5, atol=1e-5)
    loss_n.backward()
    loss_f.backward()
    for (name, pn), pf in zip(naive.named_parameters(), flash.parameters()):
        torch.testing.assert_close(pf.grad, pn.grad, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("impl", IMPLS)
def test_gqa_model_generation_matches_no_cache(impl):
    model = _models(1, seq_len=32)[impl == "flash"]
    prompt = torch.randint(0, 50, (2, 5), generator=torch.Generator().manual_seed(2))
    out = model.generate(prompt, max_new_tokens=12, temperature=0)
    # Greedy decoding without the cache: re-run the whole prefix each step.
    idx = prompt
    for _ in range(12):
        logits, _ = model(idx)
        idx = torch.cat((idx, logits[:, -1:].argmax(dim=-1)), dim=1)
    assert torch.equal(out, idx)


def test_mqa_model_learns():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, max_seq_len=32, d_model=D, n_layers=2, n_heads=H, n_kv_heads=1))
    idx = torch.randint(0, 50, (1, 33), generator=torch.Generator().manual_seed(3))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
