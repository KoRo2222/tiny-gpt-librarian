import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from webbot.model.flash_attention import flash_attention


def naive_attention(q, k, v, causal=True):
    """Reference: the full score matrix, masked so that query i (at
    position len_k - len_q + i) sees keys at positions <= it."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        len_q, len_k = q.shape[-2], k.shape[-2]
        q_pos = torch.arange(len_k - len_q, len_k)
        k_pos = torch.arange(len_k)
        scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
    return scores.softmax(dim=-1) @ v


def _qkv(len_q, len_k, d=8, batch=(2, 3), dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda n: torch.randn(*batch, n, d, generator=g, dtype=dtype)  # noqa: E731
    return mk(len_q), mk(len_k), mk(len_k)


# (len_q, len_k, block_q, block_k, causal): lengths that are and aren't
# multiples of the blocks, blocks larger than the sequence, len_q < len_k
# (queries after a KV cache), and the non-causal case.
CASES = [
    (16, 16, 4, 4, True),
    (13, 13, 4, 5, True),
    (13, 13, 32, 32, True),
    (1, 9, 4, 4, True),
    (5, 17, 2, 3, True),
    (17, 17, 1, 1, True),
    (13, 13, 4, 5, False),
    (16, 16, 32, 32, False),
]


@pytest.mark.parametrize("len_q,len_k,bq,bk,causal", CASES)
def test_forward_matches_naive(len_q, len_k, bq, bk, causal):
    q, k, v = _qkv(len_q, len_k)
    out = flash_attention(q, k, v, causal=causal, block_q=bq, block_k=bk)
    torch.testing.assert_close(out, naive_attention(q, k, v, causal), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("len_q,len_k,bq,bk,causal", CASES)
def test_gradients_match_naive(len_q, len_k, bq, bk, causal):
    q, k, v = (t.requires_grad_() for t in _qkv(len_q, len_k))
    grad_out = torch.randn(*q.shape[:-1], v.shape[-1], dtype=q.dtype)

    flash = flash_attention(q, k, v, causal=causal, block_q=bq, block_k=bk)
    got = torch.autograd.grad(flash, (q, k, v), grad_out)
    ref = torch.autograd.grad(naive_attention(q, k, v, causal), (q, k, v), grad_out)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, rtol=1e-10, atol=1e-10)


def test_gradcheck():
    q, k, v = (t.requires_grad_() for t in _qkv(5, 7, d=3, batch=(2,)))
    fn = lambda q, k, v: flash_attention(q, k, v, causal=True, block_q=2, block_k=3)  # noqa: E731
    assert torch.autograd.gradcheck(fn, (q, k, v))


def test_saved_log_sum_exp_is_exact():
    q, k, v = _qkv(11, 11)
    out = flash_attention(q.requires_grad_(), k, v, block_q=4, block_k=3)
    lse = out.grad_fn.saved_tensors[4]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(torch.ones(11, 11, dtype=torch.bool).triu(1), float("-inf"))
    torch.testing.assert_close(lse, scores.logsumexp(dim=-1), rtol=1e-12, atol=1e-12)


def _largest_saved_for_backward(fn, *args):
    """numel of the largest tensor autograd saves while running fn."""
    sizes = []

    def pack(t):
        sizes.append(t.numel())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return max(sizes)


def test_backward_keeps_no_quadratic_tensor():
    # What autograd keeps for backward is O(seq * head_dim), never seq x seq.
    n, d = 256, 8
    q, k, v = (t.requires_grad_() for t in _qkv(n, n, d=d, batch=(1, 2), dtype=torch.float32))
    flash = lambda q, k, v: flash_attention(q, k, v, block_q=32, block_k=32)  # noqa: E731
    assert _largest_saved_for_backward(flash, q, k, v) == 2 * n * d
    assert _largest_saved_for_backward(naive_attention, q, k, v) == 2 * n * n


def test_low_precision_input_is_accumulated_in_float32():
    q, k, v = _qkv(40, 40, dtype=torch.float32)
    out = flash_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), block_q=8, block_k=8)
    assert out.dtype == torch.bfloat16
    ref = naive_attention(q.bfloat16().float(), k.bfloat16().float(), v.bfloat16().float())
    # Only the final rounding to bf16 separates them.
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    assert (out.float() - ref).abs().max() <= ref.abs().max() * 2**-8


def test_autocast_does_not_lower_the_inner_precision():
    # Under bf16 autocast the tile matmuls must still run in float32, so the
    # result equals the autocast-free run on the same bf16 inputs.
    q, k, v = (t.bfloat16() for t in _qkv(40, 40, dtype=torch.float32))
    plain = flash_attention(q, k, v, block_q=8, block_k=8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        auto = flash_attention(q, k, v, block_q=8, block_k=8)
    torch.testing.assert_close(auto, plain, rtol=0, atol=0)

    qg, kg, vg = (t.float().requires_grad_() for t in (q, k, v))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        flash_attention(qg, kg, vg, block_q=8, block_k=8).float().sum().backward()
    ref = [t.float().requires_grad_() for t in (q, k, v)]
    flash_attention(*ref, block_q=8, block_k=8).sum().backward()
    for got, want in zip((qg, kg, vg), ref):
        torch.testing.assert_close(got.grad, want.grad, rtol=0, atol=0)


def test_causal_rejects_more_queries_than_keys():
    q, k, v = _qkv(5, 3)
    with pytest.raises(ValueError):
        flash_attention(q, k, v)


def _models(seq_len=160):
    from webbot.model import GPT, GPTConfig

    torch.manual_seed(0)
    cfg = dict(vocab_size=50, max_seq_len=seq_len, d_model=32, n_layers=2, n_heads=4)
    naive = GPT(GPTConfig(**cfg, attn_impl="naive"))
    flash = GPT(GPTConfig(**cfg, attn_impl="flash"))
    flash.load_state_dict(naive.state_dict())
    return naive, flash


def test_model_loss_and_gradients_match_naive():
    # 160 tokens span two of flash_attention's default 128-token blocks.
    naive, flash = _models()
    idx = torch.randint(0, 50, (2, 160), generator=torch.Generator().manual_seed(1))
    targets = idx.roll(-1, dims=1)
    logits_n, loss_n = naive(idx, targets)
    logits_f, loss_f = flash(idx, targets)
    torch.testing.assert_close(logits_f, logits_n, rtol=1e-5, atol=1e-5)
    loss_n.backward()
    loss_f.backward()
    for (name, pn), pf in zip(naive.named_parameters(), flash.parameters()):
        torch.testing.assert_close(pf.grad, pn.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_model_generation_with_kv_cache_matches_naive():
    naive, flash = _models()
    prompt = torch.randint(0, 50, (2, 7), generator=torch.Generator().manual_seed(2))
    out_n = naive.generate(prompt, max_new_tokens=20, temperature=0)
    out_f = flash.generate(prompt, max_new_tokens=20, temperature=0)
    assert torch.equal(out_f, out_n)


def test_unknown_attn_impl_rejected():
    from webbot.model import GPT, GPTConfig

    with pytest.raises(ValueError):
        GPT(GPTConfig(vocab_size=10, attn_impl="fast"))
