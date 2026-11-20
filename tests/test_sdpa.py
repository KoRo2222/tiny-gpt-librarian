import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, CausalSelfAttention, GPTConfig, RotaryEmbedding

D, H = 32, 4


def _pair(n_kv):
    torch.manual_seed(0)
    rope = RotaryEmbedding(D // H, max_seq_len=64)
    naive = CausalSelfAttention(D, H, rope, "naive", n_kv_heads=n_kv).double()
    sdpa = CausalSelfAttention(D, H, rope, "sdpa", n_kv_heads=n_kv).double()
    sdpa.load_state_dict(naive.state_dict())
    return naive, sdpa


def _x(n=2, seq=20, seed=1):
    return torch.randn(n, seq, D, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("n_kv", [4, 2, 1])
def test_matches_naive_with_gradients(n_kv):
    naive, sdpa = _pair(n_kv)
    x = _x()
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya, _ = naive(xa)
    yb, _ = sdpa(xb)
    torch.testing.assert_close(yb, ya, rtol=1e-12, atol=1e-12)
    g = torch.randn_like(ya)
    ya.backward(g)
    yb.backward(g)
    torch.testing.assert_close(xb.grad, xa.grad, rtol=1e-10, atol=1e-12)
    for (n, p), q in zip(naive.named_parameters(), sdpa.parameters()):
        torch.testing.assert_close(q.grad, p.grad, rtol=1e-10, atol=1e-12, msg=n)


@pytest.mark.parametrize("n_kv", [4, 1])
@pytest.mark.parametrize("chunks", [(6, 1, 1, 1, 1), (5, 4, 3), (1, 9)])
def test_kv_cache_in_chunks_matches_the_full_pass(n_kv, chunks):
    # After a cache, a chunk of several new tokens needs the offset causal
    # mask (is_causal=True would align it at the top left).
    naive, sdpa = _pair(n_kv)
    x = _x(seq=sum(chunks))
    full, _ = naive(x)
    outs, cache, start = [], None, 0
    for n in chunks:
        out, cache = sdpa(x[:, start : start + n], cache)
        outs.append(out)
        start += n
    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-12, atol=1e-12)
    assert cache[0].shape == (2, n_kv, sum(chunks), D // H)


def test_model_generation_matches_naive():
    torch.manual_seed(0)
    naive = GPT(GPTConfig(vocab_size=50, max_seq_len=48, d_model=D, n_layers=2, n_heads=H, n_kv_heads=2))
    sdpa = copy.deepcopy(naive)
    for block in sdpa.blocks:
        block.attn.attn_impl = "sdpa"
    prompt = torch.randint(0, 50, (2, 7), generator=torch.Generator().manual_seed(2))
    assert torch.equal(sdpa.generate(prompt, 20, temperature=0), naive.generate(prompt, 20, temperature=0))
    idx = torch.randint(0, 50, (2, 40), generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(sdpa(idx, idx)[1], naive(idx, idx)[1], rtol=1e-5, atol=1e-6)
