import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from webbot.model import GPT, GPTConfig, RotaryEmbedding, extend_context, rope_frequencies

D, L, S = 64, 256, 4.0  # head_dim, trained length, extension factor


def _thetas(method, d=D, base=10000.0, factor=S, length=L, alpha=1.0, beta=32.0):
    """Per-pair frequencies and attention scale, one pair at a time."""
    out = []
    for i in range(d // 2):
        theta = base ** (-2 * i / d)
        if method == "pi":
            theta /= factor
        elif method == "ntk":
            theta = (base * factor ** (d / (d - 2))) ** (-2 * i / d)
        elif method == "yarn":
            turns = length / (2 * math.pi / theta)  # rotations within the trained length
            if turns > beta:
                pass
            elif turns < alpha:
                theta /= factor
            else:
                g = (turns - alpha) / (beta - alpha)
                theta = (1 - g) * theta / factor + g * theta
        out.append(theta)
    mscale = 0.1 * math.log(factor) + 1 if method == "yarn" else 1.0
    return out, mscale


def _reference_rope(x, thetas, mscale, offset=0):
    d = x.shape[-1]
    out = torch.empty_like(x, dtype=torch.float64)
    x = x.double()
    for t in range(x.shape[-2]):
        for i in range(d // 2):
            a = (t + offset) * thetas[i]
            c, s = math.cos(a) * mscale, math.sin(a) * mscale
            x1, x2 = x[..., t, i], x[..., t, i + d // 2]
            out[..., t, i] = x1 * c - x2 * s
            out[..., t, i + d // 2] = x1 * s + x2 * c
    return out


def _rope(method, max_len=int(L * S)):
    return RotaryEmbedding(D, max_len, scaling=method, factor=S, original_max_seq_len=L)


@pytest.mark.parametrize("method", [None, "pi", "ntk", "yarn"])
def test_matches_per_pair_reference(method):
    thetas, mscale = _thetas(method)
    inv_freq, got_mscale = rope_frequencies(D, scaling=method, factor=S, original_max_seq_len=L)
    torch.testing.assert_close(inv_freq, torch.tensor(thetas, dtype=torch.float64), rtol=1e-12, atol=0)
    assert got_mscale == pytest.approx(mscale)
    x = torch.randn(2, 6, D, generator=torch.Generator().manual_seed(0))
    for offset in (0, 300, 1000):  # also far beyond the trained length
        torch.testing.assert_close(
            _rope(method)(x, offset=offset).double(), _reference_rope(x, thetas, mscale, offset),
            atol=1e-4, rtol=0,
        )


def test_pi_reads_position_m_times_s_as_m():
    plain, pi = RotaryEmbedding(D, L), _rope("pi")
    x = torch.randn(1, D)
    for m in (1, 7, 200):
        torch.testing.assert_close(pi(x, offset=int(m * S)), plain(x, offset=m), atol=1e-5, rtol=0)


def test_ntk_keeps_the_fastest_pair_and_divides_the_slowest_by_s():
    plain, _ = rope_frequencies(D)
    ntk, _ = rope_frequencies(D, scaling="ntk", factor=S)
    assert ntk[0] == plain[0]
    assert ntk[-1].item() == pytest.approx(plain[-1].item() / S, rel=1e-12)
    ratio = plain / ntk
    assert torch.all(ratio[1:] > ratio[:-1])  # stretched more the slower the pair


def test_yarn_keeps_fast_pairs_interpolates_slow_ones_and_blends_between():
    plain, _ = rope_frequencies(D)
    yarn, _ = rope_frequencies(D, scaling="yarn", factor=S, original_max_seq_len=L)
    turns = L * plain / (2 * math.pi)
    fast, slow = turns > 32, turns < 1
    mid = ~fast & ~slow
    assert fast.any() and slow.any() and mid.any()
    assert torch.equal(yarn[fast], plain[fast])
    torch.testing.assert_close(yarn[slow], plain[slow] / S, rtol=1e-12, atol=0)
    assert torch.all((yarn[mid] < plain[mid]) & (yarn[mid] > plain[mid] / S))


def test_yarn_scales_attention_logits_by_mscale_squared():
    rope = _rope("yarn")
    q, k = torch.randn(2, 1, D, generator=torch.Generator().manual_seed(1)).unbind(0)
    m2 = (0.1 * math.log(S) + 1) ** 2
    for pos in (0, 500):
        got = (rope(q, offset=pos) * rope(k, offset=pos)).sum()
        torch.testing.assert_close(got, m2 * (q * k).sum(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["pi", "ntk", "yarn"])
def test_scores_still_depend_only_on_relative_position(method):
    rope = _rope(method)
    q, k = torch.randn(2, 1, D, generator=torch.Generator().manual_seed(2)).unbind(0)

    def score(m, n):
        return (rope(q, offset=m) * rope(k, offset=n)).sum()

    for m, n in [(5, 2), (300, 10)]:
        torch.testing.assert_close(score(m + 600, n + 600), score(m, n), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("method", ["pi", "ntk", "yarn"])
def test_factor_one_is_plain_rope(method):
    plain = RotaryEmbedding(D, L)
    same = RotaryEmbedding(D, L, scaling=method, factor=1.0, original_max_seq_len=L)
    assert torch.equal(same.cos, plain.cos) and torch.equal(same.sin, plain.sin)


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        rope_frequencies(D, scaling="linear", factor=2)
    with pytest.raises(ValueError):
        rope_frequencies(D, scaling="pi", factor=0.5)
    with pytest.raises(ValueError):
        rope_frequencies(D, scaling="yarn", factor=2)  # needs the trained length


def _tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, max_seq_len=16, d_model=32, n_layers=2, n_heads=2))


@pytest.mark.parametrize("method", ["pi", "ntk", "yarn"])
def test_extend_context_reuses_the_weights(method):
    model = _tiny()
    ext = extend_context(model, 64, method)
    assert ext.config.max_seq_len == 64
    assert ext.config.rope_factor == 4.0 and ext.config.rope_original_max_seq_len == 16
    for (n, p), q in zip(model.state_dict().items(), ext.state_dict().values()):
        assert torch.equal(p, q), n
    idx = torch.randint(0, 50, (2, 64))
    with pytest.raises(ValueError):
        model(idx)
    logits, _ = ext(idx)
    assert logits.shape == (2, 64, 50) and torch.isfinite(logits).all()
    rope = ext.blocks[0].attn.rope
    expected, _ = rope_frequencies(16, scaling=method, factor=4.0, original_max_seq_len=16)
    torch.testing.assert_close(rope.inv_freq, expected)


def test_extend_context_rejects_shrinking_and_double_extension():
    model = _tiny()
    with pytest.raises(ValueError):
        extend_context(model, 16)
    with pytest.raises(ValueError):
        extend_context(extend_context(model, 32), 64)
