import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import RotaryEmbedding

HEAD_DIM = 8


def _reference_rope(x, base=10000.0, offset=0):
    """Loop-by-loop RoPE: rotate each (i, i + d/2) pair by an explicit 2x2
    rotation matrix at angle pos * base^(-2i/d)."""
    d = x.shape[-1]
    out = torch.empty_like(x, dtype=torch.float64)
    x = x.double()
    for t in range(x.shape[-2]):
        for i in range(d // 2):
            theta = (t + offset) * base ** (-2 * i / d)
            rot = torch.tensor(
                [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
                dtype=torch.float64,
            )
            pair = torch.stack((x[..., t, i], x[..., t, i + d // 2]), dim=-1)
            rotated = pair @ rot.T
            out[..., t, i] = rotated[..., 0]
            out[..., t, i + d // 2] = rotated[..., 1]
    return out


def test_matches_explicit_rotation_matrices():
    torch.manual_seed(0)
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=16)
    x = torch.randn(2, 3, 16, HEAD_DIM)  # (batch, heads, seq, head_dim)
    torch.testing.assert_close(rope(x).double(), _reference_rope(x), atol=1e-5, rtol=0)


def test_position_zero_is_identity():
    torch.manual_seed(0)
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=4)
    x = torch.randn(1, 1, 4, HEAD_DIM)
    torch.testing.assert_close(rope(x)[..., 0, :], x[..., 0, :])
    # ...and later positions really are changed.
    assert not torch.allclose(rope(x)[..., 1:, :], x[..., 1:, :])


def test_rotation_preserves_vector_norm():
    torch.manual_seed(0)
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=32)
    x = torch.randn(4, 2, 32, HEAD_DIM)
    torch.testing.assert_close(rope(x).norm(dim=-1), x.norm(dim=-1))


def test_attention_score_depends_only_on_relative_position():
    # The property RoPE exists for: q at position m and k at position n give
    # the same dot product as long as m - n is the same.
    torch.manual_seed(0)
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=64)
    q = torch.randn(HEAD_DIM)
    k = torch.randn(HEAD_DIM)

    def score(m, n):
        qm = rope(q.expand(1, HEAD_DIM), offset=m)[0]
        kn = rope(k.expand(1, HEAD_DIM), offset=n)[0]
        return torch.dot(qm, kn)

    for m, n in [(5, 2), (10, 3), (0, 0)]:
        for shift in (1, 7, 40):
            torch.testing.assert_close(score(m + shift, n + shift), score(m, n))
    # Different relative offsets generally give different scores.
    assert not torch.isclose(score(5, 2), score(5, 4))


def test_offset_matches_slice_of_full_sequence():
    # Generating token by token must rotate each new token exactly as if the
    # whole sequence had been processed at once.
    torch.manual_seed(0)
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=16)
    x = torch.randn(2, 3, 16, HEAD_DIM)
    full = rope(x)
    for start in (0, 5, 15):
        torch.testing.assert_close(rope(x[..., start : start + 1, :], offset=start), full[..., start : start + 1, :])


def test_gradient_flows_through_rotation():
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=4)
    x = torch.randn(1, 1, 4, HEAD_DIM, requires_grad=True)
    rope(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_tables_not_in_state_dict():
    assert RotaryEmbedding(HEAD_DIM, max_seq_len=4).state_dict() == {}


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError):
        RotaryEmbedding(7, max_seq_len=4)


def test_sequence_longer_than_max_rejected():
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=4)
    with pytest.raises(ValueError):
        rope(torch.randn(1, 1, 5, HEAD_DIM))
    with pytest.raises(ValueError):
        rope(torch.randn(1, 1, 2, HEAD_DIM), offset=3)
