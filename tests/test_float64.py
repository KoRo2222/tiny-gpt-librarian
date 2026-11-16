"""Ops that compute in float32 under half precision must not lower float64
inputs: float64 is how the tests compare implementations exactly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from webbot.model import GPT, GPTConfig, RMSNorm
from webbot.model.attention import attention_weights


def _x(*shape):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))


def test_rmsnorm_keeps_float64():
    norm = RMSNorm(16).double()
    x = _x(3, 16) * 1e3
    expected = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + norm.eps)
    out = norm(x)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, expected, rtol=1e-14, atol=0)


def test_attention_softmax_keeps_float64():
    scores = _x(2, 5, 5) * 10
    out = attention_weights(scores)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, scores.softmax(dim=-1), rtol=0, atol=0)


def test_loss_keeps_float64():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=30, max_seq_len=16, d_model=16, n_layers=1, n_heads=2)).double()
    idx = torch.randint(0, 30, (2, 8))
    logits, loss = model(idx, idx)
    assert loss.dtype == torch.float64
    expected = F.cross_entropy(logits.flatten(0, 1), idx.flatten())
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
