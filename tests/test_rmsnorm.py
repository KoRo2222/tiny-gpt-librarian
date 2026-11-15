import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import RMSNorm

DIM = 16


def test_matches_formula_written_out_by_hand():
    torch.manual_seed(0)
    norm = RMSNorm(DIM)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(DIM))
    x = torch.randn(2, 5, DIM)
    rms = (x.pow(2).sum(dim=-1, keepdim=True) / DIM + norm.eps).sqrt()
    torch.testing.assert_close(norm(x), x / rms * norm.weight)


def test_output_has_unit_rms_with_initial_weight():
    torch.manual_seed(0)
    x = torch.randn(4, 7, DIM) * 50 + 3
    out = RMSNorm(DIM)(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=0)


def test_invariant_to_rescaling_input():
    # The point of normalizing: activations blowing up or shrinking by a
    # constant factor leave the output unchanged. (Only up to eps: vectors
    # with RMS near sqrt(eps) are deliberately not blown up.)
    torch.manual_seed(0)
    norm = RMSNorm(DIM)
    x = torch.randn(3, DIM)
    for c in (0.1, 3.0, 1000.0):
        torch.testing.assert_close(norm(c * x), norm(x), atol=1e-4, rtol=1e-4)


def test_does_not_subtract_mean_unlike_layernorm():
    torch.manual_seed(0)
    norm = RMSNorm(DIM)
    layer_norm = torch.nn.LayerNorm(DIM, eps=norm.eps, elementwise_affine=False)
    x = torch.randn(3, DIM)
    centered = x - x.mean(dim=-1, keepdim=True)
    # For zero-mean vectors RMS == std, so RMSNorm and LayerNorm agree...
    torch.testing.assert_close(norm(centered), layer_norm(centered), atol=1e-5, rtol=1e-5)
    # ...but a shared offset is kept rather than removed.
    shifted = centered + 5.0
    assert not torch.allclose(norm(shifted), layer_norm(shifted), atol=1e-3)
    assert (norm(shifted).mean(dim=-1) > 0.5).all()


def test_each_vector_normalized_independently():
    torch.manual_seed(0)
    norm = RMSNorm(DIM)
    x = torch.randn(2, 6, DIM)
    y = x.clone()
    y[0, 0] *= 100  # scaling one token must not affect any other token
    torch.testing.assert_close(norm(y)[0, 1:], norm(x)[0, 1:])
    torch.testing.assert_close(norm(y)[1], norm(x)[1])


def test_zero_vector_stays_finite():
    out = RMSNorm(DIM)(torch.zeros(2, DIM))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_weight_starts_at_one_and_learns():
    torch.manual_seed(0)
    norm = RMSNorm(DIM)
    torch.testing.assert_close(norm.weight.detach(), torch.ones(DIM))
    # Learning a target gain of 2 on every dim: gradient must push the
    # weight upward toward 2.
    x = torch.randn(32, DIM)
    target = 2 * norm(x).detach()
    opt = torch.optim.SGD(norm.parameters(), lr=1.0)
    for _ in range(200):
        opt.zero_grad()
        (norm(x) - target).pow(2).mean().backward()
        opt.step()
    torch.testing.assert_close(norm.weight.detach(), torch.full((DIM,), 2.0), atol=1e-2, rtol=0)


def test_half_precision_input_keeps_dtype():
    torch.manual_seed(0)
    norm = RMSNorm(DIM)
    x = torch.randn(3, DIM) * 1e-3
    out = norm(x.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), norm(x), atol=2e-2, rtol=2e-2)
