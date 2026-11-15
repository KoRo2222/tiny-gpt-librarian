import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import SwiGLU
from webbot.model.swiglu import default_hidden_dim

D_MODEL = 16
HIDDEN = 32


def test_matches_formula_written_out_by_hand():
    torch.manual_seed(0)
    ffn = SwiGLU(D_MODEL, HIDDEN)
    x = torch.randn(2, 5, D_MODEL)

    gate = x @ ffn.w_gate.weight.T
    up = x @ ffn.w_up.weight.T
    silu = gate * (1 / (1 + torch.exp(-gate)))  # SiLU = x * sigmoid(x)
    expected = (silu * up) @ ffn.w_down.weight.T

    torch.testing.assert_close(ffn(x), expected)


def test_closed_gate_blocks_everything():
    # SiLU(0) = 0, so with the gate branch zeroed nothing reaches the output
    # no matter what the up branch carries: the gate really gates.
    torch.manual_seed(0)
    ffn = SwiGLU(D_MODEL, HIDDEN)
    with torch.no_grad():
        ffn.w_gate.weight.zero_()
    out = ffn(torch.randn(3, D_MODEL) * 10)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_is_nonlinear():
    torch.manual_seed(0)
    ffn = SwiGLU(D_MODEL, HIDDEN)
    x = torch.randn(4, D_MODEL)
    assert not torch.allclose(ffn(2 * x), 2 * ffn(x))


def test_applied_to_each_position_independently():
    # A feed-forward block must not mix information across positions
    # (that is attention's job).
    torch.manual_seed(0)
    ffn = SwiGLU(D_MODEL, HIDDEN)
    x = torch.randn(2, 6, D_MODEL)
    per_token = torch.stack(
        [torch.stack([ffn(x[b, t]) for t in range(6)]) for b in range(2)]
    )
    torch.testing.assert_close(ffn(x), per_token)


def test_default_hidden_dim_keeps_param_count_of_4x_ffn():
    # 3 matrices of d x (8d/3) == 2 matrices of d x 4d.
    d = 768
    ffn = SwiGLU(d)
    assert ffn.w_gate.out_features == default_hidden_dim(d) == 2048
    n_params = sum(p.numel() for p in ffn.parameters())
    assert n_params == 2 * d * 4 * d


def test_default_hidden_dim_rounds_up_to_multiple():
    assert default_hidden_dim(64) == 192  # 170.6 -> next multiple of 64
    assert default_hidden_dim(64, multiple_of=8) == 176


def test_gradient_reaches_all_three_weights():
    torch.manual_seed(0)
    ffn = SwiGLU(D_MODEL, HIDDEN)
    ffn(torch.randn(3, D_MODEL)).pow(2).sum().backward()
    for w in (ffn.w_gate, ffn.w_up, ffn.w_down):
        assert w.weight.grad is not None and w.weight.grad.abs().sum() > 0


def test_training_fits_a_nonlinear_target():
    # A few hundred optimizer steps should drive the error on a fixed
    # nonlinear mapping well down, i.e. gradients point the right way.
    torch.manual_seed(0)
    ffn = SwiGLU(D_MODEL, HIDDEN)
    x = torch.randn(64, D_MODEL)
    target = torch.tanh(x) * x.roll(1, dims=-1)
    opt = torch.optim.Adam(ffn.parameters(), lr=1e-2)

    def loss():
        return (ffn(x) - target).pow(2).mean()

    initial = loss().item()
    for _ in range(300):
        opt.zero_grad()
        l = loss()
        l.backward()
        opt.step()
    assert loss().item() < 0.2 * initial
