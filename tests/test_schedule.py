import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.train import d2z_multiplier, d2z_scheduler


def _curve(total, warmup):
    return [d2z_multiplier(s, total, warmup) for s in range(total)]


def test_warmup_rises_linearly_to_peak():
    curve = _curve(total=100, warmup=10)
    assert curve[:10] == pytest.approx([i / 10 for i in range(1, 11)])


def test_decays_linearly_to_exactly_zero():
    total, warmup = 100, 10
    curve = _curve(total, warmup)
    decay = curve[warmup:]
    assert decay[0] == pytest.approx(1.0)
    # Constant step size: linear, not cosine.
    diffs = [a - b for a, b in zip(decay, decay[1:])]
    assert diffs == pytest.approx([1 / (total - warmup)] * len(diffs))
    # The last update is the smallest non-zero one; zero from total on.
    assert curve[-1] == pytest.approx(1 / (total - warmup))
    assert d2z_multiplier(total, total, warmup) == 0.0
    assert d2z_multiplier(total + 50, total, warmup) == 0.0


def test_peak_reached_once_and_never_exceeded():
    curve = _curve(total=100, warmup=10)
    assert max(curve) == pytest.approx(1.0)
    assert curve.index(max(curve)) == 9
    assert min(curve) > 0


def test_no_warmup_starts_at_peak():
    curve = _curve(total=4, warmup=0)
    assert curve == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        d2z_multiplier(0, total_steps=0, warmup_steps=0)
    with pytest.raises(ValueError):
        d2z_multiplier(0, total_steps=10, warmup_steps=10)
    with pytest.raises(ValueError):
        d2z_scheduler(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0), 10, -1)


def test_scheduler_actually_scales_the_optimizer_updates():
    # With plain SGD and a constant gradient of 1, each update moves the
    # parameter by exactly the current lr, so the parameter trajectory must
    # trace the D2Z curve (times the peak lr).
    peak, total, warmup = 0.5, 20, 4
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=peak)
    sched = d2z_scheduler(opt, total, warmup)

    moves = []
    for _ in range(total):
        before = p.item()
        opt.zero_grad()
        p.sum().backward()  # gradient 1
        opt.step()
        sched.step()
        moves.append(before - p.item())

    assert moves == pytest.approx([peak * m for m in _curve(total, warmup)])
    # After the schedule ends, updates stop entirely.
    assert opt.param_groups[0]["lr"] == 0.0


def test_scheduler_scales_each_param_group_from_its_own_peak():
    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([{"params": [a], "lr": 1.0}, {"params": [b], "lr": 0.1}])
    sched = d2z_scheduler(opt, total_steps=10, warmup_steps=0)
    for _ in range(5):
        opt.step()
        sched.step()
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.5, 0.05])
