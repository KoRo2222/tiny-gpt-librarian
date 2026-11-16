import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig
from webbot.train import PretrainConfig, pretrain, resolve_precision, train_step

VOCAB = 20
FP32 = resolve_precision("cpu", "fp32")


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, **kw))


def _batch(n=6, seed=1):
    seq = torch.randint(0, VOCAB, (n, 17), generator=torch.Generator().manual_seed(seed))
    return seq[:, :-1], seq[:, 1:]


def _step(model, idx, targets, k, lr=0.0, max_grad_norm=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    scaler = FP32.make_grad_scaler()
    return train_step(model, idx, targets, opt, FP32, scaler, None, max_grad_norm, k)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 6])  # 4 splits 6 rows unevenly: 2, 2, 1, 1
def test_accumulated_gradient_equals_full_batch_gradient(k):
    idx, targets = _batch(6)
    ref = _model().double()
    _, full_loss = ref(idx, targets)
    full_loss.backward()

    model = _model().double()
    loss = _step(model, idx, targets, k)  # lr 0: gradients only
    assert loss == pytest.approx(full_loss.item(), rel=1e-12)
    for (name, p), q in zip(ref.named_parameters(), model.parameters()):
        torch.testing.assert_close(q.grad, p.grad, rtol=1e-10, atol=1e-13, msg=name)


def test_model_only_sees_micro_batches():
    model = _model()
    shapes = []
    model.embed.register_forward_hook(lambda m, inp, out: shapes.append(tuple(inp[0].shape)))
    idx, targets = _batch(8)
    _step(model, idx, targets, 4)
    assert shapes == [(2, 16)] * 4


def test_clipping_applies_once_to_the_accumulated_gradient():
    # A tight clip on the summed gradient differs from clipping each
    # micro-batch's share separately.
    idx, targets = _batch(6)
    ref = _model().double()
    _step(ref, idx, targets, 1, lr=1.0, max_grad_norm=0.05)
    model = _model().double()
    _step(model, idx, targets, 3, lr=1.0, max_grad_norm=0.05)
    for (name, p), q in zip(ref.named_parameters(), model.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-10, atol=1e-12, msg=name)


def test_invalid_accumulation_rejected():
    idx, targets = _batch(4)
    for k in (0, 5):
        with pytest.raises(ValueError):
            _step(_model(), idx, targets, k)


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.resize(rng.integers(0, VOCAB, size=7), n).astype(np.uint16)


def _config(**kw):
    base = dict(total_steps=30, batch_size=8, seq_len=16, peak_lr=3e-3, warmup_steps=5,
                eval_interval=10, eval_batches=4, precision="fp32")
    base.update(kw)
    return PretrainConfig(**base)


def test_pretraining_with_accumulation_matches_one_big_batch():
    tokens = _tokens(3000)
    train, val = tokens[:2700], tokens[2700:]
    runs = {}
    for k in (1, 4, 8):
        model = _model()
        rows = []

        def record(m, inp, out, rows=rows):
            if m.training:  # not the evaluation batches
                rows.append(inp[0].shape[0])

        hook = model.embed.register_forward_hook(record)
        history = pretrain(model, train, val, _config(grad_accum_steps=k), log=lambda s: None)
        hook.remove()
        assert set(rows) == {8 // k} and len(rows) == 30 * k  # really split into k
        runs[k] = (model, history)
    base_model, base_hist = runs[1]
    assert base_hist[-1]["val_loss"] < base_hist[0]["val_loss"]  # it does learn
    for k in (4, 8):
        model, hist = runs[k]
        for a, b in zip(base_hist, hist):
            assert b["train_loss"] == pytest.approx(a["train_loss"], rel=1e-4)
            assert b["val_loss"] == pytest.approx(a["val_loss"], rel=1e-4)
        for (name, p), q in zip(base_model.named_parameters(), model.parameters()):
            torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-5, msg=name)


def test_pretrain_rejects_more_micro_batches_than_rows():
    tokens = _tokens(2000)
    with pytest.raises(ValueError):
        pretrain(_model(), tokens, tokens, _config(grad_accum_steps=9), log=lambda s: None)
