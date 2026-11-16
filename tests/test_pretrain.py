import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.data import sequential_batches
from webbot.model import GPT, GPTConfig
from webbot.train import (
    PretrainConfig,
    evaluate,
    load_model,
    make_optimizer,
    pretrain,
    resolve_precision,
    save_checkpoint,
)

VOCAB = 20


def _model(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4))


def _pattern_tokens(n, seed=0):
    # A learnable stream: a fixed random cycle of length 7. A model that
    # learns it can predict every token after the first few.
    rng = np.random.default_rng(seed)
    cycle = rng.integers(0, VOCAB, size=7)
    return np.resize(cycle, n).astype(np.uint16)


def _config(**kw):
    base = dict(
        total_steps=60, batch_size=8, seq_len=16, peak_lr=3e-3, warmup_steps=5,
        eval_interval=20, eval_batches=4, precision="fp32",
    )
    base.update(kw)
    return PretrainConfig(**base)


def test_weight_decay_only_on_matrices():
    model = _model()
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() == 2 for p in decay["params"])
    # Exactly the RMSNorm gains are left undecayed (2 per block + final).
    assert len(no_decay["params"]) == 2 * 2 + 1
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert any(p is model.embed.weight for p in decay["params"])
    assert any(p is model.lm_head.weight for p in decay["params"])


def test_evaluate_is_mean_loss_without_touching_grads_or_mode():
    model = _model()
    tokens = _pattern_tokens(500)
    batches = list(sequential_batches(tokens, 16, 4, 3))
    expected = np.mean([model(x, y)[1].item() for x, y in batches])
    model.train()
    loss = evaluate(model, batches, resolve_precision("cpu"))
    assert loss == pytest.approx(expected, rel=1e-6)
    assert model.training
    assert all(p.grad is None for p in model.parameters())


def test_pretraining_learns_the_pattern(tmp_path):
    tokens = _pattern_tokens(4000)
    train, val = tokens[:3600], tokens[3600:]
    model = _model()
    logs = []
    history = pretrain(model, train, val, _config(), checkpoint_path=tmp_path / "c.pt", log=logs.append)

    assert [h["step"] for h in history] == [20, 40, 60]
    # Starts near a uniform guess (ln 20 ~ 3.0) and must end far below it.
    assert history[-1]["val_loss"] < 0.5
    assert history[-1]["val_loss"] < history[0]["val_loss"]
    assert any("val" in line for line in logs)


def test_checkpoint_restores_identical_model(tmp_path):
    model = _model()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model)
    restored = load_model(path)
    x = torch.randint(0, VOCAB, (2, 10))
    torch.testing.assert_close(restored(x)[0], model(x)[0])
    assert not (tmp_path / "m.pt.tmp").exists()


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    # Stop half way, rebuild everything from the checkpoint, finish: the
    # weights must equal those of a run that never stopped (same batches,
    # same optimizer moments, same point on the LR schedule).
    tokens = _pattern_tokens(3000)
    train, val = tokens[:2700], tokens[2700:]
    cfg = _config(total_steps=40, eval_interval=20)

    straight = _model()
    pretrain(straight, train, val, cfg, log=lambda s: None)

    path = tmp_path / "ckpt.pt"
    # Same 40-step schedule, interrupted right after the step-20 checkpoint.
    _run_until(_model(), train, val, cfg, path, stop_at=20)

    resumed = _model(seed=123)  # deliberately different init: must be overwritten
    pretrain(resumed, train, val, cfg, checkpoint_path=path, resume=True, log=lambda s: None)

    for a, b in zip(straight.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def _run_until(model, train, val, cfg, path, stop_at):
    """Run pretrain but abort right after the checkpoint at step stop_at."""

    class Stop(Exception):
        pass

    def log(line):
        if line.startswith(f"step {stop_at:>6}/"):
            raise Stop

    with pytest.raises(Stop):
        pretrain(model, train, val, cfg, checkpoint_path=path, log=log)


def test_lr_reaches_zero_at_the_end(tmp_path):
    tokens = _pattern_tokens(2000)
    path = tmp_path / "c.pt"
    pretrain(_model(), tokens[:1800], tokens[1800:], _config(total_steps=10, eval_interval=10),
             checkpoint_path=path, log=lambda s: None)
    sched = torch.load(path, weights_only=False)["scheduler"]
    assert sched["_last_lr"] == [0.0, 0.0]


def test_seq_len_longer_than_model_rejected():
    tokens = _pattern_tokens(2000)
    with pytest.raises(ValueError):
        pretrain(_model(), tokens, tokens, _config(seq_len=64), log=lambda s: None)
