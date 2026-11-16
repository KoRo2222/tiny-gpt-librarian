import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig
from webbot.train import (
    Muon,
    PretrainConfig,
    launch,
    make_muon_optimizer,
    newton_schulz,
    pretrain,
    rank,
)
from webbot.train.muon import muon_update

VOCAB = 20


def _g(*shape, seed=0):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def _reference_ns(g, steps=5):
    """The iteration acts on singular values only: SVD, normalize by the
    Frobenius norm, apply s -> a s + b s^3 + c s^5 to each, reassemble."""
    a, b, c = 3.4445, -4.7750, 2.0315  # Keller Jordan's quintic coefficients
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    s = s / (g.norm() + 1e-7)
    for _ in range(steps):
        s = a * s + b * s**3 + c * s**5
    return u @ torch.diag(s) @ vh


@pytest.mark.parametrize("shape", [(8, 8), (5, 12), (12, 5), (1, 7)])
def test_newton_schulz_is_the_polynomial_on_singular_values(shape):
    g = _g(*shape)
    torch.testing.assert_close(newton_schulz(g), _reference_ns(g), rtol=1e-9, atol=1e-11)


def test_newton_schulz_pushes_singular_values_near_one():
    for shape in [(32, 128), (128, 32), (64, 256)]:
        g = _g(*shape) * 1e3  # the scale of the input does not matter
        s = torch.linalg.svdvals(newton_schulz(g))
        assert s.min() > 0.6 and s.max() < 1.3, (shape, s.min(), s.max())
        s_in = torch.linalg.svdvals(g)
        assert s_in.max() / s_in.min() > 2  # while the input's were spread out


def test_muon_update_matches_the_formula():
    grad, buf = _g(12, 4, seed=1), _g(12, 4, seed=2)
    m = 0.9
    expected_buf = m * buf + (1 - m) * grad
    nesterov = (1 - m) * grad + m * expected_buf
    # Rows split into q, k, v blocks, each orthogonalized and scaled alone.
    blocks = nesterov.split([4, 4, 4])
    expected = torch.cat([_reference_ns(b) * max(1.0, b.shape[0] / b.shape[1]) ** 0.5 for b in blocks])
    got_buf = buf.clone()
    got = muon_update(grad, got_buf, m, nesterov=True, split=(4, 4, 4))
    torch.testing.assert_close(got_buf, expected_buf)
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-11)
    # Without Nesterov the buffer itself is orthogonalized; a tall matrix is
    # scaled up by sqrt(rows / cols).
    plain = muon_update(grad, buf.clone(), m, nesterov=False)
    torch.testing.assert_close(plain, _reference_ns(expected_buf) * 3**0.5, rtol=1e-9, atol=1e-11)


def _gpt(seed=0, **kw):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, **kw))


def _batch(seed):
    seq = torch.randint(0, VOCAB, (4, 17), generator=torch.Generator().manual_seed(seed))
    return seq[:, :-1], seq[:, 1:]


def test_parameter_groups():
    model = _gpt(n_experts=2, moe_top_k=1)
    opt = make_muon_optimizer(model, 0.02, 1e-3, 0.1, (0.9, 0.95))
    groups = {id(p): g for g in opt.param_groups for p in g["params"]}
    assert len(groups) == len(list(model.parameters()))  # each exactly once
    for name, p in model.named_parameters():
        g = groups[id(p)]
        if name.startswith("blocks.") and p.dim() == 2:
            assert g["use_muon"] and g["lr"] == 0.02, name
            assert g["split"] == ((32, 32, 32) if name.endswith("attn.qkv.weight") else None), name
        else:
            assert not g["use_muon"] and g["lr"] == 1e-3, name
            assert g["weight_decay"] == (0.1 if p.dim() == 2 else 0.0), name


def test_steps_match_reference_updates():
    model = _gpt().double()
    ref = copy.deepcopy(model)
    opt = make_muon_optimizer(model, 0.05, 3e-3, 0.1, (0.9, 0.95), momentum=0.9, muon_weight_decay=0.01)
    ref_params = dict(ref.named_parameters())
    adam_names = [n for n, p in ref.named_parameters() if not (n.startswith("blocks.") and p.dim() == 2)]
    adam = torch.optim.AdamW(
        [{"params": [ref_params[n] for n in adam_names if ref_params[n].dim() == 2], "weight_decay": 0.1},
         {"params": [ref_params[n] for n in adam_names if ref_params[n].dim() < 2], "weight_decay": 0.0}],
        lr=3e-3, betas=(0.9, 0.95), eps=1e-8,
    )
    bufs = {}
    for step in range(3):
        x, y = _batch(step)
        for m in (model, ref):
            m.zero_grad()
            m(x, y)[1].backward()
        opt.step()
        adam.step()
        with torch.no_grad():
            for n, p in ref.named_parameters():
                if n in adam_names:
                    continue
                buf = bufs.setdefault(n, torch.zeros_like(p))
                buf.mul_(0.9).add_(p.grad, alpha=0.1)
                g = 0.1 * p.grad + 0.9 * buf
                blocks = g.split(32) if n.endswith("attn.qkv.weight") else (g,)
                update = torch.cat([_reference_ns(b) * max(1.0, b.shape[0] / b.shape[1]) ** 0.5 for b in blocks])
                p.mul_(1 - 0.05 * 0.01).sub_(0.05 * update)
    for (n, p), q in zip(ref.named_parameters(), model.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-9, atol=1e-11, msg=n)


def test_muon_groups_take_matrices_only():
    with pytest.raises(ValueError):
        Muon([{"params": [torch.nn.Parameter(torch.zeros(3))], "use_muon": True, "lr": 0.1}])


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.resize(rng.integers(0, VOCAB, size=7), n).astype(np.uint16)


def _config(**kw):
    base = dict(total_steps=40, batch_size=8, seq_len=16, peak_lr=3e-3, warmup_steps=5,
                eval_interval=20, eval_batches=4, precision="fp32", optimizer="muon")
    base.update(kw)
    return PretrainConfig(**base)


def test_muon_alone_lowers_the_loss():
    # AdamW's lr 0 freezes the embedding and head: only the Muon-updated
    # hidden matrices learn, and the loss must still fall well below that of
    # the untrained model (a uniform guess, ln 20 ~ 3.0).
    tokens = _tokens(3000)
    train, val = tokens[:2700], tokens[2700:]
    frozen = pretrain(_gpt(), train, val, _config(peak_lr=0.0, muon_lr=0.0), log=lambda s: None)
    muon = pretrain(_gpt(), train, val, _config(peak_lr=0.0), log=lambda s: None)
    assert frozen[-1]["val_loss"] > 2.9
    assert muon[-1]["val_loss"] < frozen[-1]["val_loss"] - 0.4


def test_pretraining_with_muon_learns_the_pattern():
    tokens = _tokens(3000)
    history = pretrain(_gpt(), tokens[:2700], tokens[2700:], _config(peak_lr=1e-2), log=lambda s: None)
    assert history[-1]["val_loss"] < 0.5
    assert history[-1]["val_loss"] < history[0]["val_loss"]


def test_resumed_muon_run_matches_uninterrupted(tmp_path):
    tokens = _tokens(3000)
    train, val = tokens[:2700], tokens[2700:]
    cfg = _config()
    straight = _gpt()
    pretrain(straight, train, val, cfg, log=lambda s: None)

    class Stop(Exception):
        pass

    def stop_at_20(line):
        if line.startswith("step     20/"):
            raise Stop

    path = tmp_path / "ckpt.pt"
    with pytest.raises(Stop):
        pretrain(_gpt(), train, val, cfg, checkpoint_path=path, log=stop_at_20)
    resumed = _gpt(seed=1)
    pretrain(resumed, train, val, cfg, checkpoint_path=path, resume=True, log=lambda s: None)
    for (n, p), q in zip(straight.named_parameters(), resumed.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-6, atol=1e-7, msg=n)


def test_invalid_combinations_rejected():
    tokens = _tokens(2000)
    for bad in (dict(zero_stage=1), dict(optimizer="sgd")):
        with pytest.raises(ValueError):
            pretrain(_gpt(), tokens, tokens, _config(**bad), log=lambda s: None)


def _dp_worker(out_dir):
    tokens = _tokens(3000)
    model = _gpt(seed=rank())
    pretrain(model, tokens[:2700], tokens[2700:], _config(), log=lambda s: None)
    if rank() == 0:
        torch.save(model.state_dict(), Path(out_dir) / "dp.pt")


def test_data_parallel_muon_matches_one_process(tmp_path):
    launch(_dp_worker, 2, str(tmp_path), threads_per_rank=1)
    tokens = _tokens(3000)
    single = _gpt()
    pretrain(single, tokens[:2700], tokens[2700:], _config(), log=lambda s: None)
    got = torch.load(tmp_path / "dp.pt")
    for n, p in single.state_dict().items():
        torch.testing.assert_close(got[n], p, rtol=1e-4, atol=1e-5, msg=n)
