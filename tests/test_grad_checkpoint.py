import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig, checkpoint
from webbot.train import PretrainConfig, pretrain

VOCAB = 40


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8)).double()


def _x(*shape, seed=1):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def test_checkpoint_matches_plain_gradients():
    # Two inputs (one without grad) and two outputs, one of them unused.
    mlp = _mlp()
    x, scale = _x(4, 8).requires_grad_(), _x(4, 1, seed=2)

    def fn(a, s):
        h = mlp(a) * s
        return h, h.sum(dim=-1)

    ref_mlp = copy.deepcopy(mlp)
    x_ref = x.detach().clone().requires_grad_()
    out_ref = mlp.__class__.__call__(ref_mlp, x_ref) * scale
    out_ref.pow(2).sum().backward()

    out, _ = checkpoint(fn, x, scale)
    torch.testing.assert_close(out, out_ref.detach(), rtol=0, atol=0)
    out.pow(2).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-12, atol=1e-14)
    for p, q in zip(mlp.parameters(), ref_mlp.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-14)


def test_forward_is_recomputed_in_backward():
    mlp = _mlp()
    calls = []

    def fn(a):
        calls.append(torch.is_grad_enabled())
        return (mlp(a),)

    (out,) = checkpoint(fn, _x(4, 8).requires_grad_())
    assert calls == [False]  # the first run records no graph
    out.sum().backward()
    assert calls == [False, True]


def _saved_numel(fn):
    total = 0

    def pack(t):
        nonlocal total
        total += t.numel()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total


def test_only_the_inputs_are_saved():
    mlp = _mlp()
    x = _x(32, 8).requires_grad_()
    assert _saved_numel(lambda: checkpoint(lambda a: (mlp(a),), x)) == x.numel()
    assert _saved_numel(lambda: mlp(x)) > 3 * x.numel()  # without checkpointing


def test_recomputation_reuses_the_forward_autocast():
    # Under bf16 autocast the recomputation must also run in bf16; in fp32 it
    # would give (slightly) different gradients than the plain bf16 graph.
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 8))
    ref = copy.deepcopy(mlp)
    x = torch.randn(16, 8)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        (out,) = checkpoint(lambda a: (mlp(a),), xa)
        out_ref = ref(xb)
    out.float().pow(2).sum().backward()
    out_ref.float().pow(2).sum().backward()
    torch.testing.assert_close(xa.grad, xb.grad, rtol=0, atol=0)
    for p, q in zip(mlp.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


# --- whole model -----------------------------------------------------------

CONFIGS = {
    "dense": {},
    "flash_gqa": {"attn_impl": "flash", "n_kv_heads": 2},
    "moe": {"n_experts": 4, "moe_top_k": 2},
}


def _gpt(n_layers=2, **kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=n_layers,
                         n_heads=4, ffn_hidden_dim=64, **kw))


def _batch(n=3, seed=1):
    seq = torch.randint(0, VOCAB, (n, 25), generator=torch.Generator().manual_seed(seed))
    return seq[:, :-1], seq[:, 1:]


@pytest.mark.parametrize("name", CONFIGS)
def test_model_gradients_match_without_checkpointing(name):
    plain = _gpt(**CONFIGS[name]).double()
    ckpt = copy.deepcopy(plain)
    ckpt.config.grad_checkpoint = True
    x, y = _batch()
    _, loss_p = plain(x, y)
    _, loss_c = ckpt(x, y)
    torch.testing.assert_close(loss_c, loss_p, rtol=1e-12, atol=0)
    loss_p.backward()
    loss_c.backward()
    for (n, p), q in zip(plain.named_parameters(), ckpt.parameters()):
        torch.testing.assert_close(q.grad, p.grad, rtol=1e-10, atol=1e-13, msg=n)
    for bp, bc in zip(plain.blocks, ckpt.blocks):
        if hasattr(bp.ffn, "last_counts"):
            assert torch.equal(bc.ffn.last_counts, bp.ffn.last_counts)


def test_model_saves_only_block_inputs():
    # Each extra checkpointed layer adds exactly its input (batch x seq x
    # d_model) to what is kept for backward; without checkpointing, far more.
    x, y = _batch(4)
    per_layer_input = x.numel() * 32

    def saved(n_layers, ckpt):
        model = _gpt(n_layers=n_layers, grad_checkpoint=ckpt)
        return _saved_numel(lambda: model(x, y))

    assert saved(3, True) - saved(1, True) == 2 * per_layer_input
    assert saved(3, False) - saved(1, False) > 10 * 2 * per_layer_input


def test_no_checkpointing_without_grad():
    model = _gpt(grad_checkpoint=True)
    x, _ = _batch()
    with torch.no_grad():
        logits, caches = model.forward_with_cache(x)
    assert len(caches) == 2  # the normal path, with KV caches
    _, caches = model.forward_with_cache(x)
    assert caches == []
    out = model.generate(x[:, :4], 6, temperature=0)
    ref = _gpt().generate(x[:, :4], 6, temperature=0)
    assert torch.equal(out, ref)


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.resize(rng.integers(0, VOCAB, size=7), n).astype(np.uint16)


def test_pretraining_with_checkpointing_and_accumulation_matches_plain():
    tokens = _tokens(3000)
    train, val = tokens[:2700], tokens[2700:]
    cfg = dict(total_steps=20, batch_size=8, seq_len=16, peak_lr=3e-3, warmup_steps=5,
               eval_interval=10, eval_batches=4, precision="fp32")
    # Dense: with MoE, accumulation changes the balancing loss (per micro-batch).
    plain = _gpt()
    hist_p = pretrain(plain, train, val, PretrainConfig(**cfg), log=lambda s: None)
    ckpt = _gpt(grad_checkpoint=True)
    hist_c = pretrain(ckpt, train, val, PretrainConfig(**cfg, grad_accum_steps=2), log=lambda s: None)
    assert hist_c[-1]["val_loss"] == pytest.approx(hist_p[-1]["val_loss"], rel=1e-3)
    for (n, p), q in zip(plain.named_parameters(), ckpt.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-3, atol=1e-4, msg=n)
