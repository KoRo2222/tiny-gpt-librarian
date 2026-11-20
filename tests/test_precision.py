import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig, RotaryEmbedding
from webbot.train import d2z_scheduler, resolve_precision, train_step

VOCAB = 50


def _tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4))


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, VOCAB, (2, 17), generator=g)
    return seq[:, :-1], seq[:, 1:]


# --- choosing a precision --------------------------------------------------


def test_resolve_precision():
    assert resolve_precision("cpu").compute_dtype is None  # auto -> fp32 on CPU
    assert resolve_precision("cpu", "bf16").compute_dtype == torch.bfloat16
    assert resolve_precision("cpu", "fp16").name == "fp16"
    with pytest.raises(ValueError):
        resolve_precision("cpu", "fp8")


def test_auto_on_gpu_picks_bf16_only_with_bf16_hardware(monkeypatch):
    # A T4 (compute capability 7.5) can emulate bf16 and is_bf16_supported()
    # says True, but only fp16 runs on its tensor cores.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    for capability, expected in [((7, 5), "fp16"), ((7, 0), "fp16"), ((8, 0), "bf16"), ((9, 0), "bf16")]:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        assert resolve_precision("cuda").name == expected, capability


def test_loss_scaling_only_for_fp16():
    for name, enabled in (("fp32", False), ("bf16", False), ("fp16", True)):
        assert resolve_precision("cpu", name).make_grad_scaler().is_enabled() == enabled


# --- mixed precision inside the model --------------------------------------


def test_autocast_runs_matmuls_in_bf16_but_keeps_fp32_where_it_matters():
    model = _tiny()
    seen = {}

    def record_dtype(module, inputs, output):
        seen["qkv"] = output.dtype  # returns None: output left unchanged

    model.blocks[0].attn.qkv.register_forward_hook(record_dtype)
    idx, targets = _batch()
    with resolve_precision("cpu", "bf16").autocast():
        logits, loss = model(idx, targets)
    assert seen["qkv"] == torch.bfloat16  # linear layers really are bf16
    assert logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32  # cross-entropy computed in fp32


def test_rope_rotates_half_precision_input_in_fp32():
    # bf16 input must give exactly the fp32 rotation, rounded once at the
    # end -- not a rotation done with bf16 cos/sin tables.
    torch.manual_seed(0)
    rope = RotaryEmbedding(64, max_seq_len=4096)
    x = torch.randn(1, 2, 8, 64).bfloat16()
    for offset in (0, 1000, 4000):
        expected = rope(x.float(), offset=offset).bfloat16()
        assert torch.equal(rope(x, offset=offset), expected)


def test_bf16_forward_stays_close_to_fp32():
    model = _tiny()
    idx, targets = _batch()
    _, loss32 = model(idx, targets)
    with resolve_precision("cpu", "bf16").autocast():
        _, loss16 = model(idx, targets)
    assert abs(loss16.item() - loss32.item()) < 0.02


# --- training step ---------------------------------------------------------


def test_bf16_step_keeps_fp32_master_weights_and_optimizer_state():
    model = _tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = resolve_precision("cpu", "bf16")
    train_step(model, *_batch(), opt, precision, precision.make_grad_scaler())
    for p in model.parameters():
        assert p.dtype == torch.float32
        assert p.grad.dtype == torch.float32
        assert opt.state[p]["exp_avg"].dtype == torch.float32


def test_bf16_training_learns_like_fp32():
    # Memorizing one sequence must work in bf16 too, ending near the fp32 run.
    final = {}
    for name in ("fp32", "bf16"):
        model = _tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        precision = resolve_precision("cpu", name)
        scaler = precision.make_grad_scaler()
        idx, targets = _batch()
        losses = [train_step(model, idx, targets, opt, precision, scaler) for _ in range(150)]
        final[name] = losses[-1]
        assert losses[-1] < 0.1 * losses[0]
    assert abs(final["bf16"] - final["fp32"]) < 0.1


def test_fp16_scaled_step_with_clipping_matches_fp32_update():
    # The fp16 loss is multiplied by the scaler's large factor before
    # backward; gradients must be unscaled before clipping, otherwise the
    # clip would act on the scaled norm and the update would shrink by
    # orders of magnitude.
    updates = {}
    for name in ("fp32", "fp16"):
        model = _tiny()
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        precision = resolve_precision("cpu", name)
        scaler = precision.make_grad_scaler()
        train_step(model, *_batch(), opt, precision, scaler, max_grad_norm=0.1)
        updates[name] = torch.cat(
            [(p.detach() - b).flatten() for p, b in zip(model.parameters(), before)]
        )
        if name == "fp16":
            assert scaler.get_scale() > 1  # the loss really was scaled
    # Clipped to norm 0.1 with lr 1 -> the update itself has norm 0.1.
    assert updates["fp32"].norm().item() == pytest.approx(0.1, rel=1e-3)
    assert updates["fp16"].norm().item() == pytest.approx(0.1, rel=1e-2)
    cos = torch.nn.functional.cosine_similarity(updates["fp16"], updates["fp32"], dim=0)
    assert cos > 0.99


def test_train_step_advances_the_lr_schedule():
    model = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = d2z_scheduler(opt, total_steps=4, warmup_steps=0)
    precision = resolve_precision("cpu")
    for expected in (0.75, 0.5):
        train_step(model, *_batch(), opt, precision, precision.make_grad_scaler(), sched)
        assert opt.param_groups[0]["lr"] == pytest.approx(expected)


def test_attention_softmax_computed_in_fp32_for_bf16_scores():
    from webbot.model.attention import attention_weights

    torch.manual_seed(0)
    scores = (torch.randn(2, 4, 16, 16) * 4).bfloat16()
    with resolve_precision("cpu", "bf16").autocast():
        weights = attention_weights(scores)
    assert weights.dtype == torch.float32
    torch.testing.assert_close(weights, scores.float().softmax(dim=-1))
    # bf16 softmax would be visibly off; make sure the test can tell.
    assert not torch.allclose(scores.softmax(dim=-1).float(), weights, atol=1e-4, rtol=0)
