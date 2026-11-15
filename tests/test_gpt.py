import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, CausalSelfAttention, GPTConfig, RotaryEmbedding

VOCAB = 50


def _tiny(**overrides):
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    return GPT(config)


# --- attention -------------------------------------------------------------


def test_attention_matches_pytorch_reference():
    # Our hand-written masked softmax attention vs PyTorch's fused kernel,
    # fed the same RoPE-rotated q/k.
    torch.manual_seed(0)
    rope = RotaryEmbedding(8, max_seq_len=16)
    attn = CausalSelfAttention(d_model=32, n_heads=4, rope=rope)
    x = torch.randn(2, 10, 32)

    q, k, v = (
        t.view(2, 10, 4, 8).transpose(1, 2) for t in attn.qkv(x).split(32, dim=-1)
    )
    ref = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
    expected = attn.out(ref.transpose(1, 2).reshape(2, 10, 32))

    out, _ = attn(x)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_attention_kv_cache_matches_full_pass():
    torch.manual_seed(0)
    attn = CausalSelfAttention(32, 4, RotaryEmbedding(8, max_seq_len=16))
    x = torch.randn(2, 10, 32)
    full, _ = attn(x)
    # Prefill 6 tokens, then feed the rest one at a time.
    out, cache = attn(x[:, :6])
    outs = [out]
    for t in range(6, 10):
        out, cache = attn(x[:, t : t + 1], cache)
        outs.append(out)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-5, rtol=1e-5)
    assert cache[0].shape == (2, 4, 10, 8)


# --- whole model -----------------------------------------------------------


def test_logits_and_loss_shapes_and_definition():
    model = _tiny()
    idx = torch.randint(0, VOCAB, (3, 12))
    targets = torch.randint(0, VOCAB, (3, 12))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 12, VOCAB)
    torch.testing.assert_close(
        loss, F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
    )
    assert model(idx)[1] is None


def test_initial_loss_is_near_uniform_guess():
    # With small init the model should start out not knowing anything:
    # loss ~ ln(vocab_size). A much larger value would mean bad init.
    model = _tiny()
    idx = torch.randint(0, VOCAB, (8, 32))
    _, loss = model(idx, torch.randint(0, VOCAB, (8, 32)))
    assert abs(loss.item() - math.log(VOCAB)) < 0.1


def test_future_tokens_do_not_affect_past_logits():
    model = _tiny()
    idx = torch.randint(0, VOCAB, (1, 16))
    changed = idx.clone()
    changed[0, 10] = (idx[0, 10] + 1) % VOCAB
    a, _ = model(idx)
    b, _ = model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_embedding_and_lm_head_are_not_tied():
    model = _tiny()
    assert model.embed.weight is not model.lm_head.weight
    assert model.embed.weight.data_ptr() != model.lm_head.weight.data_ptr()
    # Training the head must leave the embedding untouched.
    before = model.embed.weight.detach().clone()
    opt = torch.optim.SGD(model.lm_head.parameters(), lr=1.0)
    _, loss = model(torch.randint(0, VOCAB, (2, 8)), torch.randint(0, VOCAB, (2, 8)))
    loss.backward()
    opt.step()
    torch.testing.assert_close(model.embed.weight.detach(), before)
    # Both matrices are counted as parameters.
    assert model.num_params() == sum(p.numel() for p in model.parameters())
    assert model.num_params() > 2 * VOCAB * 32


def test_no_dropout_no_bias_no_position_table():
    model = _tiny()
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert not any(
        isinstance(m, torch.nn.Linear) and m.bias is not None for m in model.modules()
    )
    # The only Embedding is the token table; position comes from RoPE.
    assert [m for m in model.modules() if isinstance(m, torch.nn.Embedding)] == [
        model.embed
    ]


def test_model_kv_cache_matches_full_pass():
    model = _tiny()
    idx = torch.randint(0, VOCAB, (2, 20))
    full, _ = model(idx)
    logits, caches = model.forward_with_cache(idx[:, :5])
    steps = [logits]
    for t in range(5, 20):
        logits, caches = model.forward_with_cache(idx[:, t : t + 1], caches)
        steps.append(logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)


def test_sequence_longer_than_max_rejected():
    model = _tiny()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_training_memorizes_a_sequence():
    # Gradients must point the right way through every layer: a few hundred
    # Adam steps on one fixed sequence should drive the loss near zero.
    model = _tiny()
    seq = torch.randint(0, VOCAB, (1, 25))
    idx, targets = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        _, loss = model(idx, targets)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


# --- generation ------------------------------------------------------------


def test_greedy_generate_matches_argmax_without_cache():
    model = _tiny()
    prompt = torch.randint(0, VOCAB, (2, 4))
    out = model.generate(prompt, max_new_tokens=10, temperature=0)

    expected = prompt
    for _ in range(10):
        logits, _ = model(expected)
        expected = torch.cat((expected, logits[:, -1:].argmax(-1)), dim=1)
    assert torch.equal(out, expected)


def test_generate_stops_at_eot_and_pads():
    model = _tiny()
    prompt = torch.randint(0, VOCAB, (1, 4))
    first = model.generate(prompt, max_new_tokens=1, temperature=0)[0, -1].item()
    # Declare the first greedy token to be EOT: generation stops right there.
    out = model.generate(prompt, max_new_tokens=10, temperature=0, eot_id=first)
    assert out.shape == (1, 5) and out[0, -1].item() == first


def test_generate_respects_max_seq_len():
    model = _tiny()
    out = model.generate(torch.zeros(1, 30, dtype=torch.long), max_new_tokens=10)
    assert out.shape == (1, 32)


def test_top_k_1_equals_greedy():
    model = _tiny()
    prompt = torch.randint(0, VOCAB, (2, 4))
    torch.manual_seed(123)
    sampled = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=1)
    greedy = model.generate(prompt, max_new_tokens=8, temperature=0)
    assert torch.equal(sampled, greedy)
