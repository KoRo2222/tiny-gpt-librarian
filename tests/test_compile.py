import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from webbot.model import GPT, GPTConfig, compile_model, setup_compiler

try:
    setup_compiler()
except RuntimeError as e:  # no C++ compiler on this machine
    pytest.skip(f"torch.compile unavailable: {e}", allow_module_level=True)

CONFIGS = {
    "dense": {},
    "flash_gqa": {"attn_impl": "flash", "n_kv_heads": 2},
    "moe_top1": {"n_experts": 4, "moe_top_k": 1},
    "moe_top2_gqa": {"n_experts": 4, "moe_top_k": 2, "n_kv_heads": 1},
    # Gradient checkpointing switches to torch.utils.checkpoint when compiled.
    "ckpt_dense": {"grad_checkpoint": True},
    "ckpt_moe": {"grad_checkpoint": True, "n_experts": 4, "moe_top_k": 2},
}


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    # Each test compiles new models; start from empty caches so no test
    # hits the recompile limit (which would silently fall back to eager).
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()


def _gpt(**kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, max_seq_len=32, d_model=32, n_layers=2,
                         n_heads=4, ffn_hidden_dim=64, **kw))


def _batch(seed=1):
    idx = torch.randint(0, 64, (2, 25), generator=torch.Generator().manual_seed(seed))
    return idx[:, :-1], idx[:, 1:]


@pytest.mark.parametrize("name", CONFIGS)
def test_training_forward_is_one_graph(name):
    x, y = _batch()
    explained = torch._dynamo.explain(_gpt(**CONFIGS[name]))(x, y)
    assert explained.graph_break_count == 0, explained.break_reasons
    assert explained.graph_count == 1


def _train(model, step_fn, steps=3):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for i in range(steps):
        x, y = _batch(seed=i)
        _, loss = step_fn(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("name", ["dense", "moe_top2_gqa", "flash_gqa", "ckpt_dense", "ckpt_moe"])
def test_compiled_training_matches_eager(name):
    eager = _gpt(**CONFIGS[name])
    compiled = copy.deepcopy(eager)
    compile_model(compiled)
    want = _train(eager, eager)
    got = _train(compiled, compiled)
    # Really compiled: Inductor produced a graph and nothing broke it.
    assert counters["stats"]["unique_graphs"] >= 1
    assert not counters["graph_break"]
    assert got == pytest.approx(want, rel=1e-5)
    for (n, p), q in zip(eager.named_parameters(), compiled.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-5, msg=n)


def test_compile_keeps_state_dict_eval_and_generation():
    eager = _gpt(n_experts=4, moe_top_k=2)
    compiled = compile_model(copy.deepcopy(eager))
    assert list(compiled.state_dict()) == list(eager.state_dict())
    eager.eval()
    compiled.eval()
    x, y = _batch()
    _, want = eager(x, y)
    _, got = compiled(x, y)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)  # eval: no aux loss
    prompt = x[:, :5]
    assert torch.equal(
        compiled.generate(prompt, 10, temperature=0), eager.generate(prompt, 10, temperature=0)
    )


@pytest.mark.skipif(sys.platform != "win32", reason="MSVC setup is Windows-only")
def test_setup_finds_msvc():
    import shutil

    setup_compiler()
    assert shutil.which("cl")
