import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig
from webbot.train import (
    PretrainConfig,
    ZeroOptimizer,
    launch,
    local_rows,
    make_optimizer,
    pretrain,
    rank,
    resolve_precision,
    train_step,
    world_size,
)

VOCAB = 20
FP32 = resolve_precision("cpu", "fp32")
HP = dict(lr=3e-2, weight_decay=0.1, betas=(0.9, 0.95))


def _gpt(seed=0, **kw):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, **kw))


def _batch(n=6, seed=1):
    seq = torch.randint(0, VOCAB, (n, 17), generator=torch.Generator().manual_seed(seed))
    return seq[:, :-1], seq[:, 1:]


def _steps(model, opt, n, start=0, sync=None, clip=0.5, rows=None):
    for i in range(start, start + n):
        x, y = _batch(seed=i)
        if rows is not None:
            x, y = x[rows[0] : rows[1]], y[rows[0] : rows[1]]
        train_step(model, x, y, opt, FP32, FP32.make_grad_scaler(), None, clip, 1, sync)


@pytest.mark.parametrize("stage", [1, 2])
def test_single_process_matches_adamw(stage):
    ref = _gpt().double()
    _steps(ref, make_optimizer(ref, **HP), 4)
    model = _gpt().double()
    zero = ZeroOptimizer(model, **HP, stage=stage)
    _steps(model, zero, 4, sync=zero.reduce_gradients)
    for (n, p), q in zip(ref.named_parameters(), model.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-10, atol=1e-12, msg=n)


def test_state_dict_round_trip_continues_identically():
    model = _gpt().double()
    zero = ZeroOptimizer(model, **HP, stage=2)
    _steps(model, zero, 3, sync=zero.reduce_gradients)
    state = copy.deepcopy(zero.state_dict())
    resumed = copy.deepcopy(model)
    zero2 = ZeroOptimizer(resumed, **HP, stage=1)
    zero2.load_state_dict(state)
    _steps(model, zero, 3, start=3, sync=zero.reduce_gradients)
    _steps(resumed, zero2, 3, start=3, sync=zero2.reduce_gradients)
    for (n, p), q in zip(model.named_parameters(), resumed.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-12, atol=1e-14, msg=n)


def test_invalid_stage_rejected():
    with pytest.raises(ValueError):
        ZeroOptimizer(_gpt(), **HP, stage=3)


# --- in spawned processes --------------------------------------------------


def _shard_worker(out_dir, stage):
    model = _gpt().double()
    zero = ZeroOptimizer(model, **HP, stage=stage)
    rows = local_rows(6)
    share = (rows[1] - rows[0]) / 6
    _steps(model, zero, 2, sync=lambda: zero.reduce_gradients(share), rows=rows)
    torch.save({
        "params": {n: p.detach() for n, p in model.named_parameters()},
        "state_numel": zero.state_numel(),
        "model_grad_numel": sum(p.grad.numel() for p in model.parameters() if p.grad is not None),
        "shard_grad_numel": sum(sp.grad.numel() for sp in zero.shard_params),
    }, Path(out_dir) / f"rank{rank()}.pt")


@pytest.mark.parametrize("stage", [1, 2])
def test_shards_hold_a_world_th_of_the_state(tmp_path, stage):
    world = 3  # 6 rows: 2 each
    launch(_shard_worker, world, str(tmp_path), stage, threads_per_rank=1)
    ref = _gpt().double()
    _steps(ref, make_optimizer(ref, **HP), 2)
    total = ref.num_params()
    results = [torch.load(tmp_path / f"rank{r}.pt") for r in range(world)]
    # Two moments per owned element; the shards together cover every element once.
    assert sum(res["state_numel"] for res in results) == 2 * total
    assert all(res["state_numel"] <= 2 * -(-total // world) for res in results)
    assert sum(res["shard_grad_numel"] for res in results) == total
    for res in results:
        # Stage 1 keeps the full (reduced) gradient; stage 2 releases it.
        assert res["model_grad_numel"] == (total if stage == 1 else 0)
        for n, p in ref.named_parameters():
            torch.testing.assert_close(res["params"][n], p.detach(), rtol=1e-10, atol=1e-12, msg=n)


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.resize(rng.integers(0, VOCAB, size=7), n).astype(np.uint16)


def _config(**kw):
    base = dict(total_steps=20, batch_size=8, seq_len=16, peak_lr=3e-3, warmup_steps=5,
                eval_interval=10, eval_batches=4, precision="fp32")
    base.update(kw)
    return PretrainConfig(**base)


def _single_run(path=None):
    tokens = _tokens(3000)
    model = _gpt()
    history = pretrain(model, tokens[:2700], tokens[2700:], _config(), checkpoint_path=path,
                       log=lambda s: None)
    return model, history


def _pretrain_worker(out_dir, config, resume_from, copy_at):
    tokens = _tokens(3000)
    model = _gpt(seed=rank())
    path = Path(out_dir) / "ckpt.pt"
    if resume_from is not None and rank() == 0:
        shutil.copy(resume_from, path)
    torch.distributed.barrier()

    def log(line):
        if copy_at is not None and line.startswith(f"step {copy_at:>6}/"):
            shutil.copy(path, Path(out_dir) / f"ckpt_step{copy_at}.pt")

    history = pretrain(model, tokens[:2700], tokens[2700:], config, checkpoint_path=path,
                       resume=resume_from is not None, log=log)
    torch.save({"state": model.state_dict(), "history": history, "world": world_size()},
               Path(out_dir) / f"rank{rank()}.pt")


def _assert_matches(results, model, history):
    for res in results:
        for a, b in zip(res["history"], history):
            assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-4)
        for n, p in model.state_dict().items():
            torch.testing.assert_close(res["state"][n], p, rtol=1e-4, atol=1e-5, msg=n)


@pytest.mark.parametrize("world,stage,accum", [(2, 1, 1), (3, 2, 2)])
def test_zero_pretraining_matches_one_process(tmp_path, world, stage, accum):
    cfg = _config(zero_stage=stage, grad_accum_steps=accum)
    launch(_pretrain_worker, world, str(tmp_path), cfg, None, None, threads_per_rank=1)
    single, history = _single_run(tmp_path / "single.pt")
    results = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(world)]
    _assert_matches(results, single, history)
    # The checkpoint holds the gathered optimizer state: the same moments as
    # the single process's per-parameter AdamW, in the flat layout.
    zero_state = torch.load(tmp_path / "ckpt.pt", weights_only=False)["optimizer"]
    adam = torch.load(tmp_path / "single.pt", weights_only=False)["optimizer"]
    decay = [p for p in single.parameters() if p.dim() >= 2]
    no_decay = [p for p in single.parameters() if p.dim() < 2]
    ids = adam["param_groups"][0]["params"] + adam["param_groups"][1]["params"]
    assert len(ids) == len(decay) + len(no_decay)
    for key in ("exp_avg", "exp_avg_sq"):
        flat = torch.cat([adam["state"][i][key].flatten() for i in ids])
        torch.testing.assert_close(zero_state[key], flat, rtol=1e-3, atol=1e-7)
    assert zero_state["step"] == 20


def test_resume_with_a_different_number_of_processes(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    # Two processes with ZeRO-2 until the step-10 checkpoint...
    launch(_pretrain_worker, 2, str(first), _config(zero_stage=2), None, 10, threads_per_rank=1)
    # ...then three processes with ZeRO-1 finish the same 20-step run.
    launch(_pretrain_worker, 3, str(second), _config(zero_stage=1),
           str(first / "ckpt_step10.pt"), None, threads_per_rank=1)
    single, history = _single_run()
    results = [torch.load(second / f"rank{r}.pt", weights_only=False) for r in range(3)]
    _assert_matches(results, single, history)


def test_fp16_rejected():
    tokens = _tokens(2000)
    with pytest.raises(ValueError):
        pretrain(_gpt(), tokens, tokens, _config(zero_stage=1, precision="fp16"), log=lambda s: None)
