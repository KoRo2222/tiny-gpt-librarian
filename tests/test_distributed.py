import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig
from webbot.train import (
    PretrainConfig,
    all_reduce_gradients,
    broadcast_parameters,
    launch,
    local_rows,
    pretrain,
    rank,
    world_size,
)

VOCAB = 20


def test_local_rows_partition_the_batch():
    for batch in range(1, 12):
        for world in range(1, batch + 1):
            shares = [local_rows(batch, r, world) for r in range(world)]
            assert shares[0][0] == 0 and shares[-1][1] == batch
            assert all(a[1] == b[0] for a, b in zip(shares, shares[1:]))  # contiguous
            sizes = [stop - start for start, stop in shares]
            assert max(sizes) - min(sizes) <= 1
    with pytest.raises(ValueError):
        local_rows(2, 0, 3)


# --- in spawned processes --------------------------------------------------
# Workers are module-level so they can be pickled into the child processes;
# each writes what it saw to out_dir for the parent to check.


def _gpt(seed=0, **kw):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, **kw))


def _batch(n=7):
    seq = torch.randint(0, VOCAB, (n, 17), generator=torch.Generator().manual_seed(1))
    return seq[:, :-1], seq[:, 1:]


class _Partly(nn.Module):
    """`used` gets a gradient everywhere, `rank0_only` only on rank 0,
    `unused` on no rank."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.used, self.rank0_only, self.unused = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)

    def forward(self, x):
        y = self.used(x)
        return self.rank0_only(y) if rank() == 0 else y


def _grad_worker(out_dir):
    r = rank()
    model = _gpt(seed=r).double()  # different init on every rank...
    broadcast_parameters(model)  # ...until rank 0's is broadcast
    x, y = _batch(7)
    start, stop = local_rows(7)
    _, loss = model(x[start:stop], y[start:stop])
    loss.backward()
    all_reduce_gradients(model, (stop - start) / 7)

    partly = _Partly()
    xs = torch.randn(2, 4, generator=torch.Generator().manual_seed(r))
    partly(xs).sum().backward()
    all_reduce_gradients(partly, 0.5)
    torch.save(
        {
            "params": {n: p.detach() for n, p in model.named_parameters()},
            "grads": {n: p.grad for n, p in model.named_parameters()},
            "partly": {n: p.grad for n, p in partly.named_parameters()},
        },
        Path(out_dir) / f"rank{r}.pt",
    )


def test_broadcast_and_gradient_all_reduce(tmp_path):
    world = 3  # 7 rows: 3 + 2 + 2
    launch(_grad_worker, world, str(tmp_path), threads_per_rank=1)
    ref = _gpt(seed=0).double()
    x, y = _batch(7)
    _, loss = ref(x, y)
    loss.backward()
    for r in range(world):
        got = torch.load(tmp_path / f"rank{r}.pt")
        for n, p in ref.named_parameters():
            assert torch.equal(got["params"][n], p.detach()), n
            torch.testing.assert_close(got["grads"][n], p.grad, rtol=1e-10, atol=1e-13, msg=n)
        partly = got["partly"]
        assert partly["unused.weight"] is None and partly["unused.bias"] is None
        assert partly["rank0_only.weight"] is not None
        if r > 0:
            rank0 = torch.load(tmp_path / "rank0.pt")["partly"]
            for n in ("used.weight", "rank0_only.weight"):
                assert torch.equal(partly[n], rank0[n])


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.resize(rng.integers(0, VOCAB, size=7), n).astype(np.uint16)


def _config(**kw):
    base = dict(total_steps=20, batch_size=8, seq_len=16, peak_lr=3e-3, warmup_steps=5,
                eval_interval=10, eval_batches=5, precision="fp32")
    base.update(kw)
    return PretrainConfig(**base)


def _pretrain_worker(out_dir, config, model_kw):
    tokens = _tokens(3000)
    model = _gpt(seed=rank(), **model_kw)  # broadcast makes the inits agree
    rows = {"train": [], "eval": 0}

    def record(m, inp, out):
        if m.training:
            rows["train"].append(inp[0].shape[0])
        else:
            rows["eval"] += inp[0].shape[0]

    model.embed.register_forward_hook(record)
    logs = []
    history = pretrain(model, tokens[:2700], tokens[2700:], config,
                       checkpoint_path=Path(out_dir) / "ckpt.pt", log=logs.append)
    torch.save({"state": model.state_dict(), "history": history, "logs": logs,
                "world": world_size(), "rows": rows}, Path(out_dir) / f"rank{rank()}.pt")


@pytest.mark.parametrize(
    "world,accum,model_kw",
    [(2, 1, {}), (3, 2, {}), (2, 1, {"n_experts": 4, "moe_top_k": 1})],
)
def test_data_parallel_pretraining_matches_one_process(tmp_path, world, accum, model_kw):
    # 8 rows over 3 ranks is 3 + 3 + 2; the local shares are further split
    # into micro-batches. MoE balancing is per rank, so for MoE the batch per
    # rank equals the single-process micro-batch: no accumulation there.
    config = _config(grad_accum_steps=accum)
    launch(_pretrain_worker, world, str(tmp_path), config, model_kw, threads_per_rank=1)

    tokens = _tokens(3000)
    single = _gpt(seed=0, **model_kw)
    single_cfg = _config(grad_accum_steps=world if model_kw else 1)
    history = pretrain(single, tokens[:2700], tokens[2700:], single_cfg, log=lambda s: None)
    assert history[-1]["val_loss"] < history[0]["val_loss"]

    results = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(world)]
    assert all(res["world"] == world for res in results)
    # Each rank computes only its share: its rows of every batch (in its
    # micro-batches) and every world-th validation batch.
    for r, res in enumerate(results):
        start, stop = local_rows(8, r, world)
        per_step = [len(t) for t in torch.arange(stop - start).tensor_split(accum)]
        assert res["rows"]["train"] == per_step * 20
    # Validation windows per evaluation: 299 tokens give 18 windows of 16 (< 5 x 8).
    n_evals, eval_rows = len(history), min((300 - 1) // 16, 5 * 8)
    assert sum(res["rows"]["eval"] for res in results) == n_evals * eval_rows
    for res in results:
        for a, b in zip(res["history"], history):
            assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-4)
            assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-4)
        for n, p in single.state_dict().items():
            torch.testing.assert_close(res["state"][n], p, rtol=1e-4, atol=1e-5, msg=n)
    # Replicas stay bit-identical; only rank 0 logs and writes checkpoints.
    for res in results[1:]:
        for n, p in results[0]["state"].items():
            assert torch.equal(res["state"][n], p), n
        assert res["logs"] == []
    assert any("data parallel over" in line for line in results[0]["logs"])
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    for n, p in results[0]["state"].items():
        assert torch.equal(ckpt["model"][n], p)


def _failing_worker():
    if rank() == 1:
        raise RuntimeError("boom")


def test_an_error_in_any_rank_is_raised():
    with pytest.raises(Exception, match="boom"):
        launch(_failing_worker, 2, threads_per_rank=1)
