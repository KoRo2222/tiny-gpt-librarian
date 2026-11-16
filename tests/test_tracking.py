import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig
from webbot.train import PretrainConfig, d2z_multiplier, launch, pretrain, rank

VOCAB = 20


def _fake_wandb(record):
    """A stand-in wandb module that records what pretrain asks of it."""

    class Run:
        def __init__(self, **kw):
            record["init"].append(kw)
            self.id = kw.get("id") or "run-from-init"

        def log(self, metrics, step):
            record["log"].append((step, dict(metrics)))

        def finish(self):
            record["finished"] += 1

    module = types.ModuleType("wandb")
    module.init = lambda **kw: Run(**kw)
    return module


def _record():
    return {"init": [], "log": [], "finished": 0}


def _gpt(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=32, d_model=32, n_layers=2, n_heads=4))


def _tokens(n=3000):
    rng = np.random.default_rng(0)
    return np.resize(rng.integers(0, VOCAB, size=7), n).astype(np.uint16)


def _config(**kw):
    base = dict(total_steps=30, batch_size=4, seq_len=16, peak_lr=3e-3, warmup_steps=5,
                eval_interval=10, eval_batches=2, precision="fp32", wandb_project="webbot-test")
    base.update(kw)
    return PretrainConfig(**base)


def _run(record, monkeypatch, **kw):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(record))
    tokens = _tokens()
    return pretrain(_gpt(), tokens[:2700], tokens[2700:], _config(**kw), log=lambda s: None)


def test_logs_config_steps_and_evaluations(monkeypatch):
    record = _record()
    history = _run(record, monkeypatch, wandb_run_name="trial")
    (init,) = record["init"]
    assert init["project"] == "webbot-test" and init["name"] == "trial" and init["mode"] == "offline"
    assert init["config"]["pretrain"]["total_steps"] == 30
    assert init["config"]["model"]["d_model"] == 32
    assert init["config"]["num_params"] == _gpt().num_params()
    assert record["finished"] == 1

    train = {s: m for s, m in record["log"] if "train/loss" in m}
    val = {s: m for s, m in record["log"] if "val/loss" in m}
    assert sorted(train) == list(range(1, 31))
    for step, m in train.items():
        # The lr this update used: D2Z for 0-indexed update step - 1.
        assert m["train/lr"] == pytest.approx(3e-3 * d2z_multiplier(step - 1, 30, 5))
        assert m["train/tokens"] == step * 4 * 16
        assert m["train/tokens_per_sec"] == pytest.approx(4 * 16 / m["train/step_time"])
    assert sorted(val) == [h["step"] for h in history] == [10, 20, 30]
    for h in history:
        assert val[h["step"]]["val/loss"] == h["val_loss"]
        assert val[h["step"]]["val/ppl"] == pytest.approx(np.exp(h["val_loss"]))
        assert train[h["step"]]["train/loss"] == h["train_loss"]
    steps = [s for s, _ in record["log"]]
    assert steps == sorted(steps)  # W&B needs non-decreasing steps


def test_no_project_means_no_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # importing it would fail
    tokens = _tokens()
    pretrain(_gpt(), tokens[:2700], tokens[2700:], _config(wandb_project=None), log=lambda s: None)


def test_resumed_training_continues_the_same_run(monkeypatch, tmp_path):
    class Stop(Exception):
        pass

    def stop_at_10(line):
        if line.startswith("step     10/"):
            raise Stop

    first = _record()
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(first))
    tokens = _tokens()
    path = tmp_path / "ckpt.pt"
    with pytest.raises(Stop):
        pretrain(_gpt(), tokens[:2700], tokens[2700:], _config(), checkpoint_path=path, log=stop_at_10)
    assert first["finished"] == 1  # closed even though training stopped
    assert torch.load(path, weights_only=False)["wandb_run_id"] == "run-from-init"

    second = _record()
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(second))
    pretrain(_gpt(), tokens[:2700], tokens[2700:], _config(), checkpoint_path=path, resume=True,
             log=lambda s: None)
    (init,) = second["init"]
    assert init["id"] == "run-from-init" and init["resume"] == "allow"
    assert min(s for s, _ in second["log"]) == 11


def _dp_worker(out_dir):
    record = _record()
    sys.modules["wandb"] = _fake_wandb(record)
    tokens = _tokens()
    pretrain(_gpt(seed=rank()), tokens[:2700], tokens[2700:], _config(total_steps=10),
             log=lambda s: None)
    Path(out_dir, f"rank{rank()}.json").write_text(
        json.dumps({"init": len(record["init"]), "log": len(record["log"])})
    )


def test_only_rank_zero_tracks(tmp_path):
    launch(_dp_worker, 2, str(tmp_path), threads_per_rank=1)
    r0 = json.loads((tmp_path / "rank0.json").read_text())
    r1 = json.loads((tmp_path / "rank1.json").read_text())
    assert r0["init"] == 1 and r0["log"] > 0
    assert r1 == {"init": 0, "log": 0}


def test_real_wandb_offline_run(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    tokens = _tokens()
    pretrain(_gpt(), tokens[:2700], tokens[2700:],
             _config(total_steps=8, eval_interval=4, wandb_dir=str(tmp_path)), log=lambda s: None)
    runs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(runs) == 1
    (data,) = runs[0].glob("run-*.wandb")
    assert data.stat().st_size > 0
