import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from webbot.data.prepare import prepare_jawiki

spec = importlib.util.spec_from_file_location("pretrain_script", ROOT / "scripts" / "pretrain.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

PARAGRAPH = pa.struct([("paragraph_id", pa.int64()), ("tag", pa.string()),
                       ("text", pa.string()), ("title", pa.string())])
SCHEMA = pa.schema([("id", pa.int64()), ("title", pa.string()),
                    ("paragraphs", pa.list_(PARAGRAPH)), ("is_disambiguation_page", pa.bool_())])
TEXTS = ["王様は国を治める人である。王様は城に住む。", "ドラゴンは空を飛ぶ伝説の生き物である。",
         "城は王様の住まいである。物語は毎晩語られた。", "富士山は日本で最も高い山である。"]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("pre")
    raw = root / "raw"
    raw.mkdir()
    rows = [{"id": i, "title": f"記事{i}", "is_disambiguation_page": False,
             "paragraphs": [{"paragraph_id": 0, "tag": "p", "text": TEXTS[i % 4] * 3, "title": None}]}
            for i in range(40)]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), raw / "s.parquet")
    prepare_jawiki(raw, root / "data", vocab_size=300, val_every=4, num_workers=1, log=lambda s: None)
    return root / "data"


def _args(data, out, *extra):
    return ["--data", str(data), "--out", str(out), "--steps", "6", "--batch-size", "4",
            "--seq-len", "16", "--max-seq-len", "32", "--d-model", "32", "--n-layers", "2",
            "--n-heads", "4", "--warmup", "2", "--eval-interval", "3", "--eval-batches", "2",
            "--device", "cpu", "--sample-tokens", "5", *extra]


def test_fresh_run_writes_checkpoint_log_and_samples(data, tmp_path):
    history = script.main(_args(data, tmp_path / "run"))
    assert [h["step"] for h in history] == [3, 6]
    ckpt = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
    assert ckpt["step"] == 6 and ckpt["model_config"]["d_model"] == 32
    assert ckpt["model_config"]["attn_impl"] == "sdpa"
    log = (tmp_path / "run" / "train.log").read_text(encoding="utf-8")
    assert "step      6/6" in log and log.count("sample: ") == len(script.PROMPTS)


def test_resume_without_a_checkpoint_starts_fresh(data, tmp_path):
    history = script.main(_args(data, tmp_path / "run", "--resume"))
    assert [h["step"] for h in history] == [3, 6]


def test_resume_takes_the_model_from_the_checkpoint(data, tmp_path):
    out = tmp_path / "run"
    first = script.main(_args(data, out, "--steps", "3", "--warmup", "1"))
    # Different --d-model on the command line: the checkpoint's model wins,
    # and the finished 3-step run has nothing left to do.
    again = script.main(_args(data, out, "--steps", "3", "--warmup", "1", "--resume", "--d-model", "64"))
    assert again == first
    log = (out / "train.log").read_text(encoding="utf-8")
    assert "resumed from" in log and "'d_model': 64" not in log
