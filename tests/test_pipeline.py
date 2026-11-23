import importlib.util
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from webbot.sft import load_jsonl, split_of

spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

PARAGRAPH = pa.struct([("paragraph_id", pa.int64()), ("tag", pa.string()),
                       ("text", pa.string()), ("title", pa.string())])
SCHEMA = pa.schema([("id", pa.int64()), ("title", pa.string()),
                    ("paragraphs", pa.list_(PARAGRAPH)), ("is_disambiguation_page", pa.bool_())])
COLORS = ["赤", "青", "緑", "黄", "白", "黒", "紫", "茶"]


def _ids():
    """Article ids with several in each SFT split."""
    picked = {"train": [], "val": [], "test": []}
    i = 1000
    while min(len(v) for v in picked.values()) < 4 or len(picked["train"]) < 24:
        s = split_of(i)
        if len(picked[s]) < (24 if s == "train" else 4):
            picked[s].append(i)
        i += 1
    return sorted(sum(picked.values(), []))


def _article(i, n):
    color = COLORS[n % len(COLORS)]
    return {"id": i, "title": f"品{n}号", "is_disambiguation_page": False,
            "paragraphs": [{"paragraph_id": 0, "tag": "p", "title": None,
                            "text": f"品{n}号の色は{color}である。品{n}号は倉庫にある。"}]}


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    jawiki, qa = root / "jawiki", root / "qa"
    jawiki.mkdir()
    qa.mkdir()
    ids = _ids()
    # Shard 0: other articles for pretraining; shard 1: the searched shard.
    pq.write_table(pa.Table.from_pylist([_article(100 + n, 100 + n) for n in range(20)], schema=SCHEMA),
                   jawiki / pipeline.shard_name(0))
    pq.write_table(pa.Table.from_pylist([_article(i, n) for n, i in enumerate(ids)], schema=SCHEMA),
                   jawiki / pipeline.shard_name(1))
    pq.write_table(pa.Table.from_pylist([
        {"passage_id": i, "query": f"品{n}号の色は？", "answer": COLORS[n % len(COLORS)],
         "text": f"品{n}号の色は{COLORS[n % len(COLORS)]}である。", "title": "", "url": ""}
        for n, i in enumerate(ids)]), qa / "qa.parquet")
    return jawiki, qa


def _args(raw, work, scratch, *extra):
    jawiki, qa = raw
    return ["--work", str(work), "--scratch", str(scratch), "--shards", "0-1", "--retrieval-shard", "1",
            "--jawiki-raw", str(jawiki), "--qa-raw", str(qa), "--qa-file", "qa.parquet",
            "--vocab", "300", "--workers", "1", "--device", "cpu",
            "--train-tokens", str(6 * 4 * 32), "--d-model", "32", "--n-layers", "2", "--n-heads", "4",
            "--seq-len", "32", "--max-seq-len", "256", "--batch-size", "4", "--grad-accum", "2", "--warmup", "2", "--eval-interval", "3",
            "--sft-steps", "6", "--sft-batch-size", "4", "--sft-grad-accum", "2", "--sft-warmup", "2",
            "--sft-eval-interval", "3", "--test-examples", "3", *extra]


def test_runs_every_stage_then_skips_them(raw, tmp_path):
    work = tmp_path / "work"
    assert pipeline.main(_args(raw, work, tmp_path / "scratch"))
    assert (work / "data" / "val.bin.json").exists()
    assert (work / "runs" / "base" / "DONE").exists() and (work / "runs" / "base" / "checkpoint.pt").exists()
    for split in ("train", "val", "test"):
        assert load_jsonl(work / "sft" / f"{split}.jsonl")
    sft_log = (work / "runs" / "sft" / "train.log").read_text(encoding="utf-8")
    assert "test (3 questions" in sft_log
    # Downloads and the index stay in scratch, not in the saved outputs.
    assert not (work / "index").exists() and not list(work.rglob("*.parquet"))

    assert pipeline.main(_args(raw, work, tmp_path / "scratch"))  # again: nothing to do
    log = (work / "pipeline.log").read_text(encoding="utf-8").split("all stages done")[1]
    for stage in ("data: already prepared", "pretraining: already done", "SFT data: already made", "SFT: already done"):
        assert stage in log

    # A fresh machine given the previous output continues from it.
    other = tmp_path / "other"
    assert pipeline.main(_args(raw, other, tmp_path / "scratch2", "--resume-from", str(work)))
    assert "SFT: already done" in (other / "pipeline.log").read_text(encoding="utf-8")


def test_deadline_stops_cleanly_and_a_later_run_finishes(raw, tmp_path):
    work = tmp_path / "work"
    # A second is not enough to prepare the data: the stage is cut short.
    assert not pipeline.main(_args(raw, work, tmp_path / "scratch", "--deadline-hours", str(1 / 3600)))
    log = (work / "pipeline.log").read_text(encoding="utf-8")
    assert "data: stopped at the deadline" in log
    assert "unfinished" in log and not (work / "runs" / "sft" / "DONE").exists()
    resumed = tmp_path / "resumed"
    assert pipeline.main(_args(raw, resumed, tmp_path / "scratch", "--resume-from", str(work)))
    assert (resumed / "runs" / "sft" / "DONE").exists()


def test_parse_shards():
    assert pipeline.parse_shards("0-3,7,21") == [0, 1, 2, 3, 7, 21]
    assert pipeline.parse_shards("5") == [5]
