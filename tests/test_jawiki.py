import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.data import load_token_file
from webbot.data.jawiki import (
    Paragraph,
    article_text,
    is_validation,
    iter_articles,
    iter_documents,
    iter_paragraphs,
    shard_paths,
)
from webbot.data.prepare import prepare_jawiki
from webbot.tokenizer import BPETokenizer

PARAGRAPH = pa.struct([("paragraph_id", pa.int64()), ("tag", pa.string()),
                       ("text", pa.string()), ("title", pa.string())])
SCHEMA = pa.schema([("id", pa.int64()), ("title", pa.string()),
                    ("paragraphs", pa.list_(PARAGRAPH)), ("is_disambiguation_page", pa.bool_())])


def _p(i, text, section=None):
    return {"paragraph_id": i, "tag": "p", "text": text, "title": section}


def _article(i, title, *texts, disambiguation=False):
    return {"id": i, "title": title, "is_disambiguation_page": disambiguation,
            "paragraphs": [_p(j, t, "概要" if j else None) for j, t in enumerate(texts)]}


def _write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


@pytest.fixture
def shards(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "train-00000.parquet", [
        _article(1, "王様", "王様は国を治める人である。", "王様は城に住む。"),
        _article(2, "王 (曖昧さ回避)", "王は次を指す。", disambiguation=True),
        _article(3, "空の記事"),
        _article(4, "ドラゴン", "  ", "ドラゴンは空を飛ぶ伝説の生き物である。"),
    ])
    _write(raw / "train-00001.parquet", [
        _article(5, "城", "城は王様の住まいである。"),
        _article(6, "物語", "物語は毎晩語られた。", "王様は物語を聞いた。"),
    ])
    return raw


def test_articles_skip_disambiguation_empty_and_blank(shards):
    got = [(i, t, [p["text"] for p in ps]) for i, t, ps in iter_articles(shard_paths(shards))]
    assert got == [
        (1, "王様", ["王様は国を治める人である。", "王様は城に住む。"]),
        (4, "ドラゴン", ["ドラゴンは空を飛ぶ伝説の生き物である。"]),
        (5, "城", ["城は王様の住まいである。"]),
        (6, "物語", ["物語は毎晩語られた。", "王様は物語を聞いた。"]),
    ]


def test_article_text_is_title_then_paragraphs():
    assert article_text("城", [_p(0, "一行目。"), _p(1, "二行目。")]) == "城\n一行目。\n二行目。"


def test_validation_split_is_about_one_percent_and_fixed():
    held = [i for i in range(100_000) if is_validation(i)]
    assert 800 < len(held) < 1200
    assert held == [i for i in range(100_000) if is_validation(i)]
    # Not simply every 100th id.
    assert held[:5] != [0, 100, 200, 300, 400]


def test_splits_are_disjoint_and_cover_all_articles(shards):
    paths = shard_paths(shards)
    train = list(iter_documents(paths, "train", val_every=2))
    val = list(iter_documents(paths, "val", val_every=2))
    every = [article_text(t, ps) for _, t, ps in iter_articles(paths)]
    assert sorted(train + val) == sorted(every)
    assert train and val and not set(train) & set(val)
    with pytest.raises(ValueError):
        list(iter_documents(paths, "test"))


def test_paragraphs_carry_article_and_section(shards):
    pars = list(iter_paragraphs(shard_paths(shards)))
    assert pars[0] == Paragraph(1, 0, "王様", None, "王様は国を治める人である。")
    assert pars[1] == Paragraph(1, 1, "王様", "概要", "王様は城に住む。")
    assert len(pars) == 6
    assert all(p.text.strip() for p in pars)


def test_no_shards_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        shard_paths(tmp_path)


def test_prepare_writes_tokenizer_and_both_splits(shards, tmp_path):
    out = tmp_path / "out"
    stats = prepare_jawiki(shards, out, vocab_size=300, val_every=2, num_workers=1,
                           log=lambda s: None)
    tok = BPETokenizer.load(out / "tokenizer.json")
    # A tiny corpus may run out of merges before the budget.
    assert len(tok) == stats["vocab_size"] <= 300
    paths = shard_paths(shards)
    for split in ("train", "val"):
        ids = load_token_file(out / f"{split}.bin")
        assert len(ids) == stats[f"{split}_tokens"]
        expected = list(iter_documents(paths, split, val_every=2))
        # Decoding gives back every article, each closed by <|endoftext|>.
        assert tok.decode(ids.tolist()) == "".join(d + tok.EOT_TOKEN for d in expected)
        assert stats[f"{split}_chars"] == sum(map(len, expected))


def test_validation_is_the_small_split(shards):
    # With the real ~1% rate, these few articles all land in train.
    paths = shard_paths(shards)
    assert len(list(iter_documents(paths, "train"))) == 4
    assert list(iter_documents(paths, "val")) == []


def test_tokenizer_trains_on_the_first_chars_only(shards, tmp_path):
    # One character's worth means only the first article: fewer merges.
    small = prepare_jawiki(shards, tmp_path / "a", vocab_size=300, val_every=2,
                           tokenizer_chars=1, num_workers=1, log=lambda s: None)
    full = prepare_jawiki(shards, tmp_path / "b", vocab_size=300, val_every=2,
                          num_workers=1, log=lambda s: None)
    assert small["vocab_size"] < full["vocab_size"]
