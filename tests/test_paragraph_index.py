import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.data.jawiki import Paragraph
from webbot.retrieval import (
    QA,
    BM25Index,
    ParagraphIndex,
    evaluate_retrieval,
    gold_paragraphs,
    index_text,
    load_qa,
)

PARAS = [
    Paragraph(10, 0, "富士山", None, "富士山は日本で最も高い山である。標高は3776メートル。"),
    Paragraph(10, 1, "富士山", "歴史", "富士山は古くから信仰の対象とされてきた。"),
    Paragraph(20, 0, "琵琶湖", None, "琵琶湖は滋賀県にある日本最大の湖である。"),
    Paragraph(30, 0, "日本の山", None, "日本で最も高い山の標高は3776メートルである。"),
    Paragraph(40, 0, "東京", None, "東京は日本の首都である。"),
]


def test_index_text_puts_title_and_section_first():
    assert index_text(PARAS[1]) == "富士山 歴史 富士山は古くから信仰の対象とされてきた。"
    assert index_text(PARAS[0]).startswith("富士山 富士山は")


def test_search_returns_paragraphs_and_survives_save_load(tmp_path):
    index = ParagraphIndex.build(PARAS)
    (top, score), *_ = index.search("琵琶湖はどこにある？", 3)
    assert top == PARAS[2] and score > 0
    index.save(tmp_path / "idx")
    loaded = ParagraphIndex.load(tmp_path / "idx")
    assert loaded.paragraphs == PARAS
    for q in ("富士山の歴史", "日本の首都", "最大の湖"):
        assert loaded.search(q, 5) == index.search(q, 5)


def test_index_and_paragraphs_must_agree():
    with pytest.raises(ValueError):
        ParagraphIndex(BM25Index().build(["a"]), PARAS)


def test_gold_paragraphs_overlap_the_passage_within_the_article():
    by_article = {10: [0, 1], 20: [2], 30: [3], 40: [4]}
    exact = QA("q", "a", 10, PARAS[1].text)
    inside = QA("q", "a", 10, "標高は3776メートル。")  # part of paragraph 0
    spanning = QA("q", "a", 10, PARAS[0].text + PARAS[1].text)  # both paragraphs
    other = QA("q", "a", 30, "標高は3776メートル")  # same words, article 30 only
    assert gold_paragraphs(exact, PARAS, by_article) == {1}
    assert gold_paragraphs(inside, PARAS, by_article) == {0}
    assert gold_paragraphs(spanning, PARAS, by_article) == {0, 1}
    assert gold_paragraphs(other, PARAS, by_article) == {3}


def test_evaluation_counts_by_hand():
    index = ParagraphIndex.build(PARAS)
    qas = [
        # Retrieved first.
        QA("琵琶湖は何県にある？", "滋賀県", 20, PARAS[2].text),
        # Gold is paragraph 0, but paragraph 3 (another article, closer
        # wording, the same answer) ranks first: answer@1 without gold@1.
        QA("日本で最も高い山の標高は？", "3776メートル", 10, PARAS[0].text),
        # Nothing matches.
        QA("ゼロ", "なし", 40, PARAS[4].text),
    ]
    ranked = [[d for d, _ in index.bm25.search(q.query, 20)] for q in qas]
    assert ranked[0][0] == 2 and ranked[1][:2] == [3, 0] and ranked[2] == []
    m = evaluate_retrieval(index, qas, ks=(1, 2))
    assert m["gold@1"] == pytest.approx(1 / 3)
    assert m["gold@2"] == pytest.approx(2 / 3)  # paragraph 0 is in the top 2
    assert m["answer@1"] == pytest.approx(2 / 3)
    assert m["answer@2"] == pytest.approx(2 / 3)


def test_load_qa_filters_by_article(tmp_path):
    rows = [
        {"passage_id": 10, "query": "q1", "answer": "a1", "text": "t1", "title": "x", "url": "u"},
        {"passage_id": 99, "query": "q2", "answer": "a2", "text": "t2", "title": "y", "url": "u"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "qa.parquet")
    assert load_qa([tmp_path / "qa.parquet"], {10}) == [QA("q1", "a1", 10, "t1")]
    assert len(load_qa([tmp_path / "qa.parquet"])) == 2
