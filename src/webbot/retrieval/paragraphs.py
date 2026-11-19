"""Paragraph search over the Wikipedia dump, and its evaluation with the
auto-wiki-qa questions (generated from the same dump)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from webbot.data.jawiki import Paragraph

from .bm25 import Analyzer, BM25Index, char_bigrams


def index_text(p: Paragraph) -> str:
    """What is indexed: the article title and section heading (questions
    often name the article's subject) followed by the paragraph."""
    return " ".join(x for x in (p.title, p.section, p.text) if x)


class ParagraphIndex:
    """BM25 over paragraphs; search returns the paragraphs themselves."""

    def __init__(self, bm25: BM25Index, paragraphs: list[Paragraph]):
        if bm25.num_docs != len(paragraphs):
            raise ValueError("index and paragraphs disagree in number")
        self.bm25 = bm25
        self.paragraphs = paragraphs

    @classmethod
    def build(
        cls, paragraphs: Iterable[Paragraph], analyzer: Analyzer = char_bigrams
    ) -> "ParagraphIndex":
        paragraphs = list(paragraphs)
        return cls(BM25Index(analyzer).build(index_text(p) for p in paragraphs), paragraphs)

    def search(self, query: str, k: int = 5) -> list[tuple[Paragraph, float]]:
        return [(self.paragraphs[d], s) for d, s in self.bm25.search(query, k)]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bm25.save(directory / "bm25")
        pq.write_table(pa.Table.from_pylist([asdict(p) for p in self.paragraphs]),
                       directory / "paragraphs.parquet")

    @classmethod
    def load(cls, directory: str | Path, analyzer: Analyzer = char_bigrams) -> "ParagraphIndex":
        directory = Path(directory)
        rows = pq.read_table(directory / "paragraphs.parquet").to_pylist()
        return cls(BM25Index.load(directory / "bm25", analyzer), [Paragraph(**r) for r in rows])


@dataclass(frozen=True)
class QA:
    query: str
    answer: str
    article_id: int
    passage: str  # the text the question was generated from


def load_qa(paths: Sequence[str | Path], article_ids: set[int] | None = None) -> list[QA]:
    """auto-wiki-qa rows (optionally only about the given articles)."""
    out = []
    for path in paths:
        table = pq.read_table(path, columns=["passage_id", "query", "answer", "text"])
        for pid, query, answer, text in zip(*(table.column(c).to_pylist() for c in table.column_names)):
            if article_ids is None or pid in article_ids:
                out.append(QA(query, answer, pid, text))
    return out


def gold_paragraphs(qa: QA, paragraphs: Sequence[Paragraph], by_article: dict[int, list[int]]) -> set[int]:
    """Indices of the article's paragraphs the question was generated from:
    those overlapping its passage (equal to it, inside it, or containing it)."""
    return {
        i
        for i in by_article.get(qa.article_id, [])
        if qa.passage in paragraphs[i].text or paragraphs[i].text in qa.passage
    }


def evaluate_retrieval(
    index: ParagraphIndex, qas: Sequence[QA], ks: Sequence[int] = (1, 5, 10, 20)
) -> dict[str, float]:
    """Over the questions, the fraction whose top k results include
    "gold@k": a gold paragraph (see gold_paragraphs), and
    "answer@k": a paragraph containing the answer string (what extractive
    answering needs; it may be another article)."""
    by_article: dict[int, list[int]] = {}
    for i, p in enumerate(index.paragraphs):
        by_article.setdefault(p.article_id, []).append(i)
    hits = {f"{m}@{k}": 0 for m in ("gold", "answer") for k in ks}
    for qa in qas:
        gold = gold_paragraphs(qa, index.paragraphs, by_article)
        ranked = [d for d, _ in index.bm25.search(qa.query, max(ks))]
        for k in ks:
            top = ranked[:k]
            hits[f"gold@{k}"] += any(d in gold for d in top)
            hits[f"answer@{k}"] += any(qa.answer in index.paragraphs[d].text for d in top)
    return {name: count / len(qas) for name, count in hits.items()}
