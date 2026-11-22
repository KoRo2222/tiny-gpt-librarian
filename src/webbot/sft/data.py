"""Supervised fine-tuning examples for "search, read, answer".

Each example is built from an auto-wiki-qa question: the search query (for
now the question itself; BM25 copes well with whole questions), the top-k
paragraphs the BM25 index returns for it (each cut to max_doc_chars), and
the answer. Only examples whose answer string appears in one of those
paragraphs are kept, so the answer can always be copied from what the
model reads; at inference the same index supplies the paragraphs.

Splits follow the article: "test" holds the questions about the articles
held out from pretraining (is_validation, residue 0, ~1%), so it measures
reading, not recall; "val" another ~1% of articles; "train" the rest.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from webbot.data.jawiki import Paragraph, is_validation
from webbot.retrieval import QA, ParagraphIndex


@dataclass(frozen=True)
class Doc:
    title: str
    section: str | None
    text: str


@dataclass(frozen=True)
class SFTExample:
    question: str
    query: str
    docs: tuple[Doc, ...]
    answer: str
    article_id: int


def split_of(article_id: int, every: int = 100) -> str:
    if is_validation(article_id, every):
        return "test"  # also held out from pretraining
    if is_validation(article_id + 1, every):
        return "val"
    return "train"


def make_doc(p: Paragraph, max_chars: int) -> Doc:
    return Doc(p.title, p.section, p.text[:max_chars])


def make_example(
    qa: QA, index: ParagraphIndex, k: int = 3, max_doc_chars: int = 400
) -> SFTExample | None:
    """The example for one question, or None if the answer is in none of
    the (cut) retrieved paragraphs."""
    query = qa.query  # the question itself is the search query, for now
    docs = tuple(make_doc(p, max_doc_chars) for p, _ in index.search(query, k))
    if not any(qa.answer in d.text for d in docs):
        return None
    return SFTExample(qa.query, query, docs, qa.answer, qa.article_id)


def save_jsonl(examples: Iterable[SFTExample], path: str | Path) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[SFTExample]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[SFTExample]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["docs"] = tuple(Doc(**doc) for doc in d["docs"])
            yield SFTExample(**d)
