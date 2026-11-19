"""Reading the hpprc/jawiki dump (Japanese Wikipedia, one row per article
with its paragraphs) from local parquet shards.

Disambiguation pages are skipped. For pretraining an article becomes its
title, a newline, and its paragraphs joined by newlines. A fixed ~1% of
articles, chosen by article id (independent of shard order, so adding
shards never moves an article between the splits), is held out for
validation. Paragraphs are also exposed one by one, with their article
and section titles, as the units the retriever indexes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

COLUMNS = ["id", "title", "paragraphs", "is_disambiguation_page"]


@dataclass(frozen=True)
class Paragraph:
    article_id: int
    paragraph_id: int
    title: str  # article title
    section: str | None  # section heading, None in the lead
    text: str


def shard_paths(raw_dir: str | Path) -> list[Path]:
    paths = sorted(Path(raw_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards in {raw_dir}")
    return paths


def is_validation(article_id: int, every: int = 100) -> bool:
    """Whether an article is held out: a fixed hash of its id, ~1/every."""
    return (article_id * 2654435761) % 2**32 % every == 0


def iter_articles(
    paths: list[Path], batch_size: int = 2000
) -> Iterator[tuple[int, str, list[dict]]]:
    """(id, title, paragraphs) of every non-disambiguation article with text."""
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=COLUMNS, batch_size=batch_size):
            cols = [batch.column(c).to_pylist() for c in COLUMNS]
            for article_id, title, paragraphs, disambiguation in zip(*cols):
                if disambiguation or not paragraphs:
                    continue
                paragraphs = [p for p in paragraphs if p["text"] and p["text"].strip()]
                if paragraphs:
                    yield article_id, title, paragraphs


def article_text(title: str, paragraphs: list[dict]) -> str:
    return title + "\n" + "\n".join(p["text"] for p in paragraphs)


def iter_documents(
    paths: list[Path], split: str, val_every: int = 100
) -> Iterator[str]:
    """Pretraining documents of one split ("train" or "val")."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    want_val = split == "val"
    for article_id, title, paragraphs in iter_articles(paths):
        if is_validation(article_id, val_every) == want_val:
            yield article_text(title, paragraphs)


def iter_paragraphs(paths: list[Path]) -> Iterator[Paragraph]:
    """Every paragraph of every article (both splits), in shard order."""
    for article_id, title, paragraphs in iter_articles(paths):
        for p in paragraphs:
            yield Paragraph(article_id, p["paragraph_id"], title, p["title"], p["text"])
