"""Building the SFT data files from the paragraph index and auto-wiki-qa."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from multiprocessing import Pool
from pathlib import Path

from webbot.retrieval import QA, ParagraphIndex, load_qa

from .data import SFTExample, make_example, save_jsonl, split_of

_worker_index: ParagraphIndex | None = None
_worker_args: tuple[int, int] = (3, 400)


def _init_worker(index_dir: str, k: int, max_doc_chars: int) -> None:
    global _worker_index, _worker_args
    _worker_index = ParagraphIndex.load(index_dir)
    _worker_args = (k, max_doc_chars)


def _make(qa: QA) -> SFTExample | None:
    return make_example(qa, _worker_index, *_worker_args)


def build_sft_data(
    index_dir: str | Path,
    qa_paths: Sequence[str | Path],
    out_dir: str | Path,
    k: int = 3,
    max_doc_chars: int = 400,
    workers: int = 8,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Write out_dir/{train,val,test}.jsonl; return counts.

    Questions are those about the indexed articles (the first `limit` of
    them if given), searched in `workers` processes (each loads the index).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = ParagraphIndex.load(index_dir)
    qas = load_qa(qa_paths, {p.article_id for p in index.paragraphs})
    del index
    if limit is not None:
        qas = qas[:limit]
    log(f"{len(qas):,} questions about the indexed articles")

    t = time.perf_counter()
    args = (str(index_dir), k, max_doc_chars)
    if workers > 1:
        with Pool(workers, initializer=_init_worker, initargs=args) as pool:
            results = pool.map(_make, qas, chunksize=256)
    else:
        _init_worker(*args)
        results = [_make(qa) for qa in qas]
    log(f"searched in {time.perf_counter() - t:.0f}s")

    splits: dict[str, list[SFTExample]] = {"train": [], "val": [], "test": []}
    asked = Counter(split_of(qa.article_id) for qa in qas)
    for ex in results:
        if ex is not None:
            splits[split_of(ex.article_id)].append(ex)
    stats = {"questions": len(qas)}
    for name, examples in splits.items():
        save_jsonl(examples, out_dir / f"{name}.jsonl")
        stats[name] = len(examples)
        log(f"{name}: {len(examples):,} examples kept of {asked[name]:,} questions "
            f"({len(examples) / max(asked[name], 1):.1%} have the answer in the top {k})")
    return stats
