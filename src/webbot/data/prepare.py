"""Turning the jawiki shards into what pretraining reads: a BPE tokenizer
and the token files of the train and validation splits."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from webbot.tokenizer import BPETokenizer

from .jawiki import iter_documents, shard_paths
from .tokens import write_token_file


def _take_chars(docs: Iterator[str], max_chars: int) -> Iterator[str]:
    total = 0
    for doc in docs:
        if total >= max_chars:
            return
        total += len(doc)
        yield doc


def prepare_jawiki(
    raw_dir: str | Path,
    out_dir: str | Path,
    vocab_size: int = 16000,
    tokenizer_chars: int = 20_000_000,
    val_every: int = 100,
    num_workers: int = 8,
    max_docs: int | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train a BPE tokenizer on the first tokenizer_chars characters of the
    training articles, then encode both splits.

    Writes out_dir/tokenizer.json, train.bin and val.bin (each article
    followed by <|endoftext|>; see write_token_file). max_docs limits each
    split (for quick trials). Returns token and character counts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = shard_paths(raw_dir)
    log(f"{len(paths)} shard(s) in {raw_dir}")

    def docs(split: str) -> Iterator[str]:
        it = iter_documents(paths, split, val_every)
        return itertools.islice(it, max_docs) if max_docs is not None else it

    t = time.perf_counter()
    tokenizer = BPETokenizer()
    tokenizer.train(
        list(_take_chars(docs("train"), tokenizer_chars)), vocab_size, num_workers=num_workers
    )
    tokenizer.save(out_dir / "tokenizer.json")
    log(f"tokenizer: {len(tokenizer)} tokens in {time.perf_counter() - t:.1f}s")

    stats = {"vocab_size": len(tokenizer)}
    for split in ("train", "val"):
        t = time.perf_counter()
        chars = 0

        def counted(it: Iterator[str]) -> Iterator[str]:
            nonlocal chars
            for doc in it:
                chars += len(doc)
                yield doc

        n_tokens = write_token_file(tokenizer, counted(docs(split)), out_dir / f"{split}.bin")
        stats[f"{split}_tokens"] = n_tokens
        stats[f"{split}_chars"] = chars
        log(
            f"{split}: {n_tokens:,} tokens from {chars:,} chars "
            f"({chars / max(n_tokens, 1):.2f} chars/token incl. <|endoftext|>) "
            f"in {time.perf_counter() - t:.1f}s"
        )
    return stats
