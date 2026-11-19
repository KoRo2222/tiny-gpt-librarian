"""Build the BM25 paragraph index of the jawiki shards.

    python scripts/build_index.py --raw data/raw/jawiki --out data/index
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.data.jawiki import iter_paragraphs, shard_paths  # noqa: E402
from webbot.retrieval import ParagraphIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/jawiki", help="directory of the jawiki parquet shards")
    ap.add_argument("--out", default="data/index", help="output directory")
    args = ap.parse_args()
    t = time.perf_counter()
    index = ParagraphIndex.build(iter_paragraphs(shard_paths(args.raw)))
    index.save(args.out)
    print(f"{index.bm25.num_docs:,} paragraphs, {len(index.bm25.vocab):,} terms, "
          f"{len(index.bm25.doc_ids):,} postings -> {args.out} ({time.perf_counter() - t:.0f}s)")


if __name__ == "__main__":
    main()
