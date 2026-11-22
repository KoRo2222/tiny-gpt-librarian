"""Make the SFT data (search, read, answer) from auto-wiki-qa and the BM25
paragraph index.

    python scripts/make_sft_data.py --index data/index --qa data/raw/auto-wiki-qa --out data/sft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.sft.build import build_sft_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--qa", default="data/raw/auto-wiki-qa", help="directory of the auto-wiki-qa parquet files")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--k", type=int, default=3, help="paragraphs per question")
    ap.add_argument("--max-doc-chars", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick trials)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    build_sft_data(args.index, sorted(Path(args.qa).glob("*.parquet")), args.out, args.k,
                   args.max_doc_chars, args.workers, args.limit)


if __name__ == "__main__":
    main()
