"""Prepare the pretraining data from the jawiki parquet shards.

    python scripts/prepare_data.py --raw data/raw/jawiki --out data/jawiki
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.data.prepare import prepare_jawiki  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/jawiki", help="directory of the parquet shards")
    ap.add_argument("--out", default="data/jawiki", help="output directory")
    ap.add_argument("--vocab", type=int, default=16000, help="vocabulary size incl. special tokens")
    ap.add_argument("--tokenizer-chars", type=float, default=20e6, help="characters to train the tokenizer on")
    ap.add_argument("--val-every", type=int, default=100, help="hold out ~1 in this many articles")
    ap.add_argument("--workers", type=int, default=8, help="processes for counting pretokens")
    ap.add_argument("--max-docs", type=int, default=None, help="limit each split (quick trials)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    prepare_jawiki(args.raw, args.out, args.vocab, int(args.tokenizer_chars), args.val_every,
                   args.workers, args.max_docs)


if __name__ == "__main__":
    main()
