"""Evaluate the paragraph index with auto-wiki-qa questions about the
indexed articles (a random sample).

    python scripts/eval_retrieval.py --index data/index --qa data/raw/auto-wiki-qa
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.retrieval import ParagraphIndex, evaluate_retrieval, load_qa  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--qa", default="data/raw/auto-wiki-qa", help="directory of the auto-wiki-qa parquet files")
    ap.add_argument("--n", type=int, default=2000, help="questions to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    index = ParagraphIndex.load(args.index)
    qas = load_qa(sorted(Path(args.qa).glob("*.parquet")), {p.article_id for p in index.paragraphs})
    print(f"{len(qas):,} questions about the indexed articles; sampling {min(args.n, len(qas))}")
    qas = random.Random(args.seed).sample(qas, min(args.n, len(qas)))
    t = time.perf_counter()
    metrics = evaluate_retrieval(index, qas)
    dt = (time.perf_counter() - t) / len(qas) * 1000
    for name, value in metrics.items():
        print(f"{name:10s} {value:.1%}")
    print(f"{dt:.0f} ms per question")


if __name__ == "__main__":
    main()
