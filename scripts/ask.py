"""Ask the fine-tuned model a question: it writes a search query, the BM25
index is searched, and it answers from the paragraphs.

    python scripts/ask.py --model runs/kaggle_v1/runs/sft/checkpoint.pt \
        --tokenizer runs/kaggle_v1/data/tokenizer.json "阿久澤 騰の出身大学はどこ？"

Without a question it reads questions from standard input, one per line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.retrieval import ParagraphIndex  # noqa: E402
from webbot.sft.agent import ask  # noqa: E402
from webbot.tokenizer import BPETokenizer  # noqa: E402
from webbot.train import load_model  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args(argv)
    model, tok = load_model(args.model), BPETokenizer.load(args.tokenizer)
    index = ParagraphIndex.load(args.index)
    questions = [" ".join(args.question)] if args.question else (line.strip() for line in sys.stdin)
    for q in questions:
        if not q:
            continue
        reply = ask(model, tok, index, q, args.k)
        print(f"検索語: {reply.query}")
        for d in reply.docs:
            print(f"  段落: {d.title}{' / ' + d.section if d.section else ''}: {d.text[:60]}…")
        if not reply.docs:
            print("  段落: 見つかりませんでした")
        print(f"答え: {reply.answer}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
