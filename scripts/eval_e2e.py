"""Evaluate search-read-answer end to end on the test questions (about the
articles held out from pretraining and SFT), including those whose answer
the retriever does not find: once with the queries the model writes and
once searching with the question itself.

    python scripts/eval_e2e.py --model runs/kaggle_v1/runs/sft/checkpoint.pt \
        --tokenizer runs/kaggle_v1/data/tokenizer.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from webbot.retrieval import ParagraphIndex, load_qa  # noqa: E402
from webbot.sft import split_of  # noqa: E402
from webbot.sft.agent import evaluate_end_to_end  # noqa: E402
from webbot.tokenizer import BPETokenizer  # noqa: E402
from webbot.train import load_model  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--qa", default="data/raw/auto-wiki-qa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, tok = load_model(args.model), BPETokenizer.load(args.tokenizer)
    index = ParagraphIndex.load(args.index)
    qas = [qa for qa in load_qa(sorted(Path(args.qa).glob("*.parquet")), {p.article_id for p in index.paragraphs})
           if split_of(qa.article_id) == args.split][: args.limit]
    print(f"{len(qas)} {args.split} questions", flush=True)
    results = {}
    for name, own in (("model query", True), ("question as query", False)):
        t = time.perf_counter()
        m = evaluate_end_to_end(model, tok, index, qas, use_model_query=own, log=print)
        results[name] = m
        print(f"[{name}] " + ", ".join(f"{k} {v:.1%}" for k, v in m.items())
              + f" ({(time.perf_counter() - t) / len(qas):.2f} s/question)", flush=True)
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
