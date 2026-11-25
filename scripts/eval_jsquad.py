"""Evaluate reading comprehension on JSQuAD (dev): each question is given
its paragraph as the only document.

    python scripts/eval_jsquad.py --model runs/kaggle_v1/runs/sft/checkpoint.pt \
        --tokenizer runs/kaggle_v1/data/tokenizer.json --data data/raw/jsquad/valid-v1.3.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from webbot.eval import evaluate_jsquad, load_jsquad  # noqa: E402
from webbot.tokenizer import BPETokenizer  # noqa: E402
from webbot.train import load_model  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default="data/raw/jsquad/valid-v1.3.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, tok = load_model(args.model), BPETokenizer.load(args.tokenizer)
    items = load_jsquad(args.data)[: args.limit]
    print(f"{len(items)} questions", flush=True)
    t = time.perf_counter()
    result = evaluate_jsquad(model, tok, items, log=lambda s: print(s, flush=True))
    print(", ".join(f"{k} {v:.1%}" for k, v in result.items())
          + f" ({(time.perf_counter() - t) / max(len(items), 1):.2f} s/question)", flush=True)
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
