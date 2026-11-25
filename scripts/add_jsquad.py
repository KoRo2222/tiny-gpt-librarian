"""Add the JSQuAD training questions to an SFT data directory.

    python scripts/add_jsquad.py --sft data/sft --jsquad data/raw/jsquad/train-v1.3.json --out data/sft_jsquad

The new directory's train.jsonl holds the existing training examples plus
JSQuAD's (each with its own paragraph as the one document); val.jsonl and
test.jsonl are copied unchanged, so the scores stay comparable.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.eval import load_jsquad, to_sft_examples  # noqa: E402
from webbot.sft import load_jsonl, save_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", required=True, help="SFT data directory (train/val/test.jsonl)")
    ap.add_argument("--jsquad", required=True, help="JSQuAD training json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sft, out = Path(args.sft), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_jsonl(sft / "train.jsonl")
    extra = to_sft_examples(load_jsquad(args.jsquad))
    counts = {"train_base": len(base), "train_jsquad": len(extra),
              "train": save_jsonl(base + extra, out / "train.jsonl")}
    for split in ("val", "test"):
        shutil.copy(sft / f"{split}.jsonl", out / f"{split}.jsonl")
    print(", ".join(f"{k} {v:,}" for k, v in counts.items()), flush=True)
    return counts


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
