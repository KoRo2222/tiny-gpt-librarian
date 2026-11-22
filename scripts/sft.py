"""Fine-tune a pretrained model on the search-read-answer data
(make_sft_data.py), then measure exact-match answers on the test split.

    python scripts/sft.py --init runs/base/checkpoint.pt --tokenizer data/jawiki/tokenizer.json \\
        --data data/sft --out runs/sft --steps 3000 --resume

The test questions are about articles held out from pretraining, and the
model is given the same retrieved paragraphs as in the data, so the score
measures reading the paragraphs (retrieval is evaluated separately).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from webbot.model import GPT, GPTConfig  # noqa: E402
from webbot.sft import encode_example, load_jsonl  # noqa: E402
from webbot.sft.infer import exact_match, generate_answer, normalize_answer  # noqa: E402
from webbot.sft.train import SFTConfig, train_sft  # noqa: E402
from webbot.tokenizer import BPETokenizer  # noqa: E402
from webbot.train import load_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="pretrained checkpoint to start from")
    ap.add_argument("--tokenizer", required=True, help="the tokenizer.json the model was pretrained with")
    ap.add_argument("--data", default="data/sft", help="directory with train/val/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pt if present")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-examples", type=int, default=1000, help="validation examples for the loss")
    ap.add_argument("--test-examples", type=int, default=300, help="test questions answered at the end (0: none)")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"
    log_file = (out / "train.log").open("a", encoding="utf-8")

    def log(line: str) -> None:
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tokenizer = BPETokenizer.load(args.tokenizer)
    resume = args.resume and ckpt_path.exists()
    init = load_checkpoint(ckpt_path if resume else args.init, "cpu")
    model = GPT(GPTConfig(**init["model_config"]))
    model.load_state_dict(init["model"])  # resume: overwritten again by train_sft
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("the tokenizer does not match the model's vocabulary")
    max_len = model.config.max_seq_len + 1  # inputs and targets are one shorter

    def encoded(split: str, limit: int | None = None):
        examples = load_jsonl(Path(args.data) / f"{split}.jsonl")[:limit]
        enc = [encode_example(tokenizer, ex) for ex in examples]
        kept = [e for e in enc if len(e[0]) <= max_len]
        if len(kept) < len(enc):
            log(f"{split}: dropped {len(enc) - len(kept)} examples longer than {max_len} tokens")
        return kept

    pad_id = tokenizer.special_tokens[tokenizer.PAD_TOKEN]
    config = SFTConfig(total_steps=args.steps, batch_size=args.batch_size, peak_lr=args.lr,
                       warmup_steps=args.warmup, grad_accum_steps=args.grad_accum,
                       eval_interval=args.eval_interval, precision=args.precision, seed=args.seed)
    history = train_sft(model, encoded("train"), encoded("val", args.val_examples), pad_id, config,
                        device=device, checkpoint_path=ckpt_path, resume=resume, log=log)

    result = {"history": history}
    if args.test_examples > 0:
        tests = load_jsonl(Path(args.data) / "test.jsonl")[: args.test_examples]
        hits = contained = 0
        for i, ex in enumerate(tests):
            pred = generate_answer(model, tokenizer, ex.question, ex.docs, ex.query)
            hits += exact_match(pred, ex.answer)
            contained += normalize_answer(ex.answer) in normalize_answer(pred)
            if i < 5:
                log(f"Q: {ex.question} | gold: {ex.answer} | pred: {pred}")
        result["exact_match"] = hits / len(tests)
        result["contains_answer"] = contained / len(tests)
        log(f"test ({len(tests)} questions, given the retrieved paragraphs): exact match "
            f"{result['exact_match']:.1%}, prediction contains the answer {result['contains_answer']:.1%}")
    log_file.close()
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
