"""Pretrain the GPT on the token files made by prepare_data.py.

    python scripts/pretrain.py --data data/jawiki --out runs/base --steps 20000 --resume

With --resume the run continues from <out>/checkpoint.pt when it exists and
starts fresh otherwise, so the same command can simply be run again after
an interruption (e.g. a Colab disconnect). The log is also appended to
<out>/train.log; at the end a few prompts are continued as a quick look.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from webbot.data import load_token_file  # noqa: E402
from webbot.model import GPT, GPTConfig, compile_model  # noqa: E402
from webbot.tokenizer import BPETokenizer  # noqa: E402
from webbot.train import PretrainConfig, load_checkpoint, pretrain  # noqa: E402

PROMPTS = ["日本の首都は", "富士山\n富士山は", "人工知能\n人工知能(じんこうちのう)とは、"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/jawiki", help="directory with tokenizer.json, train.bin, val.bin")
    ap.add_argument("--out", required=True, help="directory for checkpoint.pt and train.log")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pt if present")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cpu or cuda")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    m = ap.add_argument_group("model (ignored when resuming: taken from the checkpoint)")
    m.add_argument("--d-model", type=int, default=512)
    m.add_argument("--n-layers", type=int, default=8)
    m.add_argument("--n-heads", type=int, default=8)
    m.add_argument("--n-kv-heads", type=int, default=None)
    m.add_argument("--max-seq-len", type=int, default=1024)
    m.add_argument("--attn-impl", default="sdpa", choices=["naive", "flash", "sdpa"])
    t = ap.add_argument_group("training")
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--seq-len", type=int, default=1024)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (AdamW part)")
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    t.add_argument("--muon-lr", type=float, default=0.02)
    t.add_argument("--precision", default="auto")
    t.add_argument("--eval-interval", type=int, default=500)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--grad-checkpoint", action="store_true")
    w = ap.add_argument_group("Weights & Biases")
    w.add_argument("--wandb-project", default=None)
    w.add_argument("--wandb-mode", default="offline", choices=["offline", "online"])
    ap.add_argument("--sample-tokens", type=int, default=60, help="tokens to generate per prompt at the end (0: none)")
    return ap


def main(argv: list[str] | None = None) -> list[dict]:
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
    tokenizer = BPETokenizer.load(Path(args.data) / "tokenizer.json")
    resume = args.resume and ckpt_path.exists()
    if resume:
        model_config = GPTConfig(**load_checkpoint(ckpt_path, "cpu")["model_config"])
    else:
        model_config = GPTConfig(
            vocab_size=len(tokenizer), max_seq_len=args.max_seq_len, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
            attn_impl=args.attn_impl,
        )
    model_config.grad_checkpoint = args.grad_checkpoint
    model = GPT(model_config)
    log(f"device {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else "")
        + f", model {asdict(model_config)}")
    if args.compile:
        compile_model(model)
    config = PretrainConfig(
        total_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        grad_accum_steps=args.grad_accum, peak_lr=args.lr, warmup_steps=args.warmup,
        optimizer=args.optimizer, muon_lr=args.muon_lr, precision=args.precision,
        eval_interval=args.eval_interval, eval_batches=args.eval_batches, seed=args.seed,
        wandb_project=args.wandb_project, wandb_mode=args.wandb_mode, wandb_dir=str(out),
    )
    history = pretrain(
        model, load_token_file(Path(args.data) / "train.bin"), load_token_file(Path(args.data) / "val.bin"),
        config, device=device, checkpoint_path=ckpt_path, resume=resume, log=log,
    )
    if args.sample_tokens > 0:
        model.eval()
        eot = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
        for prompt in PROMPTS:
            idx = torch.tensor([tokenizer.encode(prompt)], device=device)
            torch.manual_seed(0)
            ids = model.generate(idx, args.sample_tokens, temperature=0.8, top_k=50, eot_id=eot)
            log("sample: " + tokenizer.decode(ids[0].tolist()).replace("\n", " / "))
    log_file.close()
    return history


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
