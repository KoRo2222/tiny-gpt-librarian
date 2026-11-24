"""Run the whole pipeline unattended (made for a Kaggle "Save & Run All"
commit, which keeps running with the browser closed):

    1. prepare the pretraining data       -> <work>/data
    2. pretrain                           -> <work>/runs/base
    3. build the BM25 index (one shard)   -> <scratch>/index
    4. make the SFT data                  -> <work>/sft
    5. fine-tune and answer the test set  -> <work>/runs/sft

<work> keeps what is worth saving (token files, checkpoints, logs); raw
downloads go to <scratch>. Each stage is skipped when its output already
exists, and training resumes from its checkpoint, so an unfinished run is
continued by running again with --resume-from <previous output>. Training
stops when --deadline-hours is reached (checkpoints are written atomically)
and the script still exits normally, so the platform saves the outputs.

    python scripts/run_pipeline.py --work /kaggle/working --scratch /tmp/webbot
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_JAWIKI_SHARDS = 22


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="outputs worth keeping")
    ap.add_argument("--scratch", required=True, help="downloads and other throwaway files")
    ap.add_argument("--resume-from", default=None, help="a previous run's <work> to continue from")
    ap.add_argument("--deadline-hours", type=float, default=11.0, help="stop training after this long")
    ap.add_argument("--device", default="auto")
    d = ap.add_argument_group("data")
    d.add_argument("--shards", default=f"0-{N_JAWIKI_SHARDS - 1}", help="jawiki shards to pretrain on, e.g. 0-21 or 3,21")
    d.add_argument("--retrieval-shard", type=int, default=21, help="the shard the retriever searches")
    d.add_argument("--qa-file", default="train-00003-of-00004.parquet", help="auto-wiki-qa file about that shard")
    d.add_argument("--jawiki-raw", default=None, help="use these local jawiki parquet files instead of downloading")
    d.add_argument("--qa-raw", default=None, help="directory holding --qa-file instead of downloading")
    d.add_argument("--vocab", type=int, default=16000)
    d.add_argument("--workers", type=int, default=4)
    p = ap.add_argument_group("pretraining")
    p.add_argument("--train-tokens", type=float, default=800e6)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024, help="pretraining window")
    p.add_argument("--max-seq-len", type=int, default=None, help="model context (default: --seq-len)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=500)
    s = ap.add_argument_group("SFT")
    s.add_argument("--sft-steps", type=int, default=6000)
    s.add_argument("--sft-batch-size", type=int, default=32)
    s.add_argument("--sft-grad-accum", type=int, default=4)
    s.add_argument("--sft-lr", type=float, default=3e-4)
    s.add_argument("--sft-warmup", type=int, default=100)
    s.add_argument("--sft-eval-interval", type=int, default=500)
    s.add_argument("--test-examples", type=int, default=878)
    s.add_argument("--sft-limit", type=int, default=None, help="only the first N questions (quick trials)")
    return ap


def parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return sorted(set(out))


def shard_name(i: int) -> str:
    return f"train-{i:05d}-of-{N_JAWIKI_SHARDS:05d}.parquet"


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.work = Path(args.work)
        self.scratch = Path(args.scratch)
        self.start = time.monotonic()
        self.work.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.work / "pipeline.log").open("a", encoding="utf-8")

    def log(self, line: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def remaining(self) -> float:
        return self.args.deadline_hours * 3600 - (time.monotonic() - self.start)

    def run(self, name: str, cmd: list[str]) -> bool:
        """Run a stage's command; False if the deadline cut it short."""
        left = self.remaining()
        if left <= 0:
            self.log(f"{name}: no time left")
            return False
        self.log(f"{name}: {' '.join(cmd)}")
        try:
            subprocess.run([sys.executable, *cmd], check=True, timeout=left, cwd=ROOT)
        except subprocess.TimeoutExpired:
            self.log(f"{name}: stopped at the deadline ({self.args.deadline_hours} h)")
            return False
        return True

    # --- inputs ------------------------------------------------------------

    def _download(self, repo: str, name: str, dest: Path) -> Path:
        target = dest / Path(name).name
        if not target.exists():
            from huggingface_hub import hf_hub_download

            dest.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(repo, name, repo_type="dataset", local_dir=self.scratch / "hf")
            shutil.move(path, target)
            self.log(f"downloaded {repo}/{name}")
        return target

    def jawiki_shard(self, i: int) -> Path:
        if self.args.jawiki_raw:
            return Path(self.args.jawiki_raw) / shard_name(i)
        return self._download("hpprc/jawiki", f"data/{shard_name(i)}", self.scratch / "jawiki")

    def qa_dir(self) -> Path:
        if self.args.qa_raw:
            return Path(self.args.qa_raw)
        self._download("cl-nagoya/auto-wiki-qa", f"data/{self.args.qa_file}", self.scratch / "qa")
        return self.scratch / "qa"

    def _linked_dir(self, name: str, files: list[Path]) -> Path:
        """A scratch directory holding exactly these files (hard links or copies)."""
        d = self.scratch / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for f in files:
            try:
                os.link(f, d / f.name)
            except OSError:
                shutil.copy(f, d / f.name)
        return d

    # --- stages --------------------------------------------------------------

    def resume(self) -> None:
        src = self.args.resume_from
        if src and Path(src).resolve() != self.work.resolve():
            self.log(f"copying the previous run from {src}")
            shutil.copytree(src, self.work, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("pipeline.log"))

    def prepare(self) -> bool:
        data = self.work / "data"
        if (data / "val.bin.json").exists():
            self.log("data: already prepared")
            return True
        raw = self._linked_dir("jawiki_selected", [self.jawiki_shard(i) for i in parse_shards(self.args.shards)])
        return self.run("data", ["scripts/prepare_data.py", "--raw", str(raw), "--out", str(data),
                                 "--vocab", str(self.args.vocab), "--workers", str(self.args.workers)])

    def pretrain(self) -> bool:
        out = self.work / "runs" / "base"
        if (out / "DONE").exists():
            self.log("pretraining: already done")
            return True
        a = self.args
        steps = int(a.train_tokens) // (a.batch_size * a.seq_len)
        ok = self.run("pretraining", [
            "scripts/pretrain.py", "--data", str(self.work / "data"), "--out", str(out), "--resume",
            "--steps", str(steps), "--warmup", str(a.warmup), "--lr", str(a.lr),
            "--d-model", str(a.d_model), "--n-layers", str(a.n_layers), "--n-heads", str(a.n_heads),
            "--max-seq-len", str(a.max_seq_len or a.seq_len), "--seq-len", str(a.seq_len), "--batch-size", str(a.batch_size),
            "--grad-accum", str(a.grad_accum), "--eval-interval", str(a.eval_interval), "--device", a.device,
        ])
        if ok:
            (out / "DONE").write_text("done\n")
        return ok

    def sft_data(self) -> bool:
        sft = self.work / "sft"
        if (sft / "test.jsonl").exists():
            self.log("SFT data: already made")
            return True
        index = self.scratch / "index"
        retrieval = self._linked_dir("jawiki_retrieval", [self.jawiki_shard(self.args.retrieval_shard)])
        if not self.run("index", ["scripts/build_index.py", "--raw", str(retrieval), "--out", str(index)]):
            return False
        limit = ["--limit", str(self.args.sft_limit)] if self.args.sft_limit else []
        return self.run("SFT data", ["scripts/make_sft_data.py", "--index", str(index), "--qa", str(self.qa_dir()),
                                     "--out", str(sft), "--workers", str(self.args.workers), *limit])

    def sft(self) -> bool:
        out = self.work / "runs" / "sft"
        if (out / "DONE").exists():
            self.log("SFT: already done")
            return True
        a = self.args
        ok = self.run("SFT", [
            "scripts/sft.py", "--init", str(self.work / "runs" / "base" / "checkpoint.pt"),
            "--tokenizer", str(self.work / "data" / "tokenizer.json"), "--data", str(self.work / "sft"),
            "--out", str(out), "--resume", "--steps", str(a.sft_steps), "--batch-size", str(a.sft_batch_size),
            "--grad-accum", str(a.sft_grad_accum), "--lr", str(a.sft_lr), "--warmup", str(a.sft_warmup),
            "--eval-interval", str(a.sft_eval_interval), "--test-examples", str(a.test_examples),
            "--device", a.device,
        ])
        if ok:
            (out / "DONE").write_text("done\n")
        return ok

    def __call__(self) -> bool:
        self.resume()
        for stage in (self.prepare, self.pretrain, self.sft_data, self.sft):
            if not stage():
                self.log("unfinished: run again with --resume-from <this run's output> to continue")
                return False
        self.log(f"all stages done in {(time.monotonic() - self.start) / 3600:.2f} h")
        return True


def main(argv: list[str] | None = None) -> bool:
    pipeline = Pipeline(build_parser().parse_args(argv))
    try:
        return pipeline()
    finally:
        pipeline.log_file.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
