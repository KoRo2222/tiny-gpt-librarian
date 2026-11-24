"""Submit the pipeline to Kaggle as a script kernel and start it.

    python scripts/kaggle_submit.py [--accelerator TpuV5E8] [--continue]

The kernel runs kaggle/pipeline_kernel.py pinned to the current commit
(which must be pushed to GitHub, since the kernel fetches the code from
there). --continue attaches the kernel's own previous output as an input,
so the run picks up where it stopped. Needs the Kaggle CLI and
~/.kaggle/kaggle.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def kernel_files(commit: str, user: str, slug: str, accelerator: str, cont: bool) -> tuple[str, dict]:
    """The kernel's code (pinned to commit) and its kernel-metadata.json."""
    code = (ROOT / "kaggle" / "pipeline_kernel.py").read_text(encoding="utf-8").replace("__COMMIT__", commit)
    tpu = accelerator.lower().startswith("tpu")
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": "pipeline_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": not tpu, "enable_tpu": tpu, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [f"{user}/{slug}"] if cont else [],
    }
    return code, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default=None, help="Kaggle user name (default: from kaggle.json)")
    ap.add_argument("--slug", default="webbot-pipeline")
    ap.add_argument("--accelerator", default="TpuV5E8", help="e.g. TpuV5E8 or NvidiaTeslaT4")
    ap.add_argument("--continue", dest="cont", action="store_true", help="attach the previous output")
    args = ap.parse_args()

    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no") or git("log", "origin/master..HEAD", "--oneline"):
        sys.exit("commit and push first: the kernel fetches the code from GitHub")
    user = args.user or json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    code, meta = kernel_files(commit, user, args.slug, args.accelerator, args.cont)
    kaggle = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "pipeline_kernel.py").write_text(code, encoding="utf-8")
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"submitting {meta['id']} at commit {commit[:7]} on {args.accelerator}", flush=True)
        subprocess.run([kaggle, "kernels", "push", "-p", tmp, "--accelerator", args.accelerator], check=True)


if __name__ == "__main__":
    main()
