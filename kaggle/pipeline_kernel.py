"""Kaggle kernel (script): run the webbot pipeline on a Kaggle accelerator.

Submitted with scripts/kaggle_submit.py, which pins COMMIT to the commit
being submitted. Steps: fetch that commit, make sure PyTorch/XLA is there
on a TPU, build the Rust BPE, run every stage once at toy size (so a broken
environment fails in minutes, not hours), then the real run into
/kaggle/working (continuing a previous run attached as an input).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/KoRo2222/tiny-gpt-librarian"
COMMIT = "__COMMIT__"
CODE = "/tmp/webbot_code"
WORK = "/kaggle/working"
DEADLINE_HOURS = float(os.environ.get("WEBBOT_DEADLINE_HOURS", "8.0"))  # TPU sessions end at 9 h
LIBTPU = "https://storage.googleapis.com/libtpu-releases/index.html"
START = time.monotonic()


def sh(cmd: str, check: bool = True) -> int:
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def on_tpu() -> bool:
    return (os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
            or any(k.startswith("TPU_") for k in os.environ)
            or bool(glob.glob("/dev/accel*")) or bool(glob.glob("/dev/vfio/[0-9]*")))


def have(module: str) -> bool:
    return subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True).returncode == 0


def torch_version() -> str:
    out = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__.split('+')[0])"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


# 1. code
sh(f"git clone -q {REPO} {CODE}")
sh(f"git -C {CODE} checkout -q {COMMIT}")

# 2. PyTorch (+ XLA on a TPU) and the other dependencies
if on_tpu():
    os.environ["PJRT_DEVICE"] = "TPU"
    if not have("torch_xla"):
        version = torch_version()
        spec = f"torch=={version} torch_xla[tpu]=={version}" if version else "torch torch_xla[tpu]"
        if sh(f"pip install -q {spec} -f {LIBTPU}", check=False):
            sh(f"pip install -q torch torch_xla[tpu] -f {LIBTPU}")
    check = ("import torch, torch_xla, torch_xla.core.xla_model as xm; "
             "print('torch', torch.__version__, 'torch_xla', torch_xla.__version__, 'device', xm.xla_device())")
    subprocess.run([sys.executable, "-c", check], check=True)
for module in ("numpy", "pyarrow", "huggingface_hub"):
    if not have(module):
        sh(f"pip install -q {module}")

# 3. the Rust BPE (about 25x faster tokenizer training and encoding)
if not os.path.exists(os.path.expanduser("~/.cargo/bin/cargo")):
    sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q")
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
sh("pip install -q maturin")
sh(f"pip install -q {CODE}/rust")

pipeline = f"{sys.executable} {CODE}/scripts/run_pipeline.py --scratch /tmp/webbot --workers {os.cpu_count() or 4}"

# 4. every stage at toy size: the searched shard only, a few steps each
smoke = (f"{pipeline} --work /tmp/smoke --shards 21 --train-tokens {32 * 1024 * 20} --warmup 5 --eval-interval 10 "
         f"--sft-steps 20 --sft-warmup 5 --sft-eval-interval 10 --sft-limit 3000 --test-examples 5 --deadline-hours 1")
smoke_ok = sh(smoke, check=False) == 0 and os.path.exists("/tmp/smoke/runs/sft/DONE")
os.makedirs(f"{WORK}/smoke", exist_ok=True)
for log in ("pipeline.log", "runs/base/train.log", "runs/sft/train.log"):
    if os.path.exists(f"/tmp/smoke/{log}"):
        shutil.copy(f"/tmp/smoke/{log}", f"{WORK}/smoke/{log.replace('/', '_')}")
if not smoke_ok:
    sys.exit("the toy-size run failed: see the log above and smoke/ in the output")

# 5. the real run, continuing a previous one attached as an input
previous = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pipeline.log", recursive=True)]
left = DEADLINE_HOURS - (time.monotonic() - START) / 3600
cmd = f"{pipeline} --work {WORK} --deadline-hours {left:.2f}"
if previous:
    cmd += f" --resume-from {previous[0]}"
sh(cmd, check=False)
