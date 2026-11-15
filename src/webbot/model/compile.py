"""torch.compile for the GPT, set up for this machine (Windows, CPU).

torch.compile (TorchDynamo captures the Python forward as a graph, Inductor
generates fused C++/OpenMP code for the CPU) needs a C++ compiler; on
Windows that is MSVC's cl.exe, which is only on PATH inside a Visual Studio
developer shell. setup_compiler() loads that environment (vcvars64.bat,
found with vswhere) into this process when cl is missing, and lets Dynamo
trace the data-dependent shapes of MoE routing (tokens per expert) instead
of breaking the graph there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch
from torch import nn

_VSWHERE = os.path.join(
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    "Microsoft Visual Studio", "Installer", "vswhere.exe",
)


def _load_msvc_environment() -> None:
    """Run vcvars64.bat in a child cmd and copy its environment here."""
    if not os.path.exists(_VSWHERE):
        raise RuntimeError("cl.exe is not on PATH and Visual Studio was not found")
    install = subprocess.run(
        [_VSWHERE, "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not install:
        raise RuntimeError("no Visual Studio installation with the C++ tools")
    vcvars = os.path.join(install, "VC", "Auxiliary", "Build", "vcvars64.bat")
    env = subprocess.run(
        f'cmd /s /c ""{vcvars}" >nul 2>&1 && set"',
        capture_output=True, text=True, check=True,
    ).stdout
    for line in env.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value
    if not shutil.which("cl"):
        raise RuntimeError(f"cl.exe still not found after {vcvars}")


def setup_compiler() -> None:
    """Make torch.compile usable in this process (idempotent)."""
    if sys.platform == "win32" and not shutil.which("cl"):
        _load_msvc_environment()
    # MoE splits tokens by expert with counts known only at run time;
    # trace them as symbolic sizes rather than breaking the graph.
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True


def compile_model(model: nn.Module, **kwargs) -> nn.Module:
    """Compile model in place (nn.Module.compile) and return it.

    In place, so state_dict keys stay the same as the eager model's (no
    "_orig_mod." prefix) and checkpoints remain interchangeable. Only
    calls to the module (model(idx, targets): training and evaluation)
    are compiled; GPT.generate calls forward_with_cache directly and stays
    eager, since its shapes change every step. kwargs go to torch.compile.
    """
    setup_compiler()
    model.compile(**kwargs)
    return model
