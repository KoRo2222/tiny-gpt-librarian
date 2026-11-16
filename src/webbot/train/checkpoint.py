from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from webbot.model import GPT, GPTConfig


def save_checkpoint(path: str | Path, model: GPT, **state: Any) -> None:
    """Save the model (weights + config) plus any extra training state
    (optimizer, scheduler, step, ...; objects with state_dict() are stored
    via it). Written to a temp file and renamed, so an interrupted save
    never leaves a truncated checkpoint behind."""
    path = Path(path)
    payload = {"model_config": asdict(model.config), "model": model.state_dict()}
    for key, value in state.items():
        payload[key] = value.state_dict() if hasattr(value, "state_dict") else value
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> GPT:
    """Rebuild a GPT from a checkpoint, ready for inference."""
    ckpt = load_checkpoint(path, device)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
