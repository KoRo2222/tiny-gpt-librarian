import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.model import GPT, GPTConfig
from webbot.train import is_xla, load_checkpoint, resolve_device, resolve_precision, save_checkpoint, to_cpu, xla_step
from webbot.train.device import tpu_present


def test_auto_is_cpu_without_cuda_or_tpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for k in [k for k in __import__("os").environ if k.startswith(("TPU_", "COLAB_TPU", "PJRT"))]:
        monkeypatch.delenv(k)
    assert not tpu_present()
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")


def test_tpu_is_recognised_from_the_environment(monkeypatch):
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    assert tpu_present()
    monkeypatch.delenv("PJRT_DEVICE")
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-8")
    assert tpu_present()


def test_xla_helpers_do_nothing_elsewhere():
    assert is_xla(torch.device("xla")) and not is_xla("cpu") and not is_xla("cuda")
    xla_step("cpu")  # no torch_xla needed off XLA
    assert resolve_precision("xla").name == "bf16" and not resolve_precision("xla").make_grad_scaler().is_enabled()


def test_to_cpu_keeps_the_structure():
    t = torch.ones(2, requires_grad=True)
    out = to_cpu({"a": [t, (t, 3)], "b": "x"})
    assert out["b"] == "x" and out["a"][1][1] == 3 and isinstance(out["a"][1], tuple)
    assert torch.equal(out["a"][0], t.detach()) and not out["a"][0].requires_grad


def test_checkpoints_for_an_xla_device_load_through_the_cpu(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, max_seq_len=8, d_model=16, n_layers=1, n_heads=2))
    save_checkpoint(tmp_path / "c.pt", model, step=3)
    ckpt = load_checkpoint(tmp_path / "c.pt", torch.device("xla"))  # no TPU here: must not map onto it
    assert ckpt["step"] == 3 and all(v.device.type == "cpu" for v in ckpt["model"].values())
