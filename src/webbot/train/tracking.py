"""Experiment tracking with Weights & Biases.

pretrain logs through a WandbTracker when PretrainConfig.wandb_project is
set: the configuration once, then per step the training loss, learning
rate, tokens seen and throughput, and per evaluation the validation loss.
Only rank 0 tracks in data-parallel training. The run id is kept in the
checkpoint so a resumed training continues the same run.

The default mode is "offline": runs are written under <dir>/wandb/ on this
machine and nothing is sent anywhere (`wandb sync <run dir>` uploads one
later). "online" streams to wandb.ai and needs `wandb login` first.
"""

from __future__ import annotations

from typing import Any


class WandbTracker:
    def __init__(
        self,
        project: str,
        config: dict[str, Any],
        name: str | None = None,
        mode: str = "offline",
        dir: str | None = None,
        run_id: str | None = None,
    ):
        try:
            import wandb
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError("experiment tracking needs `pip install wandb`") from e
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            mode=mode,
            dir=dir,
            id=run_id,
            resume="allow" if run_id is not None else None,
        )

    @property
    def run_id(self) -> str:
        return self.run.id

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()
