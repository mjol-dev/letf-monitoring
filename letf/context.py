"""TrainContext, TrainResult, and metric logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from letf.config import ExperimentConfig
from letf.tracker import RunPaths, append_metric


@dataclass
class TrainContext:
    config: ExperimentConfig
    paths: RunPaths
    device: str = "cpu"

    @property
    def run_id(self) -> str:
        return self.paths.run_id

    @property
    def run_dir(self):
        return self.paths.run_dir

    @property
    def hparams(self) -> dict[str, Any]:
        return self.config.hparams

    def log_metric(self, step: int | None = None, **metrics: Any) -> None:
        """Append metrics to this run's metrics.jsonl."""
        append_metric(self.paths, step=step, **metrics)


@dataclass(frozen=True)
class TrainResult:
    """Final outcome returned by a train function."""

    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)