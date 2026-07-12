"""Experiment config loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AwoConfig:
    enabled: bool = True
    interval: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train: str
    seed: int = 42
    hparams: dict[str, Any] = field(default_factory=dict)
    awo: AwoConfig = field(default_factory=AwoConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate a raw dict and return ExperimentConfig."""
    missing = [key for key in ("name", "train") if key not in raw]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

    name = raw["name"]
    train = raw["train"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' must be a non-empty string")
    if not isinstance(train, str) or not train.strip():
        raise ValueError("'train' must be a non-empty string")

    seed = raw.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("'seed' must be an int")

    hparams = raw.get("hparams", {})
    if not isinstance(hparams, dict):
        raise ValueError("'hparams' must be a mapping")

    awo_raw = raw.get("awo", {})
    if awo_raw is None:
        awo_raw = {}
    if not isinstance(awo_raw, dict):
        raise ValueError("'awo' must be a mapping")

    enabled = awo_raw.get("enabled", True)
    interval = awo_raw.get("interval", 5)
    if not isinstance(enabled, bool):
        raise ValueError("'awo.enabled' must be a bool")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("'awo.interval' must be a positive int")

    return ExperimentConfig(
        name=name.strip(),
        train=train.strip(),
        seed=seed,
        hparams=dict(hparams),
        awo=AwoConfig(enabled=enabled, interval=interval),
    )