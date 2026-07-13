"""Filesystem experiment run storage."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from letf.config import ExperimentConfig

DEFAULT_ROOT = Path("experiments")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    run_dir: Path
    config_snapshot: Path
    metrics_path: Path
    summary_path: Path


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "run"


def make_run_id(name: str, when: datetime | None = None) -> str:
    ts = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{_slugify(name)}"


def create_run(
    config: ExperimentConfig,
    root: str | Path = DEFAULT_ROOT,
    run_id: str | None = None,
) -> RunPaths:
    """Create a run directory and write the config snapshot."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    rid = run_id or make_run_id(config.name)
    run_dir = root_path / rid
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    paths = RunPaths(
        run_id=rid,
        root=root_path,
        run_dir=run_dir,
        config_snapshot=run_dir / "config.snapshot.yaml",
        metrics_path=run_dir / "metrics.jsonl",
        summary_path=run_dir / "summary.json",
    )

    with paths.config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

    paths.metrics_path.touch()
    write_summary(
        paths,
        {
            "run_id": rid,
            "name": config.name,
            "train": config.train,
            "status": "running",
        },
    )
    return paths


def append_metric(
    paths: RunPaths,
    *,
    step: int | None = None,
    **metrics: Any,
) -> None:
    """Append one metrics record to metrics.jsonl."""
    if not metrics and step is None:
        raise ValueError("append_metric requires step and/or metric fields")

    record: dict[str, Any] = {}
    if step is not None:
        record["step"] = step
    record.update(metrics)

    with paths.metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_summary(paths: RunPaths, summary: dict[str, Any]) -> None:
    """Write (overwrite) summary.json."""
    with paths.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def finalize_run(
    paths: RunPaths,
    *,
    status: str = "completed",
    result: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark the run finished and merge fields into summary.json."""
    summary = read_summary(paths)
    summary["status"] = status
    if result is not None:
        summary["result"] = result
    if extra:
        summary.update(extra)
    write_summary(paths, summary)
    return summary


def read_summary(paths: RunPaths) -> dict[str, Any]:
    with paths.summary_path.open(encoding="utf-8") as f:
        return json.load(f)


def read_metrics(paths: RunPaths) -> list[dict[str, Any]]:
    if not paths.metrics_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with paths.metrics_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_run(run_id: str, root: str | Path = DEFAULT_ROOT) -> RunPaths:
    run_dir = Path(root) / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    return RunPaths(
        run_id=run_id,
        root=Path(root),
        run_dir=run_dir,
        config_snapshot=run_dir / "config.snapshot.yaml",
        metrics_path=run_dir / "metrics.jsonl",
        summary_path=run_dir / "summary.json",
    )


def list_runs(root: str | Path = DEFAULT_ROOT) -> list[str]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted(
        p.name for p in root_path.iterdir() if p.is_dir() and not p.name.startswith(".")
    )