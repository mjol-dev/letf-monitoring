"""Orchestrate AWO, training, and tracking for a run."""

from __future__ import annotations

import time
from pathlib import Path

from letf.awo_bridge import AwoBridge
from letf.config import ExperimentConfig, load_config
from letf.context import TrainContext, TrainResult
from letf.tracker import RunPaths, create_run, finalize_run
from letf.trains import resolve_train


def run_experiment(
    config: str | Path | ExperimentConfig,
    *,
    root: str | Path = "experiments",
    device: str = "cpu",
    run_id: str | None = None,
) -> RunPaths:
    """Run one experiment end-to-end with optional in-process AWO."""
    cfg = config if isinstance(config, ExperimentConfig) else load_config(config)
    train_fn = resolve_train(cfg.train)
    paths = create_run(cfg, root=root, run_id=run_id)
    ctx = TrainContext(config=cfg, paths=paths, device=device)

    bridge: AwoBridge | None = None
    if cfg.awo.enabled:
        bridge = AwoBridge(
            log_path=paths.run_dir / "awo_log.jsonl",
            tag=paths.run_id,
            interval=float(cfg.awo.interval),
        )
        bridge.start()

    started = time.time()
    try:
        result = train_fn(ctx)
        if not isinstance(result, TrainResult):
            raise TypeError(
                f"Train plugin must return TrainResult, got {type(result)!r}"
            )
        finalize_run(
            paths,
            status="completed",
            result={
                "metrics": result.metrics,
                "artifacts": result.artifacts,
            },
            extra={
                "duration_sec": time.time() - started,
                "device": device,
                "awo_enabled": cfg.awo.enabled,
            },
        )
    except Exception as exc:
        finalize_run(
            paths,
            status="failed",
            extra={
                "duration_sec": time.time() - started,
                "device": device,
                "awo_enabled": cfg.awo.enabled,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if bridge is not None:
            bridge.stop()

    return paths