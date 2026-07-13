"""Summarize a single experiment run."""

from __future__ import annotations

from typing import Any

from letf.tracker import RunPaths, get_run, list_runs, read_metrics, read_summary


def analyze_run(run_id: str, root: str = "experiments") -> dict[str, Any]:
    """Return a dict summary for one run (also useful for CLI printing)."""
    paths = get_run(run_id, root=root)
    summary = read_summary(paths)
    metrics = read_metrics(paths)

    losses = [m["loss"] for m in metrics if isinstance(m.get("loss"), (int, float))]
    out: dict[str, Any] = {
        "run_id": paths.run_id,
        "status": summary.get("status"),
        "name": summary.get("name"),
        "train": summary.get("train"),
        "duration_sec": summary.get("duration_sec"),
        "result": summary.get("result"),
        "metric_points": len(metrics),
    }
    if losses:
        out["loss_min"] = min(losses)
        out["loss_max"] = max(losses)
        out["loss_last"] = losses[-1]

    awo_path = paths.run_dir / "awo_log.jsonl"
    out["awo_log"] = awo_path.is_file()
    return out


def format_analysis(data: dict[str, Any]) -> str:
    lines = [
        f"run_id: {data.get('run_id')}",
        f"name: {data.get('name')}",
        f"train: {data.get('train')}",
        f"status: {data.get('status')}",
        f"duration_sec: {data.get('duration_sec')}",
        f"metric_points: {data.get('metric_points')}",
        f"awo_log: {data.get('awo_log')}",
    ]
    if "loss_last" in data:
        lines.append(
            f"loss: last={data['loss_last']:.4f} "
            f"min={data['loss_min']:.4f} max={data['loss_max']:.4f}"
        )
    if data.get("result"):
        lines.append(f"result: {data['result']}")
    return "\n".join(lines)