"""Compare metrics across experiment runs."""

from __future__ import annotations

from typing import Any

from letf.analyzer import analyze_run


def compare_runs(run_ids: list[str], root: str = "experiments") -> list[dict[str, Any]]:
    """Return per-run analysis rows for side-by-side comparison."""
    return [analyze_run(run_id, root=root) for run_id in run_ids]


def format_comparison(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No runs to compare."

    headers = ["run_id", "status", "duration_sec", "loss_last", "metric_points"]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append(
            "\t".join(
                str(row.get(h, ""))
                for h in headers
            )
        )
    return "\n".join(lines)