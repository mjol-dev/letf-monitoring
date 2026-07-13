from datetime import datetime, timezone

import pytest
import yaml

from letf.config import ExperimentConfig
from letf.tracker import (
    append_metric,
    create_run,
    finalize_run,
    get_run,
    list_runs,
    make_run_id,
    read_metrics,
    read_summary,
)


def test_make_run_id_slug():
    when = datetime(2026, 7, 12, 21, 3, 45, tzinfo=timezone.utc)
    assert make_run_id("MNIST Baseline", when) == "20260712T210345Z_mnist-baseline"


def test_create_run_writes_snapshot_and_summary(tmp_path):
    cfg = ExperimentConfig(name="mnist-baseline", train="builtin:mnist")
    paths = create_run(cfg, root=tmp_path, run_id="test_run")

    assert paths.run_dir == tmp_path / "test_run"
    assert paths.config_snapshot.is_file()
    assert paths.metrics_path.is_file()

    snap = yaml.safe_load(paths.config_snapshot.read_text(encoding="utf-8"))
    assert snap["name"] == "mnist-baseline"
    assert snap["train"] == "builtin:mnist"

    summary = read_summary(paths)
    assert summary["status"] == "running"
    assert summary["run_id"] == "test_run"


def test_append_metric_and_finalize(tmp_path):
    cfg = ExperimentConfig(name="x", train="builtin:mnist")
    paths = create_run(cfg, root=tmp_path, run_id="m1")

    append_metric(paths, step=1, loss=0.5)
    append_metric(paths, step=2, loss=0.25, accuracy=0.8)

    rows = read_metrics(paths)
    assert rows == [
        {"step": 1, "loss": 0.5},
        {"step": 2, "loss": 0.25, "accuracy": 0.8},
    ]

    summary = finalize_run(
        paths,
        status="completed",
        result={"final_loss": 0.25},
        extra={"epochs": 2},
    )
    assert summary["status"] == "completed"
    assert summary["result"]["final_loss"] == 0.25
    assert summary["epochs"] == 2
    assert read_summary(paths)["status"] == "completed"


def test_list_and_get_run(tmp_path):
    cfg = ExperimentConfig(name="x", train="builtin:mnist")
    create_run(cfg, root=tmp_path, run_id="a")
    create_run(cfg, root=tmp_path, run_id="b")

    assert list_runs(tmp_path) == ["a", "b"]
    assert get_run("a", root=tmp_path).run_id == "a"


def test_duplicate_run_raises(tmp_path):
    cfg = ExperimentConfig(name="x", train="builtin:mnist")
    create_run(cfg, root=tmp_path, run_id="dup")
    with pytest.raises(FileExistsError):
        create_run(cfg, root=tmp_path, run_id="dup")


def test_append_metric_requires_payload(tmp_path):
    cfg = ExperimentConfig(name="x", train="builtin:mnist")
    paths = create_run(cfg, root=tmp_path, run_id="empty")
    with pytest.raises(ValueError):
        append_metric(paths)