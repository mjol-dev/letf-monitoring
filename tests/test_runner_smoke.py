from letf.config import ExperimentConfig
from letf.context import TrainContext, TrainResult
from letf.runner import run_experiment
from letf.tracker import read_metrics, read_summary


def _fake_train(ctx: TrainContext) -> TrainResult:
    ctx.log_metric(step=1, loss=0.5)
    return TrainResult(metrics={"final_loss": 0.5})


def test_run_experiment_with_fake_train(tmp_path, monkeypatch):
    monkeypatch.setattr("letf.runner.resolve_train", lambda target: _fake_train)

    cfg = ExperimentConfig(name="smoke", train="builtin:mnist")
    paths = run_experiment(cfg, root=tmp_path, device="cpu", run_id="smoke1")

    assert paths.run_id == "smoke1"
    assert (paths.run_dir / "config.snapshot.yaml").is_file()
    assert read_metrics(paths) == [{"step": 1, "loss": 0.5}]

    summary = read_summary(paths)
    assert summary["status"] == "completed"
    assert summary["result"]["metrics"]["final_loss"] == 0.5
    assert "duration_sec" in summary


def test_run_experiment_marks_failure(tmp_path, monkeypatch):
    def _boom(ctx: TrainContext) -> TrainResult:
        raise RuntimeError("boom")

    monkeypatch.setattr("letf.runner.resolve_train", lambda target: _boom)

    cfg = ExperimentConfig(name="fail", train="builtin:mnist")
    try:
        run_experiment(cfg, root=tmp_path, run_id="fail1")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    from letf.tracker import get_run

    summary = read_summary(get_run("fail1", root=tmp_path))
    assert summary["status"] == "failed"
    assert "boom" in summary["error"]

def test_awo_bridge_writes_log(tmp_path):
    from letf.awo_bridge import AwoBridge
    import time

    log_path = tmp_path / "awo_log.jsonl"
    bridge = AwoBridge(log_path=log_path, tag="t1", interval=0.05)
    bridge.start()
    time.sleep(0.2)
    bridge.stop()

    assert log_path.is_file()
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1
    import json
    row = json.loads(lines[0])
    assert row["tag"] == "t1"
    assert "cpu_percent" in row