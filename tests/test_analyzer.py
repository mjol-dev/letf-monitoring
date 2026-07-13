from letf.config import AwoConfig, ExperimentConfig
from letf.context import TrainContext, TrainResult
from letf.analyzer import analyze_run
from letf.runner import run_experiment


def _fake(ctx: TrainContext) -> TrainResult:
    ctx.log_metric(step=1, loss=0.9)
    ctx.log_metric(step=2, loss=0.4)
    return TrainResult(metrics={"final_loss": 0.4})


def test_analyze_run(tmp_path, monkeypatch):
    monkeypatch.setattr("letf.runner.resolve_train", lambda t: _fake)
    cfg = ExperimentConfig(
        name="a",
        train="builtin:mnist",
        awo=AwoConfig(enabled=False),
    )
    paths = run_experiment(cfg, root=tmp_path, run_id="ana1")
    data = analyze_run("ana1", root=tmp_path)
    assert data["status"] == "completed"
    assert data["metric_points"] == 2
    assert data["loss_last"] == 0.4
    assert data["loss_min"] == 0.4