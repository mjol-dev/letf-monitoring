from letf.config import AwoConfig, ExperimentConfig
from letf.context import TrainContext, TrainResult
from letf.comparator import compare_runs
from letf.runner import run_experiment


def _fake(ctx: TrainContext) -> TrainResult:
    ctx.log_metric(step=1, loss=0.5)
    return TrainResult(metrics={"final_loss": 0.5})


def test_compare_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("letf.runner.resolve_train", lambda t: _fake)
    cfg = ExperimentConfig(name="c", train="builtin:mnist", awo=AwoConfig(enabled=False))
    run_experiment(cfg, root=tmp_path, run_id="c1")
    run_experiment(cfg, root=tmp_path, run_id="c2")
    rows = compare_runs(["c1", "c2"], root=tmp_path)
    assert len(rows) == 2
    assert {r["run_id"] for r in rows} == {"c1", "c2"}