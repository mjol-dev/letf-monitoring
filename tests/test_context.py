from letf.config import ExperimentConfig
from letf.context import TrainContext, TrainResult
from letf.tracker import create_run, read_metrics


def test_log_metric_writes_jsonl(tmp_path):
    cfg = ExperimentConfig(name="ctx", train="builtin:mnist", hparams={"lr": 0.01})
    paths = create_run(cfg, root=tmp_path, run_id="ctx1")
    ctx = TrainContext(config=cfg, paths=paths, device="cpu")

    assert ctx.run_id == "ctx1"
    assert ctx.hparams["lr"] == 0.01

    ctx.log_metric(step=1, loss=0.9)
    ctx.log_metric(step=2, loss=0.4, accuracy=0.7)

    assert read_metrics(paths) == [
        {"step": 1, "loss": 0.9},
        {"step": 2, "loss": 0.4, "accuracy": 0.7},
    ]


def test_train_result_defaults():
    result = TrainResult()
    assert result.metrics == {}
    assert result.artifacts == {}

    result = TrainResult(metrics={"loss": 0.1}, artifacts={"model": "model.pt"})
    assert result.metrics["loss"] == 0.1
    assert result.artifacts["model"] == "model.pt"