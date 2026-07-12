from pathlib import Path

import pytest

from letf.config import load_config, parse_config


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "mnist.yaml"


def test_load_mnist_example():
    cfg = load_config(EXAMPLES)
    assert cfg.name == "mnist-baseline"
    assert cfg.train == "builtin:mnist"
    assert cfg.seed == 42
    assert cfg.hparams["epochs"] == 1
    assert cfg.hparams["batch_size"] == 64
    assert cfg.hparams["lr"] == 0.01
    assert cfg.awo.enabled is True
    assert cfg.awo.interval == 5


def test_parse_defaults():
    cfg = parse_config({"name": "x", "train": "builtin:mnist"})
    assert cfg.seed == 42
    assert cfg.hparams == {}
    assert cfg.awo.enabled is True
    assert cfg.awo.interval == 5


def test_missing_required_keys():
    with pytest.raises(ValueError, match="Missing required keys"):
        parse_config({"name": "x"})


def test_invalid_awo_interval():
    with pytest.raises(ValueError, match="awo.interval"):
        parse_config(
            {
                "name": "x",
                "train": "builtin:mnist",
                "awo": {"interval": 0},
            }
        )


def test_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist.yaml")