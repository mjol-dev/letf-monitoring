import pytest

from letf.trains import resolve_train
from letf.trains import mnist


def test_resolve_builtin_mnist():
    fn = resolve_train("builtin:mnist")
    assert fn is mnist.train


def test_resolve_unknown_builtin():
    with pytest.raises(ValueError, match="Unknown builtin"):
        resolve_train("builtin:nope")


def test_resolve_unsupported_scheme():
    with pytest.raises(ValueError, match="Unsupported train target"):
        resolve_train("other:thing")


def test_resolve_byo_module():
    fn = resolve_train("module:letf.trains.mnist:train")
    assert fn is mnist.train