"""Builtin train target registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from letf.context import TrainContext, TrainResult
from letf.trains import mnist

TrainFn = Callable[[TrainContext], TrainResult]

BUILTIN: dict[str, TrainFn] = {
    "mnist": mnist.train,
}


def resolve_train(target: str) -> TrainFn:
    """Resolve a train target string to a train plugin callable."""
    target = target.strip()
    if not target:
        raise ValueError("train target must be a non-empty string")

    if target.startswith("builtin:"):
        name = target.split(":", 1)[1]
        try:
            return BUILTIN[name]
        except KeyError as exc:
            known = ", ".join(sorted(BUILTIN)) or "(none)"
            raise ValueError(
                f"Unknown builtin train target {target!r}. Known: {known}"
            ) from exc

    if target.startswith("module:"):
        # module:pkg.mod:attr
        rest = target[len("module:") :]
        if ":" not in rest:
            raise ValueError(
                "BYO target must look like 'module:package.module:callable'"
            )
        module_path, attr = rest.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        try:
            fn = getattr(mod, attr)
        except AttributeError as exc:
            raise ValueError(
                f"Module {module_path!r} has no attribute {attr!r}"
            ) from exc
        if not callable(fn):
            raise ValueError(f"Train attribute {attr!r} is not callable")
        return fn  # type: ignore[return-value]

    raise ValueError(
        f"Unsupported train target {target!r}. "
        "Use 'builtin:<name>' or 'module:pkg.mod:callable'."
    )