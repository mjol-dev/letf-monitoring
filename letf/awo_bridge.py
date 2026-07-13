"""In-process AWO metrics collection bridge."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from awo.collector import collect_all_metrics


class AwoBridge:
    """Poll system/GPU metrics into a run-local JSONL file."""

    def __init__(self, log_path: Path, tag: str, interval: float = 5.0) -> None:
        self.log_path = Path(log_path)
        self.tag = tag
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="awo-bridge", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            metrics = collect_all_metrics()
            entry = {
                "timestamp": datetime.now().isoformat(),
                "tag": self.tag,
                **metrics,
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._stop.wait(self.interval)