from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any


class SessionWriter:
    def __init__(self, root: str | Path = "sessions", buffer_seconds: float = 3.0, fps: int = 30):
        self.root = Path(root)
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self.path = self.root / self.session_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.buffer: deque[dict[str, Any]] = deque(maxlen=max(1, int(buffer_seconds * fps)))
        self.events_path = self.path / "events.jsonl"
        self.metrics: dict[str, Any] = {"frames": 0, "events": 0}
        (self.path / "metadata.json").write_text(
            json.dumps({"session_id": self.session_id, "created_unix": time.time()}, indent=2),
            encoding="utf-8",
        )

    def observe_frame(self, frame: dict[str, Any]) -> None:
        self.metrics["frames"] += 1
        self.buffer.append(frame)

    def record_event(self, name: str, payload: dict[str, Any]) -> None:
        self.metrics["events"] += 1
        event = {"event": name, "payload": payload, "pre_context": list(self.buffer)}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def close(self) -> None:
        (self.path / "metrics.json").write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")
