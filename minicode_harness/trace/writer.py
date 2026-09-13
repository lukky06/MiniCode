"""JSONL execution trace writer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any


class TraceWriter:
    """Append execution trace events to a JSONL file."""

    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self._lock = Lock()

    def write_event(
        self,
        event_type: str,
        *,
        step: int | None = None,
        time: datetime | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Append one trace event and return the serialized event."""

        if not event_type:
            raise ValueError("Trace event type must not be empty.")

        event_time = time or datetime.now(timezone.utc)
        event: dict[str, Any] = {
            "type": event_type,
            "time": event_time.isoformat(),
        }
        if step is not None:
            event["step"] = step
        event.update(payload)

        with self._lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        return event
