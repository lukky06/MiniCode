"""Append-only audit events for Repository Memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import utc_now


class RepositoryMemoryEventStore:
    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir)
        self.path = self.memory_dir / "events.jsonl"

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "created_at": utc_now(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def list(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def processed_candidate_ids(self) -> set[str]:
        processed: set[str] = set()
        for event in self.list():
            if event.get("event") not in {
                "candidate_processed",
                "candidate_rejected",
            }:
                continue
            processed.update(str(value) for value in event.get("candidate_ids") or [])
        return processed
