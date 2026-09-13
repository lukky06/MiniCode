"""Deterministic Memory Index injection for one Run snapshot."""

from __future__ import annotations

from typing import Any

from minicode_harness.trace import TraceWriter

from .repository_memory import RepositoryMemoryStore


class RepositoryMemoryIndexService:
    def __init__(self, *, trace_writer: TraceWriter | None = None) -> None:
        self.trace_writer = trace_writer

    def render(self, *, store: RepositoryMemoryStore) -> str:
        rendered = store.index_store.ensure().content
        return self.record(store=store, rendered=rendered)

    def record(
        self,
        *,
        store: RepositoryMemoryStore,
        rendered: str,
        registered_topics: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "repository_id": store.repository_id,
            "index_hash": store.index_store.content_hash(rendered),
            "index_chars": len(rendered),
            "registered_topics": (
                list(registered_topics)
                if registered_topics is not None
                else list(store.index_store.registered_topics())
            ),
            "model_call_count": 0,
        }
        if self.trace_writer is not None:
            self.trace_writer.write_event("memory_index_injected", **payload)
        try:
            store.event_store.append("memory_index_injected", **payload)
        except OSError as exc:
            if self.trace_writer is not None:
                self.trace_writer.write_event(
                    "memory_event_log_failed",
                    event="memory_index_injected",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        return rendered
