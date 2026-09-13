"""Deterministic publication of approved or explicitly requested memory."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .repository_memory import RepositoryMemoryStore
from .types import MemoryCandidateType, MemoryTopicName


class MemoryPublishResult(BaseModel):
    status: str
    topic: MemoryTopicName
    entry_id: str = Field(min_length=1, max_length=160)
    index_status: str
    topic_size_bytes: int = Field(ge=0)
    topic_limit_bytes: int = Field(gt=0)
    capacity_warning: str | None = None


class RepositoryMemoryPublisher:
    """Publish one exact ADD without any model call or proposal state."""

    def __init__(self, store: RepositoryMemoryStore) -> None:
        self.store = store

    def publish(
        self,
        *,
        topic: MemoryTopicName,
        text: str,
        entry_id: str | None = None,
        evidence_ids: list[str] | None = None,
    ) -> MemoryPublishResult:
        normalized = " ".join(text.split())
        duplicate = next(
            (
                item
                for item in self.store.topic_store.active_entries(topic)
                if " ".join(item.summary.split()).casefold()
                == normalized.casefold()
            ),
            None,
        )
        if duplicate is not None:
            ensured = self.store.index_store.ensure()
            capacity = self.store.topic_store.capacity_status(topic)
            return MemoryPublishResult(
                status="already_exists",
                topic=topic,
                entry_id=duplicate.entry_id,
                index_status=ensured.status,
                topic_size_bytes=capacity.size_bytes,
                topic_limit_bytes=capacity.limit_bytes,
                capacity_warning=capacity.warning,
            )

        entry = self.store.topic_store.add_entry(
            topic=topic,
            entry_type=_entry_type_for_topic(topic),
            summary=normalized,
            evidence_ids=sorted(set(evidence_ids or [])),
            entry_id=entry_id,
        )
        ensured = self.store.index_store.ensure()
        capacity = self.store.topic_store.capacity_status(topic)
        self.store.event_store.append(
            "memory_entry_added",
            topic=topic,
            entry_id=entry.entry_id,
            topic_size_bytes=capacity.size_bytes,
            topic_limit_bytes=capacity.limit_bytes,
            capacity_warning=capacity.warning,
        )
        return MemoryPublishResult(
            status="saved",
            topic=topic,
            entry_id=entry.entry_id,
            index_status=ensured.status,
            topic_size_bytes=capacity.size_bytes,
            topic_limit_bytes=capacity.limit_bytes,
            capacity_warning=capacity.warning,
        )


def _entry_type_for_topic(topic: MemoryTopicName) -> MemoryCandidateType:
    return {
        "instructions": "user_instruction",
        "build-and-test": "procedure",
        "debugging": "pitfall",
        "decisions": "decision",
        "environment": "environment",
    }[topic]  # type: ignore[return-value]
