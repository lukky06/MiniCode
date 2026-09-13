"""Atomic Markdown Topic storage for Repository Memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml

from .events import RepositoryMemoryEventStore
from .types import (
    MemoryCandidateType,
    MemoryTopicDocument,
    MemoryTopicEntry,
    MemoryTopicName,
    TOPIC_NAMES,
    utc_now,
)

MAX_TOPIC_BYTES = 8 * 1024
TOPIC_CAPACITY_WARNING_RATIO = 0.80
_FRONT_MATTER = "---"
_TOPIC_TITLES: dict[str, str] = {
    "instructions": "Repository Instructions",
    "build-and-test": "Build and Test",
    "debugging": "Debugging",
    "decisions": "Architecture Decisions",
    "environment": "Environment",
}


@dataclass(frozen=True)
class TopicCapacityStatus:
    topic: MemoryTopicName
    size_bytes: int
    limit_bytes: int = MAX_TOPIC_BYTES

    @property
    def ratio(self) -> float:
        return self.size_bytes / self.limit_bytes

    @property
    def near_capacity(self) -> bool:
        return self.ratio >= TOPIC_CAPACITY_WARNING_RATIO

    @property
    def warning(self) -> str | None:
        if not self.near_capacity:
            return None
        percent = round(self.ratio * 100)
        return (
            f"Memory Topic {self.topic} uses {self.size_bytes}/{self.limit_bytes} bytes "
            f"({percent}%); review and forget stale entries before the next write."
        )


class RepositoryMemoryTopicStore:
    def __init__(
        self,
        memory_dir: Path | str,
        *,
        event_store: RepositoryMemoryEventStore | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.topics_dir = self.memory_dir / "topics"
        self.event_store = event_store or RepositoryMemoryEventStore(self.memory_dir)

    def path_for(self, topic: MemoryTopicName | str) -> Path:
        normalized = self._validate_topic(topic)
        return self.topics_dir / f"{normalized}.md"

    def read(self, topic: MemoryTopicName | str) -> MemoryTopicDocument:
        normalized = self._validate_topic(topic)
        path = self.path_for(normalized)
        if not path.is_file():
            return MemoryTopicDocument(topic=normalized)
        text = path.read_text(encoding="utf-8")
        payload = _parse_front_matter(text)
        payload.setdefault("topic", normalized)
        return MemoryTopicDocument.model_validate(payload)

    def write(self, document: MemoryTopicDocument) -> Path:
        path = self.path_for(document.topic)
        rendered = _render_document(document)
        size = len(rendered.encode("utf-8"))
        if size > MAX_TOPIC_BYTES:
            raise ValueError(
                f"Memory topic {document.topic} exceeds {MAX_TOPIC_BYTES} bytes: {size}."
            )
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(rendered, encoding="utf-8")
        temp.replace(path)
        self.event_store.append(
            "topic_written",
            topic=document.topic,
            entry_count=len(document.entries),
            active_count=sum(1 for item in document.entries if item.status == "active"),
            size_bytes=size,
        )
        return path

    def registered_topics(self) -> list[MemoryTopicName]:
        registered: list[MemoryTopicName] = []
        for topic in TOPIC_NAMES:
            document = self.read(topic)
            if any(item.status == "active" for item in document.entries):
                registered.append(topic)
        return registered

    def active_entries(self, topic: MemoryTopicName | str) -> list[MemoryTopicEntry]:
        return [item for item in self.read(topic).entries if item.status == "active"]

    def capacity_status(self, topic: MemoryTopicName | str) -> TopicCapacityStatus:
        normalized = self._validate_topic(topic)
        rendered = _render_document(self.read(normalized))
        return TopicCapacityStatus(
            topic=normalized,
            size_bytes=len(rendered.encode("utf-8")),
        )

    def get_entry(
        self,
        topic: MemoryTopicName | str,
        entry_id: str,
    ) -> MemoryTopicEntry | None:
        return next(
            (item for item in self.read(topic).entries if item.entry_id == entry_id),
            None,
        )

    def add_entry(
        self,
        *,
        topic: MemoryTopicName,
        entry_type: MemoryCandidateType,
        summary: str,
        evidence_ids: list[str],
        entry_id: str | None = None,
    ) -> MemoryTopicEntry:
        document = self.read(topic)
        normalized_summary = " ".join(summary.split())
        duplicate = next(
            (
                item
                for item in document.entries
                if item.status == "active"
                and item.type == entry_type
                and item.summary.casefold() == normalized_summary.casefold()
            ),
            None,
        )
        if duplicate is not None:
            return duplicate
        now = utc_now()
        entry = MemoryTopicEntry(
            entry_id=entry_id or f"memory_{uuid4().hex[:20]}",
            type=entry_type,
            summary=normalized_summary,
            evidence_ids=sorted(set(evidence_ids)),
            created_at=now,
            updated_at=now,
        )
        document.entries.append(entry)
        document.updated_at = now
        self.write(document)
        return entry

    def update_entry(
        self,
        *,
        topic: MemoryTopicName,
        entry_id: str,
        summary: str,
        evidence_ids: list[str],
    ) -> MemoryTopicEntry:
        document = self.read(topic)
        for index, item in enumerate(document.entries):
            if item.entry_id != entry_id:
                continue
            updated = item.model_copy(
                update={
                    "summary": " ".join(summary.split()),
                    "evidence_ids": sorted(set([*item.evidence_ids, *evidence_ids])),
                    "status": "active",
                    "updated_at": utc_now(),
                }
            )
            document.entries[index] = updated
            document.updated_at = updated.updated_at
            self.write(document)
            return updated
        raise KeyError(f"Memory topic entry not found: {entry_id}")

    def deactivate_entry(
        self,
        *,
        topic: MemoryTopicName,
        entry_id: str,
    ) -> MemoryTopicEntry:
        document = self.read(topic)
        for index, item in enumerate(document.entries):
            if item.entry_id != entry_id:
                continue
            if item.status == "inactive":
                return item
            updated = item.model_copy(update={"status": "inactive", "updated_at": utc_now()})
            document.entries[index] = updated
            document.updated_at = updated.updated_at
            self.write(document)
            return updated
        raise KeyError(f"Memory topic entry not found: {entry_id}")

    @staticmethod
    def _validate_topic(topic: MemoryTopicName | str) -> MemoryTopicName:
        normalized = str(topic).strip()
        if normalized not in TOPIC_NAMES:
            raise ValueError(f"Unknown memory topic: {topic}")
        return normalized  # type: ignore[return-value]


def _parse_front_matter(text: str) -> dict[str, object]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != _FRONT_MATTER:
        raise ValueError("Memory topic is missing structured front matter.")
    try:
        closing = lines[1:].index(_FRONT_MATTER) + 1
    except ValueError as exc:
        raise ValueError("Memory topic front matter is not closed.") from exc
    payload = yaml.safe_load("\n".join(lines[1:closing])) or {}
    if not isinstance(payload, dict):
        raise ValueError("Memory topic front matter must be a mapping.")
    return payload


def _render_document(document: MemoryTopicDocument) -> str:
    metadata = document.model_dump(mode="json")
    front_matter = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    title = _TOPIC_TITLES[str(document.topic)]
    active = [item for item in document.entries if item.status == "active"]
    body = [f"# {title}", ""]
    if not active:
        body.append("No active repository memory entries.")
    else:
        for item in active:
            body.append(f"- **{item.type}** `{item.entry_id}`: {item.summary}")
    return f"{_FRONT_MATTER}\n{front_matter}\n{_FRONT_MATTER}\n\n" + "\n".join(body) + "\n"
