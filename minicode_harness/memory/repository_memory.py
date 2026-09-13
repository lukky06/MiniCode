"""Facade over one repository's durable memory stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minicode_harness.storage import default_data_dir

from .events import RepositoryMemoryEventStore
from .index_store import RepositoryMemoryIndexStore
from .repository_id import RepositoryIdentity, resolve_repository_identity
from .topic_store import RepositoryMemoryTopicStore
from .types import MemoryTopicName
from .workflow import RepositoryMemoryWorkflowStore

@dataclass(frozen=True)
class RepositoryMemorySnapshotSource:
    rendered_index: str
    topic_payloads: dict[MemoryTopicName, dict[str, str]]


class RepositoryMemoryStore:
    """Resolve repository identity once and expose bounded memory stores."""

    def __init__(
        self,
        workspace: Path | str,
        data_dir: Path | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.identity: RepositoryIdentity = resolve_repository_identity(self.workspace)
        self.memory_dir = self.identity.memory_dir(self.data_dir)
        self.event_store = RepositoryMemoryEventStore(self.memory_dir)
        self.workflow_store = RepositoryMemoryWorkflowStore(
            self.memory_dir,
            event_store=self.event_store,
        )
        self.topic_store = RepositoryMemoryTopicStore(
            self.memory_dir,
            event_store=self.event_store,
        )
        self.index_store = RepositoryMemoryIndexStore(
            self.memory_dir,
            topic_store=self.topic_store,
            event_store=self.event_store,
        )

    @property
    def repository_id(self) -> str:
        return self.identity.repository_id

    def render_index(self, *, include_read_instructions: bool = True) -> str:
        return self.index_store.render(
            include_read_instructions=include_read_instructions,
        )

    def refresh_index(self) -> str:
        return self.index_store.refresh()

    def snapshot_topics(self) -> dict[MemoryTopicName, dict[str, str]]:
        return {
            topic: self.read_topic(topic)
            for topic in self.index_store.registered_topics()
        }

    def capture_snapshot_source(
        self,
        *,
        max_attempts: int = 3,
    ) -> RepositoryMemorySnapshotSource:
        for _ in range(max(1, max_attempts)):
            rendered_index = self.index_store.ensure().content
            topics = tuple(self.index_store.registered_topics())
            topic_payloads = {topic: self.read_topic(topic) for topic in topics}
            if self.index_store.ensure().content != rendered_index:
                continue
            if tuple(self.index_store.registered_topics()) != topics:
                continue
            if any(self.read_topic(topic) != payload for topic, payload in topic_payloads.items()):
                continue
            return RepositoryMemorySnapshotSource(
                rendered_index=rendered_index,
                topic_payloads=topic_payloads,
            )
        raise RuntimeError("Repository Memory changed while the Run snapshot was captured.")

    def read_topic(self, topic: MemoryTopicName | str) -> dict[str, str]:
        normalized = self.topic_store._validate_topic(topic)
        if normalized not in self.index_store.registered_topics():
            raise FileNotFoundError(f"Memory topic is not registered in the index: {normalized}")
        document = self.topic_store.read(normalized)
        active_entries = [
            entry for entry in document.entries if entry.status == "active"
        ]
        lines = [
            f"Repository Memory Requirements ({normalized}):",
            "Treat every listed requirement relevant to the current task as an acceptance criterion.",
            *[f"- {entry.summary}" for entry in active_entries],
        ]
        content = "\n".join(lines) + "\n"
        return {
            "topic": normalized,
            "content": content,
            "content_hash": self.index_store.content_hash(content),
            "updated_at": document.updated_at,
        }
