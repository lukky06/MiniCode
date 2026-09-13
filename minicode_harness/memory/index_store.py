"""Small always-on Memory Index rendered into the System Prefix."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .events import RepositoryMemoryEventStore
from .topic_store import RepositoryMemoryTopicStore
from .types import MemoryTopicName, TOPIC_NAMES

MAX_MEMORY_INDEX_BYTES = 4 * 1024
_TOPIC_DESCRIPTIONS: dict[str, str] = {
    "instructions": "stable user and repository operating instructions",
    "build-and-test": "verified build commands and focused test procedures",
    "debugging": "verified failure-recovery paths and recurring pitfalls",
    "decisions": "confirmed architecture and implementation decisions",
    "environment": "verified environment and toolchain facts",
}


@dataclass(frozen=True)
class IndexEnsureResult:
    status: str
    content: str


class RepositoryMemoryIndexStore:
    def __init__(
        self,
        memory_dir: Path | str,
        *,
        topic_store: RepositoryMemoryTopicStore | None = None,
        event_store: RepositoryMemoryEventStore | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.path = self.memory_dir / "MEMORY.md"
        self.event_store = event_store or RepositoryMemoryEventStore(self.memory_dir)
        self.topic_store = topic_store or RepositoryMemoryTopicStore(
            self.memory_dir,
            event_store=self.event_store,
        )

    def render(self, *, include_read_instructions: bool = True) -> str:
        if not self.path.is_file():
            return ""
        content = self.path.read_text(encoding="utf-8")
        if not include_read_instructions:
            content = _without_read_instructions(content)
        encoded = content.encode("utf-8")
        if len(encoded) <= MAX_MEMORY_INDEX_BYTES:
            return content.strip()
        return encoded[:MAX_MEMORY_INDEX_BYTES].decode("utf-8", errors="ignore").rstrip()

    def ensure(self) -> IndexEnsureResult:
        content = self._render_expected()
        current = (
            self.path.read_text(encoding="utf-8")
            if self.path.is_file()
            else None
        )
        if current == content:
            return IndexEnsureResult(status="unchanged", content=content.strip())

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".md.tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(self.path)
        status = "created" if current is None else "repaired"
        self.event_store.append(
            "memory_index_refreshed",
            status=status,
            registered_topics=list(self.registered_topics()),
            size_bytes=len(content.encode("utf-8")),
            index_hash=self.content_hash(content),
        )
        return IndexEnsureResult(status=status, content=content.strip())

    def refresh(self) -> str:
        """Regenerate the bounded index and return its content."""

        return self.ensure().content

    def _render_expected(self) -> str:
        lines = ["Repository Memory Index:"]
        for topic in TOPIC_NAMES:
            entries = self.topic_store.active_entries(topic)
            if not entries:
                continue
            description = _TOPIC_DESCRIPTIONS[str(topic)]
            lines.append(
                f'- {topic}: {description}; {len(entries)} active '
                f'entr{"y" if len(entries) == 1 else "ies"}. '
                f'Use read(source="memory", target="{topic}") for details.'
            )
        content = "" if len(lines) == 1 else "\n".join(lines) + "\n"
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_MEMORY_INDEX_BYTES:
            raise ValueError(
                f"Memory index exceeds {MAX_MEMORY_INDEX_BYTES} bytes: {len(encoded)}."
            )
        return content

    def registered_topics(self) -> list[MemoryTopicName]:
        return self.topic_store.registered_topics()

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _without_read_instructions(content: str) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        prefix, marker, _ = line.partition(" Use read(source=\"memory\",")
        lines.append(prefix.rstrip() if marker else line)
    return "\n".join(lines)
