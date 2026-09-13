"""Immutable per-Run Repository Memory snapshots used by normal execution and Resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Literal

from pydantic import BaseModel, Field

from .types import TOPIC_NAMES, MemoryTopicName, utc_now

MEMORY_SNAPSHOT_MARKDOWN = "memory-snapshot.md"
MEMORY_SNAPSHOT_METADATA = "memory-snapshot.json"
MEMORY_TOPIC_SNAPSHOT_DIR = "memory-topic-snapshots"


class MemorySnapshot(BaseModel):
    version: Literal[2] = 2
    repository_id: str
    index_hash: str
    rendered_index: str = ""
    topic_hashes: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class MemoryTopicSnapshot(BaseModel):
    topic: MemoryTopicName
    content_hash: str
    payload: dict[str, str]
    created_at: str = Field(default_factory=utc_now)


class MemorySnapshotStore:
    def __init__(self, run_path: Path | str) -> None:
        self.run_path = Path(run_path)
        self.markdown_path = self.run_path / MEMORY_SNAPSHOT_MARKDOWN
        self.metadata_path = self.run_path / MEMORY_SNAPSHOT_METADATA
        self.topic_dir = self.run_path / MEMORY_TOPIC_SNAPSHOT_DIR

    def save(
        self,
        *,
        repository_id: str,
        rendered_index: str,
        topic_payloads: dict[MemoryTopicName | str, dict[str, str]] | None = None,
    ) -> MemorySnapshot:
        topic_snapshots: dict[MemoryTopicName, MemoryTopicSnapshot] = {}
        for topic, payload in sorted((topic_payloads or {}).items()):
            normalized = _normalize_topic(topic)
            if payload.get("topic") != normalized:
                raise ValueError(
                    f"Memory Topic payload name does not match snapshot key: {normalized}."
                )
            topic_snapshots[normalized] = MemoryTopicSnapshot(
                topic=normalized,
                content_hash=_payload_hash(payload),
                payload=dict(payload),
            )

        snapshot = MemorySnapshot(
            version=2,
            repository_id=repository_id,
            index_hash=_content_hash(rendered_index),
            rendered_index=rendered_index,
            topic_hashes={
                topic: item.content_hash for topic, item in topic_snapshots.items()
            },
        )
        self.run_path.mkdir(parents=True, exist_ok=True)
        if self.topic_dir.exists():
            shutil.rmtree(self.topic_dir)
        if topic_snapshots:
            self.topic_dir.mkdir(parents=True, exist_ok=True)
            for item in topic_snapshots.values():
                self._write_topic_snapshot(item)
        _atomic_write_text(self.markdown_path, rendered_index)
        _atomic_write_text(
            self.metadata_path,
            json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
        )
        return snapshot

    def load(self, *, expected_hash: str | None = None) -> MemorySnapshot | None:
        if not self.metadata_path.is_file():
            return None
        snapshot = MemorySnapshot.model_validate_json(
            self.metadata_path.read_text(encoding="utf-8")
        )
        actual_hash = _content_hash(snapshot.rendered_index)
        if actual_hash != snapshot.index_hash:
            raise ValueError("Memory snapshot metadata hash does not match rendered_index.")
        if expected_hash is not None and expected_hash != snapshot.index_hash:
            raise ValueError("Memory snapshot hash does not match the checkpoint.")
        if self.markdown_path.is_file():
            markdown = self.markdown_path.read_text(encoding="utf-8")
            if _content_hash(markdown) != snapshot.index_hash:
                raise ValueError("Memory snapshot Markdown does not match the metadata hash.")
        expected_topics = set(snapshot.topic_hashes)
        actual_topics = (
            {path.stem for path in self.topic_dir.glob("*.json")}
            if self.topic_dir.is_dir()
            else set()
        )
        if actual_topics != expected_topics:
            raise ValueError(
                "Memory Topic snapshot files do not match snapshot metadata."
            )
        for topic, expected_topic_hash in snapshot.topic_hashes.items():
            topic_snapshot = self._load_topic_snapshot(_normalize_topic(topic))
            if topic_snapshot.content_hash != expected_topic_hash:
                raise ValueError(
                    f"Memory Topic snapshot hash does not match metadata: {topic}."
                )
        return snapshot

    def read_topic(self, topic: MemoryTopicName | str) -> dict[str, str]:
        return dict(self._load_topic_snapshot(_normalize_topic(topic)).payload)

    def _load_topic_snapshot(self, topic: MemoryTopicName) -> MemoryTopicSnapshot:
        path = self.topic_dir / f"{topic}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Memory Topic is not present in the Run snapshot: {topic}")
        snapshot = MemoryTopicSnapshot.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if snapshot.topic != topic:
            raise ValueError("Memory Topic snapshot name does not match its file.")
        if _payload_hash(snapshot.payload) != snapshot.content_hash:
            raise ValueError("Memory Topic snapshot payload hash does not match metadata.")
        return snapshot

    def _write_topic_snapshot(self, snapshot: MemoryTopicSnapshot) -> None:
        path = self.topic_dir / f"{snapshot.topic}.json"
        _atomic_write_text(
            path,
            json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
        )

    @property
    def checkpoint_path(self) -> str:
        return MEMORY_SNAPSHOT_METADATA


def _normalize_topic(topic: MemoryTopicName | str) -> MemoryTopicName:
    normalized = str(topic).strip()
    if normalized not in TOPIC_NAMES:
        raise ValueError(f"Unknown memory topic: {topic}")
    return normalized  # type: ignore[return-value]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict[str, str]) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _content_hash(rendered)


def _atomic_write_text(path: Path, content: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)
