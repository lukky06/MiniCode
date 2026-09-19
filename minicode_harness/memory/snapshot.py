"""Immutable per-Run Memory V3 snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Literal

from pydantic import BaseModel, Field

from .types import utc_now


MEMORY_SNAPSHOT_METADATA = "memory-snapshot.json"
MEMORY_SNAPSHOT_DIR = "memory-snapshot"
MEMORY_SNAPSHOT_SUMMARY = "memory_summary.md"
MEMORY_SNAPSHOT_HANDBOOK = "MEMORY.md"
MEMORY_SNAPSHOT_ROLLOUTS = "rollout_summaries"


class MemorySnapshot(BaseModel):
    version: Literal[3] = 3
    repository_id: str
    index_hash: str
    memory_summary_hash: str
    memory_hash: str
    rollout_summary_hashes: dict[str, str] = Field(default_factory=dict)
    rollout_summary_files: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class MemorySnapshotStore:
    """Persist the immutable Memory view visible to one Run."""

    def __init__(self, run_path: Path | str) -> None:
        self.run_path = Path(run_path)
        self.metadata_path = self.run_path / MEMORY_SNAPSHOT_METADATA
        self.snapshot_dir = self.run_path / MEMORY_SNAPSHOT_DIR
        self.summary_path = self.snapshot_dir / MEMORY_SNAPSHOT_SUMMARY
        self.memory_path = self.snapshot_dir / MEMORY_SNAPSHOT_HANDBOOK
        self.rollout_dir = self.snapshot_dir / MEMORY_SNAPSHOT_ROLLOUTS

    def save(
        self,
        *,
        repository_id: str,
        memory_summary: str,
        memory_md: str,
        rollout_summary_files: dict[str, str] | None = None,
    ) -> MemorySnapshot:
        rollouts = dict(sorted((rollout_summary_files or {}).items()))
        _validate_rollout_names(rollouts)
        hashes = {
            name: _content_hash(content)
            for name, content in rollouts.items()
        }
        snapshot = MemorySnapshot(
            repository_id=repository_id,
            index_hash=_snapshot_hash(memory_summary, memory_md, hashes),
            memory_summary_hash=_content_hash(memory_summary),
            memory_hash=_content_hash(memory_md),
            rollout_summary_hashes=hashes,
            rollout_summary_files=list(rollouts),
        )

        self.run_path.mkdir(parents=True, exist_ok=True)
        if self.snapshot_dir.exists():
            shutil.rmtree(self.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.summary_path, memory_summary)
        _atomic_write_text(self.memory_path, memory_md)
        if rollouts:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            for name, content in rollouts.items():
                _atomic_write_text(self.rollout_dir / name, content)
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
        summary = self.read_summary()
        memory_md = self.read_memory()
        if _content_hash(summary) != snapshot.memory_summary_hash:
            raise ValueError("Memory snapshot summary hash mismatch.")
        if _content_hash(memory_md) != snapshot.memory_hash:
            raise ValueError("Memory snapshot handbook hash mismatch.")

        actual_rollouts = (
            {
                path.name: _content_hash(path.read_text(encoding="utf-8"))
                for path in sorted(self.rollout_dir.glob("*.md"))
                if path.is_file()
            }
            if self.rollout_dir.is_dir()
            else {}
        )
        if actual_rollouts != snapshot.rollout_summary_hashes:
            raise ValueError("Memory snapshot rollout summaries do not match metadata.")
        if list(actual_rollouts) != snapshot.rollout_summary_files:
            raise ValueError("Memory snapshot rollout summary order does not match metadata.")

        actual_snapshot_hash = _snapshot_hash(
            summary,
            memory_md,
            actual_rollouts,
        )
        if actual_snapshot_hash != snapshot.index_hash:
            raise ValueError("Memory snapshot hash mismatch.")
        if expected_hash is not None and expected_hash != snapshot.index_hash:
            raise ValueError("Memory snapshot hash does not match the checkpoint.")
        return snapshot

    def read_summary(self) -> str:
        if not self.summary_path.is_file():
            return ""
        return self.summary_path.read_text(encoding="utf-8")

    def read_memory(self) -> str:
        if not self.memory_path.is_file():
            return ""
        return self.memory_path.read_text(encoding="utf-8")

    def read_rollout_summary(self, filename: str) -> str:
        snapshot = self.load()
        if snapshot is None or filename not in snapshot.rollout_summary_files:
            raise FileNotFoundError(
                f"Rollout summary is not present in the Run snapshot: {filename}"
            )
        path = self.rollout_dir / filename
        return path.read_text(encoding="utf-8")

    @property
    def checkpoint_path(self) -> str:
        return MEMORY_SNAPSHOT_METADATA


def _validate_rollout_names(rollouts: dict[str, str]) -> None:
    for name in rollouts:
        path = Path(name)
        if path.name != name or path.suffix != ".md":
            raise ValueError(f"Invalid rollout summary filename: {name}")


def _snapshot_hash(
    memory_summary: str,
    memory_md: str,
    rollout_hashes: dict[str, str],
) -> str:
    rendered = json.dumps(
        {
            "memory_summary_hash": _content_hash(memory_summary),
            "memory_hash": _content_hash(memory_md),
            "rollout_summary_hashes": rollout_hashes,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _content_hash(rendered)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
