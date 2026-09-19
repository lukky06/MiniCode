"""Repository-scoped Memory V3 persistence primitives."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from minicode_harness.storage import default_data_dir

from .events import RepositoryMemoryEventStore
from .repository_id import RepositoryIdentity, resolve_repository_identity
from .types import MemoryPipelineState, Stage1Record


@dataclass(frozen=True)
class RepositoryMemorySnapshotSource:
    memory_summary: str
    memory_md: str
    rollout_summary_files: dict[str, str]


class _MemoryFileLock:
    """One held non-blocking operating-system file lock."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        if os.name == "nt":
            import msvcrt

            self._stream.seek(0)
            msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._released = True

    def __enter__(self) -> "_MemoryFileLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class RepositoryMemoryStore:
    """Persist one repository's Memory V3 pipeline state."""

    def __init__(
        self,
        workspace: Path | str,
        data_dir: Path | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.identity: RepositoryIdentity = resolve_repository_identity(self.workspace)
        self.memory_dir = self.identity.memory_dir(self.data_dir)
        self.stage1_dir = self.memory_dir / "stage1"
        self.rollout_summaries_dir = self.memory_dir / "rollout_summaries"
        self.state_path = self.memory_dir / "state.json"
        self.memory_path = self.memory_dir / "MEMORY.md"
        self.summary_path = self.memory_dir / "memory_summary.md"
        self.raw_memories_path = self.memory_dir / "raw_memories.md"
        self.pipeline_lock_path = self.memory_dir / "pipeline.lock"
        self.durable_lock_path = self.memory_dir / "durable.lock"
        self.event_store = RepositoryMemoryEventStore(self.memory_dir)

    @property
    def repository_id(self) -> str:
        return self.identity.repository_id

    def load_state(self) -> MemoryPipelineState:
        if not self.state_path.is_file():
            return MemoryPipelineState()
        return MemoryPipelineState.model_validate_json(
            self.state_path.read_text(encoding="utf-8")
        )

    def save_state(self, state: MemoryPipelineState) -> None:
        _atomic_write_text(
            self.state_path,
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
        )

    def load_stage1(self, run_id: str) -> Stage1Record | None:
        path = self.stage1_path(run_id)
        if not path.is_file():
            return None
        return Stage1Record.model_validate_json(path.read_text(encoding="utf-8"))

    def stage1_path(self, run_id: str) -> Path:
        return self.stage1_dir / f"{run_id}.json"

    def write_stage1_no_output(self, run_id: str) -> Stage1Record:
        record = Stage1Record(run_id=run_id, status="no_output")
        self._write_new_stage1(record)
        return record

    def write_stage1_memory(
        self,
        *,
        run_id: str,
        raw_memory: str,
        rollout_summary: str,
        rollout_slug: str,
    ) -> Stage1Record:
        state = self.load_state()
        record = Stage1Record(
            run_id=run_id,
            status="memory",
            seq=state.latest_stage1_seq + 1,
            rollout_slug=rollout_slug.strip(),
            raw_memory=raw_memory.strip(),
            rollout_summary=rollout_summary.strip(),
        )
        self._write_new_stage1(record)
        self.save_state(
            state.model_copy(update={"latest_stage1_seq": record.seq})
        )
        return record

    def pending_stage1_records(self) -> list[Stage1Record]:
        state = self.load_state()
        records = [
            record
            for record in self.list_stage1_records()
            if record.status == "memory"
            and record.seq is not None
            and state.last_phase2_input_seq < record.seq <= state.latest_stage1_seq
        ]
        return sorted(records, key=lambda record: record.seq or 0)

    def read_memory(self) -> str:
        return (
            self.memory_path.read_text(encoding="utf-8")
            if self.memory_path.is_file()
            else ""
        )

    def read_memory_summary(self) -> str:
        return (
            self.summary_path.read_text(encoding="utf-8")
            if self.summary_path.is_file()
            else ""
        )

    def write_durable_memory(self, memory_md: str, memory_summary_md: str) -> None:
        with self.durable_view_lock():
            self._write_durable_memory_unlocked(memory_md, memory_summary_md)

    def write_raw_memories(self, content: str) -> None:
        _atomic_write_text(self.raw_memories_path, content)

    def capture_snapshot_source(self) -> RepositoryMemorySnapshotSource:
        with self.durable_view_lock():
            memory_summary = self.read_memory_summary()
            if not memory_summary:
                return RepositoryMemorySnapshotSource(
                    memory_summary="",
                    memory_md="",
                    rollout_summary_files={},
                )
            return RepositoryMemorySnapshotSource(
                memory_summary=memory_summary,
                memory_md=self.read_memory(),
                rollout_summary_files=self._read_rollout_summaries(),
            )

    def _read_rollout_summaries(self) -> dict[str, str]:
        if not self.rollout_summaries_dir.is_dir():
            return {}
        return {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(self.rollout_summaries_dir.glob("*.md"))
            if path.is_file()
        }

    def rollout_summary_filename(
        self,
        *,
        run_id: str,
        rollout_slug: str,
    ) -> str:
        return (
            f"{_safe_filename_component(run_id)}--"
            f"{_safe_filename_component(rollout_slug)}.md"
        )

    def write_rollout_summary(
        self,
        *,
        run_id: str,
        rollout_slug: str,
        content: str,
    ) -> Path:
        with self.durable_view_lock():
            return self._write_rollout_summary_unlocked(
                run_id=run_id,
                rollout_slug=rollout_slug,
                content=content,
            )

    def commit_consolidation(
        self,
        *,
        rollout_summaries: list[tuple[str, str, str]],
        memory_md: str,
        memory_summary_md: str,
    ) -> None:
        with self.durable_view_lock():
            for run_id, rollout_slug, content in rollout_summaries:
                self._write_rollout_summary_unlocked(
                    run_id=run_id,
                    rollout_slug=rollout_slug,
                    content=content,
                )
            self._write_durable_memory_unlocked(memory_md, memory_summary_md)

    def list_stage1_records(self) -> list[Stage1Record]:
        if not self.stage1_dir.is_dir():
            return []
        return [
            Stage1Record.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.stage1_dir.glob("*.json"))
        ]

    def try_pipeline_lock(self) -> _MemoryFileLock | None:
        return _acquire_file_lock(self.pipeline_lock_path, blocking=False)

    def durable_view_lock(self) -> _MemoryFileLock:
        lock = _acquire_file_lock(self.durable_lock_path, blocking=True)
        assert lock is not None
        return lock

    def _write_durable_memory_unlocked(
        self,
        memory_md: str,
        memory_summary_md: str,
    ) -> None:
        _atomic_write_text(self.memory_path, memory_md)
        _atomic_write_text(self.summary_path, memory_summary_md)

    def _write_rollout_summary_unlocked(
        self,
        *,
        run_id: str,
        rollout_slug: str,
        content: str,
    ) -> Path:
        self.rollout_summaries_dir.mkdir(parents=True, exist_ok=True)
        filename = self.rollout_summary_filename(
            run_id=run_id,
            rollout_slug=rollout_slug,
        )
        path = self.rollout_summaries_dir / filename
        if path.is_file():
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Rollout summary is immutable: {filename}")
            return path
        _atomic_write_text(path, content)
        return path

    def _write_new_stage1(self, record: Stage1Record) -> None:
        path = self.stage1_path(record.run_id)
        if path.exists():
            raise FileExistsError(f"Stage-1 terminal record already exists: {record.run_id}")
        _atomic_write_text(
            path,
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
        )


def _acquire_file_lock(
    path: Path,
    *,
    blocking: bool,
) -> _MemoryFileLock | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(stream.fileno(), mode, 1)
        else:
            import fcntl

            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            fcntl.flock(stream.fileno(), flags)
    except OSError as exc:
        stream.close()
        if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
            return None
        raise
    return _MemoryFileLock(stream)


def _safe_filename_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_")
    return normalized or "rollout"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)
