"""Checkpoint persistence for resumable runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from minicode_harness.context import (
    ContextObservation,
    RunState,
    SessionCompactionState,
)
from minicode_harness.workspace import WorkspaceGuard

from .tasks import TaskListState


LATEST_CHECKPOINT_FILE = "latest.json"
CANONICAL_HISTORY_FILE = "history.jsonl"
EMPTY_HISTORY_SHA256 = hashlib.sha256(b"[]").hexdigest()


class WorkspaceConflict(BaseModel):
    """One file whose digest changed since the checkpoint."""

    path: str
    expected_digest: str | None
    actual_digest: str | None


class RunCheckpoint(BaseModel):
    """Current checkpoint schema for one resumable run."""

    model_config = {"extra": "forbid"}

    run_id: str
    step: int
    task: str
    workspace: str
    run_state: RunState = Field(default_factory=RunState)
    task_state: TaskListState = Field(default_factory=TaskListState)
    recent_observations: list[ContextObservation] = Field(default_factory=list)
    history_length: int = Field(default=0, ge=0)
    history_sha256: str = Field(default=EMPTY_HISTORY_SHA256, min_length=64, max_length=64)
    compaction_state: SessionCompactionState = Field(
        default_factory=SessionCompactionState
    )
    user_turn_id: str | None = None
    model_call_count: int = 0
    modified_files: list[str] = Field(default_factory=list)
    workspace_digest: dict[str, str | None] = Field(default_factory=dict)
    memory_snapshot_hash: str | None = None
    memory_snapshot_path: str | None = None
    tool_calls: int = 0
    status: str = "running"
    reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CheckpointStore:
    """File-backed checkpoint store under ``runs/<run_id>/checkpoints``."""

    def __init__(self, checkpoints_dir: Path | str) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)

    @property
    def latest_path(self) -> Path:
        return self.checkpoints_dir / LATEST_CHECKPOINT_FILE

    @property
    def history_path(self) -> Path:
        return self.checkpoints_dir.parent / CANONICAL_HISTORY_FILE

    def save(
        self,
        checkpoint: RunCheckpoint,
        *,
        message_history: list[dict[str, Any]] | None = None,
    ) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if message_history is not None:
            self._save_history(checkpoint, message_history)
            checkpoint = checkpoint.model_copy(
                update={
                    "history_length": len(message_history),
                    "history_sha256": _history_sha256(message_history),
                }
            )
        payload = json.dumps(checkpoint.model_dump(mode="json"), indent=2) + "\n"
        temporary = self.latest_path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.latest_path)
        return self.latest_path

    def load_latest(self) -> RunCheckpoint | None:
        if not self.latest_path.is_file():
            return None
        return RunCheckpoint.model_validate_json(self.latest_path.read_text(encoding="utf-8"))

    def load_history(self, checkpoint: RunCheckpoint) -> list[dict[str, Any]]:
        if checkpoint.history_length == 0:
            if checkpoint.history_sha256 != EMPTY_HISTORY_SHA256:
                raise ValueError("Checkpoint history hash mismatch.")
            return []
        lines = self._history_lines()
        if len(lines) < checkpoint.history_length:
            raise ValueError("Checkpoint canonical history is incomplete.")
        prefix = _decode_history_lines(lines[: checkpoint.history_length])
        if _history_sha256(prefix) != checkpoint.history_sha256:
            raise ValueError("Checkpoint history hash mismatch.")
        return prefix

    def _save_history(
        self,
        checkpoint: RunCheckpoint,
        message_history: list[dict[str, Any]],
    ) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        previous = self.load_latest()
        accepted_length = previous.history_length if previous is not None else 0
        accepted_hash = (
            previous.history_sha256 if previous is not None else EMPTY_HISTORY_SHA256
        )
        lines = self._history_lines(allow_missing=True)
        if len(lines) < accepted_length:
            raise ValueError("Checkpoint canonical history is incomplete.")
        accepted = _decode_history_lines(lines[:accepted_length])
        if _history_sha256(accepted) != accepted_hash:
            raise ValueError("Checkpoint history hash mismatch.")
        history_diverged = (
            accepted_length > len(message_history)
            or _history_sha256(message_history[:accepted_length]) != accepted_hash
        )
        if history_diverged:
            if previous is not None and (
                checkpoint.run_id != previous.run_id or checkpoint.step <= previous.step
            ):
                self._rewrite_history(message_history)
                return
            if accepted_length > len(message_history):
                raise ValueError("New checkpoint history is shorter than the accepted prefix.")
            raise ValueError("New checkpoint history diverges from the accepted prefix.")

        if len(lines) != accepted_length:
            self._rewrite_history(accepted)
        elif not self.history_path.exists():
            self.history_path.touch()

        tail = message_history[accepted_length:]
        if tail:
            with self.history_path.open("a", encoding="utf-8", newline="\n") as stream:
                for message in tail:
                    stream.write(
                        json.dumps(
                            message,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        )
                    )
                    stream.write("\n")

    def _history_lines(self, *, allow_missing: bool = False) -> list[str]:
        if not self.history_path.is_file():
            if allow_missing:
                return []
            raise ValueError("Checkpoint canonical history is missing.")
        try:
            return self.history_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError("Checkpoint canonical history is invalid.") from exc

    def _rewrite_history(self, messages: list[dict[str, Any]]) -> None:
        payload = "".join(
            json.dumps(
                message,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
            for message in messages
        )
        temporary = self.history_path.with_suffix(".jsonl.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.history_path)


def _decode_history_lines(lines: list[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line in lines:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Checkpoint canonical history is invalid.") from exc
        if not isinstance(message, dict):
            raise ValueError("Checkpoint canonical history is invalid.")
        messages.append(message)
    return messages


def _history_sha256(message_history: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        message_history,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def digest_workspace_files(
    workspace: Path | str,
    relative_paths: list[str],
) -> dict[str, str | None]:
    guard = WorkspaceGuard(workspace)
    digests: dict[str, str | None] = {}
    for relative_path in sorted(set(relative_paths)):
        resolved = guard.resolve(relative_path)
        if not resolved.exists() or not resolved.is_file():
            digests[relative_path] = None
            continue
        digests[relative_path] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return digests


def detect_workspace_conflicts(
    workspace: Path | str,
    checkpoint: RunCheckpoint,
) -> list[WorkspaceConflict]:
    current_digest = digest_workspace_files(workspace, list(checkpoint.workspace_digest.keys()))
    conflicts: list[WorkspaceConflict] = []
    for path, expected_digest in checkpoint.workspace_digest.items():
        actual_digest = current_digest.get(path)
        if actual_digest != expected_digest:
            conflicts.append(
                WorkspaceConflict(
                    path=path,
                    expected_digest=expected_digest,
                    actual_digest=actual_digest,
                )
            )
    return conflicts
