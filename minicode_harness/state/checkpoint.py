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
    message_history: list[dict[str, Any]] = Field(default_factory=list)
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

    def save(self, checkpoint: RunCheckpoint) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(checkpoint.model_dump(mode="json"), indent=2) + "\n"
        step_path = self.checkpoints_dir / f"step_{checkpoint.step:04d}.json"
        step_path.write_text(payload, encoding="utf-8")
        self.latest_path.write_text(payload, encoding="utf-8")
        return step_path

    def load_latest(self) -> RunCheckpoint | None:
        if not self.latest_path.is_file():
            return None
        return RunCheckpoint.model_validate_json(self.latest_path.read_text(encoding="utf-8"))


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
