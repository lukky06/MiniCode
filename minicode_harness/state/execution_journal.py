"""Append-only persistence for side-effect execution crash windows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .checkpoint import digest_workspace_files


ExecutionJournalEventType = Literal["PREPARED", "COMPLETED", "RECONCILED"]
ExecutionEffectKind = Literal["filesystem", "command"]
ExecutionResolution = Literal[
    "effect_applied",
    "effect_not_applied",
    "effect_unknown",
]


class ExecutionJournalEvent(BaseModel):
    """One durable lifecycle fact for a side-effecting Tool Call."""

    model_config = ConfigDict(extra="forbid")

    event: ExecutionJournalEventType
    entry_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    argument_fingerprint: str = Field(min_length=1)
    effect_kind: ExecutionEffectKind
    target_paths: list[str] = Field(default_factory=list)
    before_hashes: dict[str, str | None] = Field(default_factory=dict)
    expected_after_hashes: dict[str, str | None] = Field(default_factory=dict)
    after_hashes: dict[str, str | None] = Field(default_factory=dict)
    command_category: str | None = None
    result_status: str | None = None
    returncode: int | None = None
    resolution: ExecutionResolution | None = None
    reason: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionJournalReconciliation(BaseModel):
    """Deterministic reconciliation result for one uncheckpointed entry."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    step: int = Field(ge=0)
    tool_call_id: str
    tool_name: str
    effect_kind: ExecutionEffectKind
    resolution: ExecutionResolution
    reason: str
    target_paths: list[str] = Field(default_factory=list)
    current_hashes: dict[str, str | None] = Field(default_factory=dict)
    result_status: str | None = None
    returncode: int | None = None


class ExecutionJournal:
    """Thread-safe append-only JSONL store under one Run directory."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append_prepared(
        self,
        *,
        entry_id: str,
        run_id: str,
        step: int,
        tool_call_id: str,
        tool_name: str,
        argument_fingerprint: str,
        effect_kind: ExecutionEffectKind,
        target_paths: list[str],
        before_hashes: dict[str, str | None],
        expected_after_hashes: dict[str, str | None],
        command_category: str | None = None,
    ) -> ExecutionJournalEvent:
        event = ExecutionJournalEvent(
            event="PREPARED",
            entry_id=entry_id,
            run_id=run_id,
            step=step,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            argument_fingerprint=argument_fingerprint,
            effect_kind=effect_kind,
            target_paths=target_paths,
            before_hashes=before_hashes,
            expected_after_hashes=expected_after_hashes,
            command_category=command_category,
        )
        self.append(event)
        return event

    def append_completed(
        self,
        prepared: ExecutionJournalEvent,
        *,
        after_hashes: dict[str, str | None],
        result_status: str | None = None,
        returncode: int | None = None,
    ) -> ExecutionJournalEvent:
        event = prepared.model_copy(
            update={
                "event": "COMPLETED",
                "after_hashes": after_hashes,
                "result_status": result_status,
                "returncode": returncode,
                "timestamp": datetime.now(timezone.utc),
            }
        )
        self.append(event)
        return event

    def append_reconciled(
        self,
        event: ExecutionJournalEvent,
        reconciliation: ExecutionJournalReconciliation,
    ) -> ExecutionJournalEvent:
        reconciled = event.model_copy(
            update={
                "event": "RECONCILED",
                "resolution": reconciliation.resolution,
                "reason": reconciliation.reason,
                "after_hashes": reconciliation.current_hashes,
                "timestamp": datetime.now(timezone.utc),
            }
        )
        self.append(reconciled)
        return reconciled

    def reconcile_uncheckpointed(
        self,
        workspace: Path | str,
        *,
        checkpointed_tool_call_ids: set[str],
    ) -> list[ExecutionJournalReconciliation]:
        """Classify side effects missing from the latest Checkpoint."""

        grouped: dict[str, list[ExecutionJournalEvent]] = {}
        for event in self.load_events():
            grouped.setdefault(event.entry_id, []).append(event)

        reconciliations: list[ExecutionJournalReconciliation] = []
        for entry_events in grouped.values():
            latest = entry_events[-1]
            if latest.tool_call_id in checkpointed_tool_call_ids:
                continue
            if latest.event == "RECONCILED" and latest.resolution is not None:
                reconciliations.append(
                    ExecutionJournalReconciliation(
                        entry_id=latest.entry_id,
                        step=latest.step,
                        tool_call_id=latest.tool_call_id,
                        tool_name=latest.tool_name,
                        effect_kind=latest.effect_kind,
                        resolution=latest.resolution,
                        reason=latest.reason or "previously_reconciled",
                        target_paths=latest.target_paths,
                        current_hashes=latest.after_hashes,
                        result_status=latest.result_status,
                        returncode=latest.returncode,
                    )
                )
                continue
            reconciliations.append(_reconcile_entry(workspace, entry_events))
        return reconciliations

    def append(self, event: ExecutionJournalEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(record + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def load_events(self) -> list[ExecutionJournalEvent]:
        if not self.path.is_file():
            return []
        raw = self.path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        events: list[ExecutionJournalEvent] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(ExecutionJournalEvent.model_validate_json(line))
            except ValidationError:
                if index == len(lines) - 1 and not raw.endswith("\n"):
                    break
                raise
        return events


def _reconcile_entry(
    workspace: Path | str,
    events: list[ExecutionJournalEvent],
) -> ExecutionJournalReconciliation:
    latest = events[-1]
    if latest.effect_kind == "command":
        return ExecutionJournalReconciliation(
            entry_id=latest.entry_id,
            step=latest.step,
            tool_call_id=latest.tool_call_id,
            tool_name=latest.tool_name,
            effect_kind=latest.effect_kind,
            resolution="effect_unknown",
            reason="command_effect_not_reconstructible",
            target_paths=latest.target_paths,
            result_status=latest.result_status,
            returncode=latest.returncode,
        )

    current_hashes = digest_workspace_files(workspace, latest.target_paths)
    completed = next(
        (event for event in reversed(events) if event.event == "COMPLETED"),
        None,
    )
    if completed is not None:
        if (
            completed.after_hashes
            and set(completed.after_hashes) == set(latest.target_paths)
            and current_hashes == completed.after_hashes
        ):
            return ExecutionJournalReconciliation(
                entry_id=latest.entry_id,
                step=latest.step,
                tool_call_id=latest.tool_call_id,
                tool_name=latest.tool_name,
                effect_kind=latest.effect_kind,
                resolution="effect_applied",
                reason="completed_after_hash_matches_workspace",
                target_paths=latest.target_paths,
                current_hashes=current_hashes,
                result_status=completed.result_status,
            )
        return ExecutionJournalReconciliation(
            entry_id=latest.entry_id,
            step=latest.step,
            tool_call_id=latest.tool_call_id,
            tool_name=latest.tool_name,
            effect_kind=latest.effect_kind,
            resolution="effect_unknown",
            reason="completed_after_hash_mismatch",
            target_paths=latest.target_paths,
            current_hashes=current_hashes,
            result_status=completed.result_status,
        )

    prepared = next(
        (event for event in reversed(events) if event.event == "PREPARED"),
        latest,
    )
    if current_hashes == prepared.before_hashes:
        return ExecutionJournalReconciliation(
            entry_id=prepared.entry_id,
            step=prepared.step,
            tool_call_id=prepared.tool_call_id,
            tool_name=prepared.tool_name,
            effect_kind=prepared.effect_kind,
            resolution="effect_not_applied",
            reason="workspace_matches_before_hash",
            target_paths=prepared.target_paths,
            current_hashes=current_hashes,
        )
    if (
        prepared.expected_after_hashes
        and set(prepared.expected_after_hashes) == set(prepared.target_paths)
        and current_hashes == prepared.expected_after_hashes
    ):
        return ExecutionJournalReconciliation(
            entry_id=prepared.entry_id,
            step=prepared.step,
            tool_call_id=prepared.tool_call_id,
            tool_name=prepared.tool_name,
            effect_kind=prepared.effect_kind,
            resolution="effect_applied",
            reason="workspace_matches_expected_after_hash",
            target_paths=prepared.target_paths,
            current_hashes=current_hashes,
        )
    return ExecutionJournalReconciliation(
        entry_id=prepared.entry_id,
        step=prepared.step,
        tool_call_id=prepared.tool_call_id,
        tool_name=prepared.tool_name,
        effect_kind=prepared.effect_kind,
        resolution="effect_unknown",
        reason="workspace_matches_neither_before_nor_expected_after",
        target_paths=prepared.target_paths,
        current_hashes=current_hashes,
    )
