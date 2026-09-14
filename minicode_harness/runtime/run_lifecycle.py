"""Deterministic persistence and terminal lifecycle actions for one Agent run."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from minicode_harness.context import ContextObservation, RunState, SessionCompactionState
from minicode_harness.context.limits import MAX_CHECKPOINT_OBSERVATIONS
from minicode_harness.state import (
    ReplSessionMemory,
    CheckpointStore,
    RunCheckpoint,
    TaskListState,
    digest_workspace_files,
)
from minicode_harness.trace import TraceWriter


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable view of the mutable Loop state needed for persistence."""

    step: int
    messages: list[dict[str, Any]]
    compaction_state: SessionCompactionState
    run_state: RunState
    task_state: TaskListState
    observations: list[ContextObservation]
    modified_files: list[str]
    workspace_digest_paths: list[str]
    user_turn_id: str
    model_call_count: int
    tool_calls: int
    memory_snapshot_hash: str | None
    memory_snapshot_path: str | None


class RunLifecycle:
    """Execute deterministic persistence, terminal Trace, Hook, and close actions."""

    def __init__(
        self,
        *,
        run_id: str,
        task: str,
        workspace: str,
        trace_writer: TraceWriter,
        checkpoint_store: CheckpointStore,
        session_memory: ReplSessionMemory | None,
        emit_hook: Callable[[str, int | None, dict[str, Any]], None],
        resource_closers: list[tuple[str, Callable[[], None]]],
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.workspace = workspace
        self.trace_writer = trace_writer
        self.checkpoint_store = checkpoint_store
        self.session_memory = session_memory
        self.emit_hook = emit_hook
        self.resource_closers = resource_closers

    def persist_session(self, snapshot: RunSnapshot) -> None:
        if self.session_memory is None:
            return
        self.session_memory.replace_session_state(
            messages=snapshot.messages,
            compaction_state=snapshot.compaction_state,
        )

    def save_checkpoint(
        self,
        snapshot: RunSnapshot,
        *,
        status: str,
        reason: str | None,
    ) -> None:
        self.emit_hook(
            "before_checkpoint",
            snapshot.step,
            {"status": status, "reason": reason},
        )
        checkpoint = RunCheckpoint(
            run_id=self.run_id,
            step=snapshot.step,
            task=self.task,
            workspace=self.workspace,
            run_state=snapshot.run_state,
            task_state=snapshot.task_state,
            recent_observations=snapshot.observations[-MAX_CHECKPOINT_OBSERVATIONS:],
            compaction_state=snapshot.compaction_state,
            user_turn_id=snapshot.user_turn_id,
            model_call_count=snapshot.model_call_count,
            modified_files=snapshot.modified_files,
            workspace_digest=digest_workspace_files(
                self.workspace,
                snapshot.workspace_digest_paths,
            ),
            memory_snapshot_hash=snapshot.memory_snapshot_hash,
            memory_snapshot_path=snapshot.memory_snapshot_path,
            tool_calls=snapshot.tool_calls,
            status=status,
            reason=reason,
        )
        path = self.checkpoint_store.save(
            checkpoint,
            message_history=snapshot.messages,
        )
        self.trace_writer.write_event(
            "checkpoint_saved",
            step=snapshot.step,
            path=str(path),
            status=status,
            reason=reason,
            modified_files=snapshot.modified_files,
            workspace_digest_paths=snapshot.workspace_digest_paths,
            memory_snapshot_hash=snapshot.memory_snapshot_hash,
            memory_snapshot_path=snapshot.memory_snapshot_path,
        )

    def persist_and_checkpoint(
        self,
        snapshot: RunSnapshot,
        *,
        status: str,
        reason: str,
    ) -> None:
        self.persist_session(snapshot)
        self.save_checkpoint(snapshot, status=status, reason=reason)

    def finish_completed(
        self,
        snapshot: RunSnapshot,
        *,
        final_text: str,
        emit_final_text: Callable[[], None],
    ) -> None:
        self.run_stage(
            step=snapshot.step,
            started_event="session_history_persist_started",
            finished_event="session_history_persist_finished",
            action=lambda: self.persist_session(snapshot),
        )
        self.run_stage(
            step=snapshot.step,
            started_event="final_text_emit_started",
            finished_event="final_text_emit_finished",
            action=emit_final_text,
        )
        self.record_run_outcome(
            snapshot=snapshot,
            status="completed",
            final_text=final_text,
        )
        self.run_stage(
            step=snapshot.step,
            started_event="final_checkpoint_started",
            finished_event="final_checkpoint_finished",
            action=lambda: self.save_checkpoint(
                snapshot,
                status="completed",
                reason="final_text",
            ),
        )
        self.emit_hook(
            "stop",
            snapshot.step,
            {"reason": "final_text", "status": "completed"},
        )
        self.close_runtime_resources()

    def finish_cancelled(self, snapshot: RunSnapshot) -> None:
        reason = "cancelled"
        self.persist_session(snapshot)
        self.record_run_outcome(
            snapshot=snapshot,
            status="cancelled",
            final_text=None,
        )
        self.save_checkpoint(snapshot, status="cancelled", reason=reason)
        self.trace_writer.write_event(
            "run_cancelled",
            run_id=self.run_id,
            step=snapshot.step,
            modified_files=snapshot.modified_files,
        )
        self.emit_hook(
            "stop",
            snapshot.step,
            {"reason": reason, "status": "cancelled"},
        )
        self.close_runtime_resources()

    def finish_stopped(
        self,
        snapshot: RunSnapshot,
        *,
        reason: str,
        final_text: str | None,
        stop_summary: str | None,
        modified_before_stop: list[str],
    ) -> None:
        self.persist_session(snapshot)
        self.record_run_outcome(
            snapshot=snapshot,
            status="stopped",
            final_text=final_text,
            stop_summary=stop_summary,
        )
        if stop_summary is not None:
            self.trace_writer.write_event(
                "budget_stop_summary",
                step=snapshot.step,
                reason=reason,
                summary=stop_summary,
                modified_before_stop=modified_before_stop,
                modified_after_stop=snapshot.modified_files,
                verification_status=snapshot.run_state.verification.status,
            )
        self.save_checkpoint(snapshot, status="stopped", reason=reason)
        self.emit_hook(
            "stop",
            snapshot.step,
            {"reason": reason, "status": "stopped"},
        )
        self.close_runtime_resources()

    def stop_run(
        self,
        *,
        snapshot_factory: Callable[[], RunSnapshot],
        reason: str,
        final_text: str | None,
        rollback: Callable[[], None] | None = None,
    ) -> str | None:
        """Apply optional rollback and persist one deterministic stopped Run."""

        modified_before_stop = list(snapshot_factory().modified_files)
        if rollback is not None:
            rollback()
        snapshot = snapshot_factory()
        stop_summary = self.render_budget_stop_summary(
            snapshot=snapshot,
            reason=reason,
            modified_before_stop=modified_before_stop,
        )
        self.finish_stopped(
            snapshot,
            reason=reason,
            final_text=final_text,
            stop_summary=stop_summary,
            modified_before_stop=modified_before_stop,
        )
        return stop_summary

    @staticmethod
    def render_budget_stop_summary(
        *,
        snapshot: RunSnapshot,
        reason: str,
        modified_before_stop: list[str],
    ) -> str | None:
        if reason not in {"max_steps", "max_tool_calls"}:
            return None

        budget_label = (
            "model-call budget" if reason == "max_steps" else "tool-call budget"
        )
        lines = [f"Run stopped because the {budget_label} was exhausted."]

        if modified_before_stop:
            rendered_files = ", ".join(modified_before_stop[:8])
            if len(modified_before_stop) > 8:
                rendered_files += f", ... (+{len(modified_before_stop) - 8})"
            if snapshot.modified_files:
                lines.append(f"Changes retained: {rendered_files}.")
            else:
                lines.append(f"Changes rolled back: {rendered_files}.")
        else:
            lines.append("Changes: none.")

        verification = snapshot.run_state.verification
        if verification.status == "passed":
            detail = f" using `{verification.command}`" if verification.command else ""
            lines.append(f"Verification: passed{detail}.")
        elif verification.status == "failed":
            detail = f" using `{verification.command}`" if verification.command else ""
            returncode = (
                f" (exit {verification.returncode})"
                if verification.returncode is not None
                else ""
            )
            lines.append(f"Verification: failed{detail}{returncode}.")
        elif verification.status == "rolled_back":
            lines.append("Verification: prior changes were rolled back before completion.")
        else:
            lines.append("Verification: not run.")

        if snapshot.observations:
            observation = snapshot.observations[-1]
            status = str(observation.metadata.get("status") or "ok")
            detail = (observation.summary or observation.output_preview or "").strip()
            detail = " ".join(detail.split())
            if len(detail) > 240:
                detail = detail[:237] + "..."
            suffix = f" — {detail}" if detail else ""
            lines.append(f"Last tool result: {observation.tool_name} [{status}]{suffix}.")

        lines.append("No final model answer was produced.")
        return "\n".join(lines)

    def record_run_outcome(
        self,
        *,
        snapshot: RunSnapshot,
        status: str,
        final_text: str | None,
        stop_summary: str | None = None,
    ) -> None:
        self.trace_writer.write_event(
            "run_outcome",
            run_id=self.run_id,
            status=status,
            modified_files=snapshot.modified_files,
            final_text=final_text,
            stop_summary=stop_summary,
        )

    def run_stage(
        self,
        *,
        step: int,
        started_event: str,
        finished_event: str,
        action: Callable[[], None],
    ) -> None:
        started = time.monotonic()
        self.trace_writer.write_event(started_event, step=step)
        try:
            action()
        except Exception as exc:
            self.trace_writer.write_event(
                finished_event,
                step=step,
                status="error",
                duration_ms=int((time.monotonic() - started) * 1000),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self.trace_writer.write_event(
            finished_event,
            step=step,
            status="ok",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def close_runtime_resources(self) -> None:
        for name, close in self.resource_closers:
            try:
                close()
            except Exception as exc:  # pragma: no cover - defensive shutdown path
                self.trace_writer.write_event(
                    "runtime_resource_close_error",
                    resource=name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
