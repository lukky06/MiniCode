"""Resume interrupted runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minicode_harness.context import (
    ContextObservation,
    RunState,
    SessionCompactionState,
    initialize_run_state,
    mark_verification_not_run,
)
from minicode_harness.context.limits import MAX_CHECKPOINT_OBSERVATIONS
from minicode_harness.loop import AgentLoop, AgentRunResult
from minicode_harness.memory import MemorySnapshotStore, RepositoryMemoryStore
from minicode_harness.state import ReplSessionMemory, ReplSessionStore
from minicode_harness.models import ModelClient, NormalizedToolCall, create_model_client
from minicode_harness.output import OutputSink
from minicode_harness.policy import CommandRule, PermissionMode
from minicode_harness.runtime.collaboration import CollaborationMode
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.request_orchestrator import RequestOrchestrator
from minicode_harness.runtime.run_executor import build_agent_loop_from_session
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.storage import HarnessDataStore
from minicode_harness.tools import create_command_executor
from minicode_harness.state import (
    ApprovalClient,
    ApprovalRequest,
    ApprovalStore,
    CheckpointStore,
    ExecutionJournal,
    ExecutionJournalReconciliation,
    InteractiveApprovalClient,
    RunCheckpoint,
    RunSession,
    TaskListState,
    RunStore,
    detect_workspace_conflicts,
    digest_workspace_files,
)
from minicode_harness.trace import TraceWriter


@dataclass(frozen=True)
class ResumeResult:
    status: str
    run_id: str
    reason: str
    final_text: str | None = None
    stop_summary: str | None = None
    conflicts: list[str] | None = None
    agent_result: AgentRunResult | None = None


RECOVERABLE_CHECKPOINT_STATUSES = {"running", "stopped", "cancelled"}


def latest_recoverable_run_id(
    *,
    run_store: RunStore | None = None,
    workspace: Path | str | None = None,
    conversation_session_id: str | None = None,
) -> str:
    """Return the newest scoped Run with a recoverable Checkpoint."""

    store = run_store or RunStore()
    run_ids = store.list_run_ids(
        workspace=workspace,
        conversation_session_id=conversation_session_id,
    )
    for run_id in reversed(run_ids):
        session = store.load_session(run_id)
        checkpoint = CheckpointStore(
            store.path_for(run_id) / "checkpoints"
        ).load_latest()
        if session.status == "completed" or checkpoint is None:
            continue
        if checkpoint.status in RECOVERABLE_CHECKPOINT_STATUSES:
            return run_id
    raise FileNotFoundError("No recoverable Run found in the requested scope.")


def _reconcile_execution_journal_before_resume(
    *,
    run_id: str,
    run_path: Path,
    workspace: str,
    task: str,
    checkpoint: RunCheckpoint | None,
    message_history: list[dict[str, Any]],
    checkpoint_store: CheckpointStore,
    trace_writer: TraceWriter,
) -> tuple[RunCheckpoint | None, ResumeResult | None]:
    journal = ExecutionJournal(run_path / "execution-journal.jsonl")
    if not journal.path.is_file():
        return checkpoint, None
    try:
        events = journal.load_events()
    except (OSError, ValueError) as exc:
        trace_writer.write_event(
            "resume_blocked",
            run_id=run_id,
            reason="execution_journal_invalid",
            error=str(exc),
        )
        return checkpoint, ResumeResult(
            status="blocked",
            run_id=run_id,
            reason="execution_journal_invalid",
        )
    if not events:
        return checkpoint, None
    if checkpoint is None:
        trace_writer.write_event(
            "resume_blocked",
            run_id=run_id,
            reason="execution_journal_without_checkpoint",
        )
        return None, ResumeResult(
            status="blocked",
            run_id=run_id,
            reason="execution_journal_without_checkpoint",
        )

    checkpointed_tool_call_ids = {
        observation.tool_call_id
        for observation in checkpoint.recent_observations
        if observation.tool_call_id
    }
    checkpointed_tool_call_ids.update(
        str(message.get("tool_call_id"))
        for message in message_history
        if message.get("role") == "tool" and message.get("tool_call_id")
    )
    reconciliations = journal.reconcile_uncheckpointed(
        workspace,
        checkpointed_tool_call_ids=checkpointed_tool_call_ids,
    )
    if not reconciliations:
        return checkpoint, None

    unresolved = [
        item for item in reconciliations if item.resolution == "effect_unknown"
    ]
    if unresolved:
        conflict_paths = [
            path
            for item in unresolved
            for path in item.target_paths
        ]
        if not conflict_paths:
            conflict_paths = [
                f"{item.tool_name}:{item.tool_call_id}" for item in unresolved
            ]
        trace_writer.write_event(
            "resume_blocked",
            run_id=run_id,
            reason="execution_effect_unknown",
            entries=[item.model_dump(mode="json") for item in unresolved],
        )
        return checkpoint, ResumeResult(
            status="blocked",
            run_id=run_id,
            reason="execution_effect_unknown",
            conflicts=list(dict.fromkeys(conflict_paths)),
        )

    observations = list(checkpoint.recent_observations)
    message_history = list(message_history)
    modified_files = list(checkpoint.modified_files)
    run_state = initialize_run_state(checkpoint.run_state)
    step = checkpoint.step
    tool_calls = checkpoint.tool_calls
    for item in reconciliations:
        notification = _render_execution_reconciliation_notification(item)
        observations.append(
            ContextObservation(
                tool_call_id=item.tool_call_id,
                tool_name=item.tool_name,
                content=notification,
                output_preview=notification,
                token_estimate=max(1, len(notification) // 4),
                summary=notification,
                is_important=True,
                metadata={
                    "status": f"reconciled_{item.resolution}",
                    "side_effect": item.resolution,
                    "reason": item.reason,
                    "target_paths": item.target_paths,
                },
            )
        )
        message_history.append({"role": "assistant", "content": notification})
        tool_calls += 1
        step = max(step, item.step)
        if item.resolution == "effect_applied" and item.effect_kind == "filesystem":
            for path in item.target_paths:
                if path not in modified_files:
                    modified_files.append(path)
            mark_verification_not_run(run_state)

    updated = _checkpoint_for_resume(
        run_id=run_id,
        step=step,
        task=task,
        workspace=workspace,
        observations=observations,
        compaction_state=checkpoint.compaction_state,
        modified_files=modified_files,
        run_state=run_state,
        task_state=checkpoint.task_state,
        tool_calls=tool_calls,
        memory_snapshot_hash=checkpoint.memory_snapshot_hash,
        memory_snapshot_path=checkpoint.memory_snapshot_path,
        status="running",
        reason="execution_journal_reconciled",
    )
    saved_path = checkpoint_store.save(
        updated,
        message_history=message_history,
    )
    updated = checkpoint_store.load_latest() or updated

    latest_by_entry: dict[str, Any] = {}
    for event in events:
        latest_by_entry[event.entry_id] = event
    for item in reconciliations:
        latest = latest_by_entry.get(item.entry_id)
        if latest is not None and latest.event != "RECONCILED":
            journal.append_reconciled(latest, item)

    trace_writer.write_event(
        "execution_journal_reconciled",
        run_id=run_id,
        path=str(saved_path),
        entries=[item.model_dump(mode="json") for item in reconciliations],
    )
    return updated, None


def _render_execution_reconciliation_notification(
    item: ExecutionJournalReconciliation,
) -> str:
    target_text = ", ".join(item.target_paths) or "no workspace path"
    if item.resolution == "effect_applied":
        detail = (
            f"Recovered interrupted {item.tool_name} side effect. "
            f"The workspace already contains the effect for {target_text}. "
            "MiniCode did not replay the operation. Re-inspect exact content if needed."
        )
    else:
        detail = (
            f"Recovered interrupted {item.tool_name} attempt. "
            f"No side effect is present for {target_text}. "
            "MiniCode did not replay it; retry through the normal tool flow if still needed."
        )
    return "[MiniCode runtime notification]\n" + detail


def resume_run(
    run_id: str,
    *,
    run_store: RunStore | None = None,
    force_rebuild_context: bool = False,
    model_client: ModelClient | None = None,
    approval_client: ApprovalClient | None = None,
    data_dir: str | None = None,
    memory_store: HarnessDataStore | None = None,
    session_store: ReplSessionStore | None = None,
    output_sink: OutputSink | None = None,
    stream_model: bool = False,
    cancellation_token: CancellationToken | None = None,
    steering_queue: SteeringQueue | None = None,
) -> ResumeResult:
    """Resume a run from the current checkpoint schema."""

    store = run_store or RunStore()
    session = store.load_session(run_id)
    command_rules = [CommandRule.model_validate(item) for item in session.command_rules]
    run_path = store.path_for(run_id)
    trace_writer = TraceWriter(run_path / "trace.jsonl")
    checkpoint_store = CheckpointStore(run_path / "checkpoints")
    approval_store = ApprovalStore(run_path / "approvals")
    approval_client = approval_client or InteractiveApprovalClient()
    checkpoint = checkpoint_store.load_latest()
    conversation_session = _load_conversation_session(
        session=session,
        session_store=session_store or ReplSessionStore(),
        trace_writer=trace_writer,
    )

    trace_writer.write_event(
        "resume_started",
        run_id=run_id,
        conversation_session_id=session.conversation_session_id,
        force_rebuild_context=force_rebuild_context,
    )

    message_history: list[dict[str, Any]] = []
    if checkpoint is not None:
        try:
            message_history = checkpoint_store.load_history(checkpoint)
        except ValueError as exc:
            trace_writer.write_event(
                "resume_blocked",
                run_id=run_id,
                reason="canonical_history_invalid",
                error=str(exc),
            )
            return ResumeResult(
                status="blocked",
                run_id=run_id,
                reason="canonical_history_invalid",
            )

    original_checkpoint = checkpoint
    checkpoint, reconciliation_block = _reconcile_execution_journal_before_resume(
        run_id=run_id,
        run_path=run_path,
        workspace=session.workspace,
        task=session.task,
        checkpoint=checkpoint,
        message_history=message_history,
        checkpoint_store=checkpoint_store,
        trace_writer=trace_writer,
    )
    if reconciliation_block is not None:
        return reconciliation_block
    if checkpoint is not original_checkpoint and checkpoint is not None:
        message_history = checkpoint_store.load_history(checkpoint)
        _sync_conversation_history(
            conversation_session,
            message_history,
            checkpoint.compaction_state,
            trace_writer=trace_writer,
            run_id=run_id,
        )

    if checkpoint is not None and not force_rebuild_context:
        conflicts = detect_workspace_conflicts(session.workspace, checkpoint)
        if conflicts:
            trace_writer.write_event(
                "resume_blocked",
                run_id=run_id,
                reason="workspace_conflict",
                conflicts=[conflict.model_dump() for conflict in conflicts],
            )
            return ResumeResult(
                status="blocked",
                run_id=run_id,
                reason="workspace_conflict",
                conflicts=[conflict.path for conflict in conflicts],
            )

    observations = list(checkpoint.recent_observations if checkpoint else [])
    compaction_state = (
        checkpoint.compaction_state.model_copy(deep=True)
        if checkpoint is not None
        else SessionCompactionState()
    )
    modified_files = list(checkpoint.modified_files if checkpoint else [])
    run_state = initialize_run_state(checkpoint.run_state if checkpoint else None)
    task_state = checkpoint.task_state if checkpoint else TaskListState()
    tool_calls = checkpoint.tool_calls if checkpoint else 0
    start_step = checkpoint.step if checkpoint else 0

    pending = approval_store.load_pending()

    resolved_model_client = model_client or create_model_client(
        provider=session.provider,
        model=session.model,
    )
    resolved_data_dir = (
        data_dir
        or (memory_store.data_dir if memory_store is not None else None)
    )
    repository_memory = None
    request_orchestrator = None
    memory_snapshot_store = MemorySnapshotStore(run_path)
    memory_snapshot = None
    if session.repository_memory_enabled:
        repository_memory = RepositoryMemoryStore(
            session.workspace,
            data_dir=resolved_data_dir,
        )
        request_orchestrator = RequestOrchestrator(
            repository_memory=repository_memory,
            trace_writer=trace_writer,
            review_model_client=resolved_model_client,
        )
        if force_rebuild_context:
            latest_memory = repository_memory.capture_snapshot_source()
            memory_snapshot = memory_snapshot_store.save(
                repository_id=repository_memory.repository_id,
                rendered_index=latest_memory.rendered_index,
                topic_payloads=latest_memory.topic_payloads,
            )
            snapshot_source = "latest_repository_memory"
        else:
            try:
                memory_snapshot = memory_snapshot_store.load(
                    expected_hash=(checkpoint.memory_snapshot_hash if checkpoint else None),
                )
            except ValueError as exc:
                trace_writer.write_event(
                    "resume_blocked",
                    run_id=run_id,
                    reason="memory_snapshot_invalid",
                    error=str(exc),
                )
                return ResumeResult(
                    status="blocked",
                    run_id=run_id,
                    reason="memory_snapshot_invalid",
                )
            if memory_snapshot is None:
                trace_writer.write_event(
                    "resume_blocked",
                    run_id=run_id,
                    reason="memory_snapshot_missing",
                )
                return ResumeResult(
                    status="blocked",
                    run_id=run_id,
                    reason="memory_snapshot_missing",
                )
            snapshot_source = "original_run_snapshot"
        long_term_context = memory_snapshot.rendered_index
        trace_writer.write_event(
            "memory_snapshot_restored",
            repository_id=memory_snapshot.repository_id,
            index_hash=memory_snapshot.index_hash,
            path=memory_snapshot_store.checkpoint_path,
            source=snapshot_source,
            force_rebuild_context=force_rebuild_context,
        )
    else:
        long_term_context = ""
    command_executor = create_command_executor(
        session.sandbox_mode,
        image=session.sandbox_image,
        command_rules=command_rules,
        workspace_writable=PermissionMode(session.permission_mode) != PermissionMode.READ_ONLY,
    )
    loop = build_agent_loop_from_session(
        session,
        model_client=resolved_model_client,
        trace_writer=trace_writer,
        repository_memory=repository_memory,
        memory_snapshot_hash=(memory_snapshot.index_hash if memory_snapshot else None),
        memory_snapshot_path=(
            memory_snapshot_store.checkpoint_path if memory_snapshot is not None else None
        ),
        long_term_context=long_term_context,
        command_executor=command_executor,
        approval_client=approval_client,
        session_memory=conversation_session,
        output_sink=output_sink,
        stream_model=stream_model,
        cancellation_token=cancellation_token,
        steering_queue=steering_queue,
        data_dir=(
            repository_memory.data_dir
            if repository_memory is not None
            else resolved_data_dir
        ),
        checkpoint_store=checkpoint_store,
        approval_store=approval_store,
        start_step=start_step,
        initial_observations=observations,
        initial_message_history=message_history,
        initial_compaction_state=compaction_state,
        initial_modified_files=modified_files,
        initial_run_state=run_state,
        initial_task_state=task_state,
        initial_tool_calls=tool_calls,
        subagent_model_client_factory=(
            (lambda: create_model_client(
                provider=session.provider,
                model=session.model,
            ))
            if model_client is None
            else None
        ),
    )
    if pending is not None:
        restored_result = _restore_pending_approval_with_runtime(loop, pending)
        if restored_result is not None:
            store.update_session_state(
                run_id,
                status=restored_result.status,
                current_step=restored_result.steps,
            )
            _sync_conversation_history(
                conversation_session,
                loop.user_turn.messages,
                loop.compaction_state,
                trace_writer=trace_writer,
                run_id=run_id,
            )
            trace_writer.write_event(
                "resume_finished",
                run_id=run_id,
                conversation_session_id=session.conversation_session_id,
                status=restored_result.status,
                stop_reason=restored_result.stop_reason,
            )
            return ResumeResult(
                status=restored_result.status,
                run_id=run_id,
                reason=restored_result.stop_reason,
                final_text=restored_result.final_text,
                stop_summary=restored_result.stop_summary,
                agent_result=restored_result,
            )
        loop._consume_one_steering_message(pending.step or start_step)
    agent_result = loop.run()
    store.update_session_state(
        run_id,
        status=agent_result.status,
        current_step=agent_result.steps,
    )
    if (
        agent_result.status == "completed"
        and agent_result.final_text
        and request_orchestrator is not None
    ):
        request_orchestrator.finalize_completed_run(
            run_id=run_id,
            user_input=session.task,
            assistant_text=agent_result.final_text,
            observations=list(loop.observations),
            modified_files=list(loop.modified_files),
            verification=loop.run_state.verification,
        )
    trace_writer.write_event(
        "resume_finished",
        run_id=run_id,
        conversation_session_id=session.conversation_session_id,
        status=agent_result.status,
        stop_reason=agent_result.stop_reason,
    )
    return ResumeResult(
        status=agent_result.status,
        run_id=run_id,
        reason=agent_result.stop_reason,
        final_text=agent_result.final_text,
        stop_summary=agent_result.stop_summary,
        agent_result=agent_result,
    )


def _restore_pending_approval_with_runtime(
    loop: AgentLoop,
    pending: ApprovalRequest,
) -> AgentRunResult | None:
    """Resume one pending Tool Call through the normal ToolRuntime path."""

    step = pending.step or loop.config.start_step
    loop.trace_writer.write_event(
        "approval_restored",
        step=step,
        approval_id=pending.id,
        tool_call_id=pending.tool_call_id,
        tool=pending.tool_name,
        preview_summary=pending.preview.get("summary", pending.preview),
    )
    _ensure_tool_call_owner(loop.user_turn.messages, pending)
    tool_call = NormalizedToolCall(
        id=pending.tool_call_id,
        name=pending.tool_name,
        arguments=dict(pending.arguments),
    )
    hidden_tool_names = set(loop.config.hidden_tool_names)
    collaboration_tool_names = (
        set(loop.tools.readonly_tool_names())
        if loop.collaboration_mode == CollaborationMode.PLAN
        else None
    )
    tool_names = tuple(
        str(schema["function"]["name"])
        for schema in loop.tools.schemas()
        if str(schema["function"]["name"]) not in hidden_tool_names
        and (
            collaboration_tool_names is None
            or str(schema["function"]["name"]) in collaboration_tool_names
        )
    )
    if loop.tools.counts_against_tool_budget(tool_call.name):
        loop.tool_call_count += 1
    outcome = loop.tool_runtime.execute(
        step=step,
        tool_call=tool_call,
        available_tool_names=tool_names,
        workspace_generation=loop.workspace_generation,
    )
    guidance = loop.tool_batch.commit_outcome(
        loop,
        step,
        tool_call,
        outcome,
    )
    if guidance is not None:
        loop.user_turn.append_message({"role": "user", "content": guidance.message})
        loop.trace_writer.write_event(
            "progress_stagnation_nudge",
            step=step,
            level=guidance.level,
            guidance=guidance.message,
            non_write_calls_since_progress=(
                loop.progress_policy.non_write_calls_since_progress
            ),
            remaining_steps=max(0, loop.config.max_steps - step),
        )
        loop._persist_and_checkpoint(step, reason="progress_guidance")
    if outcome.stop_reason:
        return loop._finish_stopped_run(
            step=step,
            reason=outcome.stop_reason,
            final_text=None,
            rollback=False,
        )
    return None


def _load_conversation_session(
    *,
    session: RunSession,
    session_store: ReplSessionStore,
    trace_writer: TraceWriter,
) -> ReplSessionMemory | None:
    session_id = session.conversation_session_id
    if not session_id:
        return None
    try:
        return session_store.load(session.workspace, session_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        trace_writer.write_event(
            "session_sync_skipped",
            run_id=session.run_id,
            conversation_session_id=session_id,
            reason="session_load_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None


def _sync_conversation_history(
    session: ReplSessionMemory | None,
    message_history: list[dict[str, Any]],
    compaction_state: SessionCompactionState,
    *,
    trace_writer: TraceWriter,
    run_id: str,
) -> None:
    if session is None:
        return
    try:
        session.replace_session_state(
            messages=message_history,
            compaction_state=compaction_state,
        )
    except (OSError, ValueError) as exc:
        trace_writer.write_event(
            "session_sync_skipped",
            run_id=run_id,
            conversation_session_id=session.session_id,
            reason="session_save_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )


def _ensure_tool_call_owner(
    message_history: list[dict[str, Any]],
    pending: ApprovalRequest,
) -> None:
    """Ensure a restored tool result has its canonical assistant owner."""

    for message in reversed(message_history):
        if message.get("role") != "assistant":
            continue
        tool_ids = {
            str(call.get("id"))
            for call in message.get("tool_calls") or []
            if call.get("id") is not None
        }
        if pending.tool_call_id in tool_ids:
            return
        break
    message_history.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": pending.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": pending.tool_name,
                        "arguments": json.dumps(
                            pending.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    },
                }
            ],
        }
    )


def _checkpoint_for_resume(
    *,
    run_id: str,
    step: int,
    task: str,
    workspace: str,
    observations: list[Any],
    compaction_state: SessionCompactionState,
    modified_files: list[str],
    run_state: RunState,
    task_state: TaskListState,
    tool_calls: int,
    memory_snapshot_hash: str | None = None,
    memory_snapshot_path: str | None = None,
    status: str,
    reason: str,
) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=run_id,
        step=step,
        task=task,
        workspace=workspace,
        run_state=run_state,
        task_state=task_state,
        recent_observations=observations[-MAX_CHECKPOINT_OBSERVATIONS:],
        compaction_state=compaction_state,
        modified_files=modified_files,
        workspace_digest=digest_workspace_files(
            workspace,
            _checkpoint_digest_paths(modified_files, run_state),
        ),
        memory_snapshot_hash=memory_snapshot_hash,
        memory_snapshot_path=memory_snapshot_path,
        tool_calls=tool_calls,
        status=status,
        reason=reason,
    )


def _checkpoint_digest_paths(modified_files: list[str], run_state: RunState) -> list[str]:
    paths: list[str] = []
    for path in modified_files:
        _append_unique_limited(paths, path, limit=10_000)
    for inspected in run_state.inspected_files:
        _append_unique_limited(paths, inspected.path, limit=10_000)
    return paths


def _append_unique_limited(values: list[str], value: str, *, limit: int) -> None:
    compact = " ".join(value.split())
    if not compact:
        return
    if compact in values:
        values.remove(compact)
    values.append(compact)
    del values[:-limit]
