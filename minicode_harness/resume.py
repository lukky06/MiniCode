"""Resume interrupted runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from minicode_harness.context import (
    ContextObservation,
    RunState,
    SessionCompactionState,
    build_observation,
    initialize_run_state,
    mark_verification_failed,
    mark_verification_not_run,
    mark_verification_passed,
    record_inspected_file,
    render_tool_result_message,
    group_messages,
)
from minicode_harness.context.limits import MAX_CHECKPOINT_OBSERVATIONS
from minicode_harness.loop import AgentLoop, AgentLoopConfig, AgentRunResult
from minicode_harness.mcp import MCPManager
from minicode_harness.memory import MemorySnapshotStore, RepositoryMemoryStore
from minicode_harness.state import ReplSessionMemory, ReplSessionStore
from minicode_harness.models import ModelClient, create_model_client
from minicode_harness.output import OutputSink
from minicode_harness.policy import (
    ApprovalPolicy,
    PermissionMode,
    render_argv,
    resolve_command_executable_identity,
    resolve_command_session_grant,
)
from minicode_harness.runtime.collaboration import CollaborationMode
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.request_orchestrator import RequestOrchestrator
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.storage import HarnessDataStore
from minicode_harness.tools import create_command_executor
from minicode_harness.state import (
    ApprovalClient,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
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
from minicode_harness.tools import StaleWriteError, ToolRegistry
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
        try:
            session = store.load_session(run_id)
            checkpoint = CheckpointStore(
                store.path_for(run_id) / "checkpoints"
            ).load_latest()
        except (FileNotFoundError, OSError, ValueError):
            continue
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
        for message in checkpoint.message_history
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
    message_history = list(checkpoint.message_history)
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
        message_history=message_history,
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
    saved_path = checkpoint_store.save(updated)

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

    original_checkpoint = checkpoint
    checkpoint, reconciliation_block = _reconcile_execution_journal_before_resume(
        run_id=run_id,
        run_path=run_path,
        workspace=session.workspace,
        task=session.task,
        checkpoint=checkpoint,
        checkpoint_store=checkpoint_store,
        trace_writer=trace_writer,
    )
    if reconciliation_block is not None:
        return reconciliation_block
    if checkpoint is not original_checkpoint:
        _sync_conversation_history(
            conversation_session,
            checkpoint,
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
    message_history = list(checkpoint.message_history if checkpoint else [])
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
    if pending is not None:
        restored = _resolve_pending_approval(
            pending=pending,
            session_workspace=session.workspace,
            trace_writer=trace_writer,
            approval_store=approval_store,
            checkpoint_store=checkpoint_store,
            approval_client=approval_client,
            conversation_session=conversation_session,
            run_id=run_id,
            task=session.task,
            observations=observations,
            message_history=message_history,
            compaction_state=compaction_state,
            modified_files=modified_files,
            run_state=run_state,
            task_state=task_state,
            tool_calls=tool_calls,
            mcp_config=session.mcp_config,
            cancellation_token=cancellation_token,
        )
        if restored.status != "continued":
            _sync_conversation_history(
                conversation_session,
                checkpoint_store.load_latest(),
                trace_writer=trace_writer,
                run_id=run_id,
            )
            return restored
        checkpoint = checkpoint_store.load_latest()
        observations = list(checkpoint.recent_observations if checkpoint else observations)
        message_history = list(checkpoint.message_history if checkpoint else message_history)
        compaction_state = (
            checkpoint.compaction_state.model_copy(deep=True)
            if checkpoint is not None
            else compaction_state
        )
        modified_files = list(checkpoint.modified_files if checkpoint else modified_files)
        run_state = initialize_run_state(checkpoint.run_state if checkpoint else run_state)
        task_state = checkpoint.task_state if checkpoint else task_state
        tool_calls = checkpoint.tool_calls if checkpoint else tool_calls
        start_step = checkpoint.step if checkpoint else start_step

        if (
            checkpoint is not None
            and steering_queue is not None
            and (cancellation_token is None or not cancellation_token.is_cancelled)
            and start_step < session.max_steps
            and _restored_tool_batch_is_complete(
                message_history,
                pending_tool_call_id=pending.tool_call_id,
            )
        ):
            steering_message = steering_queue.dequeue()
            if steering_message is not None:
                message_history.append(
                    {"role": "user", "content": steering_message}
                )
                checkpoint = _checkpoint_for_resume(
                    run_id=run_id,
                    step=start_step,
                    task=session.task,
                    workspace=session.workspace,
                    observations=observations,
                    message_history=message_history,
                    compaction_state=compaction_state,
                    modified_files=modified_files,
                    run_state=run_state,
                    task_state=task_state,
                    tool_calls=tool_calls,
                    memory_snapshot_hash=checkpoint.memory_snapshot_hash,
                    memory_snapshot_path=checkpoint.memory_snapshot_path,
                    status="running",
                    reason="steering_message",
                )
                saved_path = checkpoint_store.save(checkpoint)
                trace_writer.write_event(
                    "steering_message_consumed",
                    step=start_step,
                    content=steering_message,
                    remaining=len(steering_queue),
                    path=str(saved_path),
                    source="resume_pending_approval",
                )
                _sync_conversation_history(
                    conversation_session,
                    checkpoint,
                    trace_writer=trace_writer,
                    run_id=run_id,
                )

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
    )
    loop = AgentLoop(
        task=session.task,
        workspace=session.workspace,
        model_client=resolved_model_client,
        trace_writer=trace_writer,
        config=AgentLoopConfig(
            max_steps=session.max_steps,
            start_step=start_step,
            prompt_cache_enabled=session.prompt_cache_enabled,
            repository_memory_enabled=session.repository_memory_enabled,
            enable_subagents=session.subagents_enabled,
            enable_worktree_workers=not session.no_write,
        ),
        skill_names=_parse_skill_names(session.skills),
        no_skills=session.no_skills,
        data_dir=(
            repository_memory.data_dir
            if repository_memory is not None
            else resolved_data_dir
        ),
        repository_memory=repository_memory,
        memory_snapshot_hash=(memory_snapshot.index_hash if memory_snapshot else None),
        memory_snapshot_path=(
            memory_snapshot_store.checkpoint_path if memory_snapshot is not None else None
        ),
        long_term_context=long_term_context,
        enable_write=not session.no_write,
        approval_policy=ApprovalPolicy(session.approval_policy),
        permission_mode=PermissionMode(session.permission_mode),
        collaboration_mode=CollaborationMode(session.collaboration_mode),
        approval_client=approval_client,
        approval_store=approval_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        initial_observations=observations,
        initial_message_history=message_history,
        initial_compaction_state=compaction_state,
        initial_modified_files=modified_files,
        initial_run_state=run_state,
        initial_task_state=task_state,
        initial_tool_calls=tool_calls,
        provider=session.provider,
        model=session.model or getattr(resolved_model_client, "model", None),
        session_memory=conversation_session,
        output_sink=output_sink,
        stream_model=stream_model,
        cancellation_token=cancellation_token,
        mcp_config=session.mcp_config,
        command_executor=command_executor,
        subagent_model_client_factory=(
            (lambda: create_model_client(
                provider=session.provider,
                model=session.model,
            ))
            if model_client is None
            else None
        ),
        steering_queue=steering_queue,
    )
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


def _resume_expected_write_hash(
    run_state: RunState,
    tool_name: str,
    arguments: dict[str, Any],
) -> str | None:
    if tool_name != "write" or not bool(arguments.get("overwrite", False)):
        return None
    path = str(arguments.get("path") or "").strip().replace("\\", "/").removeprefix("./")
    for inspected in reversed(run_state.inspected_files):
        inspected_path = inspected.path.replace("\\", "/").removeprefix("./")
        if inspected_path != path:
            continue
        if inspected.content_status not in {
            "full_content_available_in_context",
            "full_content_available_via_artifact",
        }:
            return None
        return inspected.content_sha256
    return None


def _resolve_pending_approval(
    *,
    pending: ApprovalRequest,
    session_workspace: str,
    trace_writer: TraceWriter,
    approval_store: ApprovalStore,
    checkpoint_store: CheckpointStore,
    approval_client: ApprovalClient,
    conversation_session: ReplSessionMemory | None,
    run_id: str,
    task: str,
    observations: list[Any],
    message_history: list[dict[str, Any]],
    compaction_state: SessionCompactionState,
    modified_files: list[str],
    run_state: RunState,
    task_state: TaskListState,
    tool_calls: int,
    mcp_config: str | None,
    cancellation_token: CancellationToken | None,
) -> ResumeResult:
    step = pending.step or 0
    prior_checkpoint = checkpoint_store.load_latest()
    trace_writer.write_event(
        "approval_restored",
        step=step,
        approval_id=pending.id,
        tool_call_id=pending.tool_call_id,
        tool=pending.tool_name,
        preview_summary=pending.preview.get("summary", pending.preview),
    )
    response = approval_client.decide(pending)
    if response.decision == ApprovalDecision.APPROVE_SESSION:
        argv = pending.arguments.get("argv")
        session_grant = (
            resolve_command_session_grant(session_workspace, argv)
            if pending.tool_name == "run_command" and isinstance(argv, list) and argv
            else None
        )
        executable = (
            resolve_command_executable_identity(session_workspace, str(argv[0]))
            if pending.tool_name == "run_command" and isinstance(argv, list) and argv
            else None
        )
        if session_grant is None or conversation_session is None:
            response = ApprovalResponse(
                decision=ApprovalDecision.REJECT,
                reason="Session command grant could not be restored safely.",
            )
        else:
            conversation_session.grant_command_approval(session_grant)
            trace_writer.write_event(
                "approval_session_granted",
                step=step,
                approval_id=pending.id,
                tool_call_id=pending.tool_call_id,
                tool=pending.tool_name,
                executable=executable,
                grant_id=session_grant,
                restored=True,
            )
    trace_writer.write_event(
        "approval_resolved",
        step=step,
        approval_id=pending.id,
        tool_call_id=pending.tool_call_id,
        tool=pending.tool_name,
        decision=response.decision.value,
        reason=response.reason,
    )
    approval_store.clear_pending()

    if response.decision not in {
        ApprovalDecision.APPROVE,
        ApprovalDecision.APPROVE_SESSION,
    }:
        reason = f"approval_{response.decision.value}"
        content = (
            f"Tool {pending.tool_name} was not executed because approval decision was "
            f"{response.decision.value}. Choose another safe action or explain the blocker."
        )
        observation = ContextObservation(
            tool_call_id=pending.tool_call_id,
            tool_name=pending.tool_name,
            content=content,
            output_preview=content,
            token_estimate=max(1, len(content) // 4),
            summary=content,
            is_important=True,
            metadata={
                "status": reason,
                "approval_fingerprint": _pending_tool_fingerprint(pending),
            },
        )
        observations.append(observation)
        _ensure_tool_call_owner(message_history, pending)
        tool_message = {
            "role": "tool",
            "tool_call_id": pending.tool_call_id,
            "content": render_tool_result_message(observation),
        }
        message_history.append(dict(tool_message))
        status = "stopped" if response.decision == ApprovalDecision.ABORT else "running"
        checkpoint_store.save(
            _checkpoint_for_resume(
                run_id=run_id,
                step=step,
                task=task,
                workspace=session_workspace,
                observations=observations,
                message_history=message_history,
                compaction_state=compaction_state,
                modified_files=modified_files,
                run_state=run_state,
                task_state=task_state,
                tool_calls=tool_calls,
                memory_snapshot_hash=(
                    prior_checkpoint.memory_snapshot_hash if prior_checkpoint else None
                ),
                memory_snapshot_path=(
                    prior_checkpoint.memory_snapshot_path if prior_checkpoint else None
                ),
                status=status,
                reason=reason,
            )
        )
        if response.decision == ApprovalDecision.ABORT:
            return ResumeResult(status="stopped", run_id=run_id, reason=reason)
        return ResumeResult(status="continued", run_id=run_id, reason=reason)

    mcp_manager = MCPManager.from_config_file(mcp_config) if mcp_config else None
    registry = ToolRegistry(
        session_workspace,
        enable_write=True,
        mcp_manager=mcp_manager,
        cancellation_token=cancellation_token,
    )
    try:
        admission = registry.admit(pending.tool_name, pending.arguments)
        expected_file_sha256 = _resume_expected_write_hash(
            run_state,
            admission.name,
            admission.arguments,
        )
        try:
            result = registry.execute_admitted(
                admission,
                approval_granted=True,
                expected_file_sha256=expected_file_sha256,
            )
        except StaleWriteError as exc:
            result = None
            content = (
                f"Tool {admission.name} was not executed: {exc} "
                "No workspace write occurred. Read the complete target file again before retrying."
            )
            observation = ContextObservation(
                tool_call_id=pending.tool_call_id,
                tool_name=admission.name,
                content=content,
                output_preview=content,
                token_estimate=max(1, len(content) // 4),
                summary=content,
                is_important=True,
                metadata={
                    "status": "stale_write",
                    "error_type": "stale_write",
                    "reason": "resume_write_version_unavailable_or_changed",
                    "side_effect": "none",
                },
            )
        else:
            observation, _ = build_observation(
                tool_call_id=pending.tool_call_id,
                tool_name=admission.name,
                result=result,
                artifact_dir=trace_writer.trace_path.parent / "artifacts",
            )
    finally:
        if mcp_manager is not None:
            mcp_manager.close()
    observations.append(observation)
    _ensure_tool_call_owner(message_history, pending)
    tool_message = {
        "role": "tool",
        "tool_call_id": pending.tool_call_id,
        "content": render_tool_result_message(observation),
    }
    message_history.append(dict(tool_message))
    modified = (
        _modified_files_from_result(pending.tool_name, result)
        if result is not None
        else []
    )
    for modified_file in modified:
        if modified_file not in modified_files:
            modified_files.append(modified_file)
    if modified:
        mark_verification_not_run(run_state)

    if pending.tool_name == "run_command":
        pending_argv = pending.arguments.get("argv")
        fallback_command = (
            render_argv(pending_argv)
            if isinstance(pending_argv, list)
            and all(isinstance(item, str) for item in pending_argv)
            else ""
        )
        command = str(observation.metadata.get("command") or fallback_command)
        returncode = observation.metadata.get("returncode")
        if returncode == 0:
            mark_verification_passed(run_state, command=command, returncode=0)
        elif returncode is not None:
            mark_verification_failed(
                run_state,
                command=command,
                returncode=int(returncode),
            )

    record_inspected_file(
        state=run_state,
        tool_name=pending.tool_name,
        tool_call_id=pending.tool_call_id,
        arguments=pending.arguments,
        observation=observation,
        step=step,
        workspace_generation=1 if modified_files else 0,
    )
    trace_writer.write_event(
        "run_state_updated",
        step=step,
        modified_files=len(modified_files),
        verification_status=run_state.verification.status,
        inspected_files=len(run_state.inspected_files),
        tool_calls=tool_calls,
    )
    trace_writer.write_event(
        "tool_result",
        step=step,
        tool_call_id=pending.tool_call_id,
        tool=pending.tool_name,
        status="ok",
        truncated=observation.is_truncated,
        preview=observation.output_preview,
        artifact_path=observation.artifact_path,
    )
    checkpoint = _checkpoint_for_resume(
        run_id=run_id,
        step=step,
        task=task,
        workspace=session_workspace,
        observations=observations,
        message_history=message_history,
        compaction_state=compaction_state,
        modified_files=modified_files,
        run_state=run_state,
        task_state=task_state,
        tool_calls=tool_calls,
        memory_snapshot_hash=(
            prior_checkpoint.memory_snapshot_hash if prior_checkpoint else None
        ),
        memory_snapshot_path=(
            prior_checkpoint.memory_snapshot_path if prior_checkpoint else None
        ),
        status="running",
        reason=f"restored_approval:{pending.tool_name}",
    )
    saved_path = checkpoint_store.save(checkpoint)
    trace_writer.write_event(
        "checkpoint_saved",
        step=step,
        path=str(saved_path),
        status=checkpoint.status,
        reason=checkpoint.reason,
        modified_files=modified_files,
    )
    return ResumeResult(status="continued", run_id=run_id, reason="approval_restored")


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
    checkpoint: RunCheckpoint | None,
    *,
    trace_writer: TraceWriter,
    run_id: str,
) -> None:
    if session is None or checkpoint is None:
        return
    try:
        session.replace_session_state(
            messages=checkpoint.message_history,
            compaction_state=checkpoint.compaction_state,
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


def _restored_tool_batch_is_complete(
    messages: list[dict[str, Any]],
    *,
    pending_tool_call_id: str,
) -> bool:
    """Return whether the restored pending call now closes its Tool group."""

    for group in group_messages(messages):
        owner = group[0]
        if owner.get("role") != "assistant" or not owner.get("tool_calls"):
            continue
        expected_ids = {
            str(call.get("id"))
            for call in owner.get("tool_calls") or []
            if call.get("id") is not None
        }
        if pending_tool_call_id not in expected_ids:
            continue
        result_ids = [
            str(message.get("tool_call_id"))
            for message in group[1:]
            if message.get("role") == "tool"
        ]
        return len(result_ids) == len(expected_ids) and set(result_ids) == expected_ids
    return False


def _pending_tool_fingerprint(pending: ApprovalRequest) -> str:
    payload = json.dumps(
        {"name": pending.tool_name, "arguments": pending.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    message_history: list[dict[str, Any]],
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
        message_history=message_history,
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


def _modified_files_from_result(tool_name: str, result: Any) -> list[str]:
    model_dump = getattr(result, "model_dump", None)
    payload = model_dump(mode="json") if callable(model_dump) else result
    if not isinstance(payload, dict):
        return []
    if tool_name == "apply_patch":
        return list(payload.get("files") or [])
    if tool_name in {"edit", "write"}:
        path = payload.get("path")
        return [path] if path else []
    return []


def _parse_skill_names(skills: str | None) -> list[str] | None:
    if skills is None:
        return None
    return [skill.strip() for skill in skills.split(",") if skill.strip()]


def _append_unique_limited(values: list[str], value: str, *, limit: int) -> None:
    compact = " ".join(value.split())
    if not compact:
        return
    if compact in values:
        values.remove(compact)
    values.append(compact)
    del values[:-limit]
