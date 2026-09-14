"""Deterministic tool admission, execution, reuse, approval, and rollback runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any

from pydantic import ValidationError

from minicode_harness.context import (
    ContextObservation,
    RunState,
    build_observation,
)
from minicode_harness.context.token import estimate_tokens
from minicode_harness.hooks import HookDecision, HookEvent, HookManager
from minicode_harness.models import NormalizedToolCall
from minicode_harness.output import OutputSink
from minicode_harness.policy import (
    ApprovalPolicy,
    PermissionDecision,
    PermissionMode,
    decide_permission,
    render_argv,
    resolve_command_executable_identity,
    resolve_command_session_grant,
)
from minicode_harness.state import (
    ApprovalClient,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStore,
    ExecutionJournal,
    ExecutionJournalEvent,
    ReplSessionMemory,
    digest_workspace_files,
)
from minicode_harness.tools import FileReadResult, StaleWriteError, ToolRegistry
from minicode_harness.tools.registry import ToolAdmission
from minicode_harness.tools.semantics import (
    is_edit_tool,
    is_memory_read,
    is_workspace_read,
)
from minicode_harness.trace import TraceWriter
from minicode_harness.workspace import WorkspaceAccessError, WorkspaceGuard, extract_source_paths

from .tool_reuse import (
    ReadReuseMatch,
    SearchReuseMatch,
    ToolReuseTracker,
    coerce_optional_int,
    normalize_workspace_path,
)


MAX_REPEATED_MALFORMED_TOOL_CALLS = 3
MAX_TOOL_ERROR_MESSAGE_CHARS = 4000
MAX_TUI_DIFF_PREVIEW_CHARS = 2000
MAX_TUI_DIFF_PREVIEW_LINES = 12


@dataclass
class ModelOutputBudget:
    current: int
    maximum: int


@dataclass(frozen=True)
class ToolExecutionOutcome:
    observation: ContextObservation
    stop_reason: str | None = None
    modified_files: list[str] | None = None


@dataclass(frozen=True)
class RollbackOutcome:
    performed: bool
    restored: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class WorkspaceFileSnapshot:
    path: str
    existed: bool
    content: bytes | None = None


class ToolRuntime:
    """Own the complete deterministic pipeline for one admitted Tool Call."""

    def __init__(
        self,
        *,
        workspace: str,
        task: str,
        run_id: str,
        tools: ToolRegistry,
        trace_writer: TraceWriter,
        hook_manager: HookManager,
        hook_owner: Any,
        approval_client: ApprovalClient,
        approval_store: ApprovalStore,
        approval_policy: ApprovalPolicy,
        permission_mode: PermissionMode,
        execution_journal: ExecutionJournal | None,
        session_memory: ReplSessionMemory | None,
        artifact_dir: Path,
        output_sink: OutputSink,
        initial_observations: list[ContextObservation],
        run_state: RunState,
        initial_workspace_generation: int,
        max_memory_topic_reads: int,
        rollback_on_unfinished_stop: bool,
        output_budget: ModelOutputBudget,
    ) -> None:
        self.workspace = workspace
        self.task = task
        self.run_id = run_id
        self.tools = tools
        self.trace_writer = trace_writer
        self.hook_manager = hook_manager
        self.hook_owner = hook_owner
        self.approval_client = approval_client
        self.approval_store = approval_store
        self.approval_policy = approval_policy
        self.permission_mode = permission_mode
        self.execution_journal = execution_journal
        self.session_memory = session_memory
        self.artifact_dir = artifact_dir
        self.output_sink = output_sink
        self.max_memory_topic_reads = max(0, max_memory_topic_reads)
        self.rollback_on_unfinished_stop = rollback_on_unfinished_stop
        self.output_budget = output_budget
        self.run_state = run_state

        self._state_lock = Lock()
        self._denied_tool_calls: set[str] = set()
        self._malformed_tool_call_counts: dict[str, int] = {}
        self._reuse_tracker = ToolReuseTracker()
        self._failed_command_attempts: dict[tuple[str, ...], tuple[int, int]] = {}
        self._write_snapshots: dict[str, WorkspaceFileSnapshot] = {}
        self._memory_topic_read_count = sum(
            1
            for observation in initial_observations
            if observation.tool_name == "read"
            and observation.metadata.get("read_source") == "memory"
        )
        self._denied_tool_calls.update(
            str(observation.metadata.get("approval_fingerprint"))
            for observation in initial_observations
            if observation.metadata.get("approval_fingerprint")
        )
        self._restore_failed_command_attempts(initial_observations)
        self._reuse_tracker.restore(
            observations=initial_observations,
            run_state=run_state,
            workspace_generation=initial_workspace_generation,
        )

    def plan_memory_read_batch(
        self,
        tool_calls: list[NormalizedToolCall],
    ) -> dict[str, bool]:
        """Reserve bounded memory-read slots in provider response order."""

        decisions: dict[str, bool] = {}
        with self._state_lock:
            remaining = max(
                0,
                self.max_memory_topic_reads - self._memory_topic_read_count,
            )
            for tool_call in tool_calls:
                if not is_memory_read(tool_call.name, tool_call.arguments):
                    continue
                decisions[tool_call.id] = remaining > 0
                if remaining > 0:
                    remaining -= 1
        return decisions

    def execute(
        self,
        *,
        step: int,
        tool_call: NormalizedToolCall,
        available_tool_names: tuple[str, ...],
        workspace_generation: int,
        memory_read_allowed: bool | None = None,
    ) -> ToolExecutionOutcome:
        """Validate and execute one Tool Call without mutating AgentLoop state."""

        self.trace_writer.write_event(
            "tool_called",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            args=tool_call.arguments,
        )
        self.emit_tool_call_started(step, tool_call)
        try:
            if tool_call.name not in available_tool_names:
                return self._unavailable_tool_outcome(step, tool_call, available_tool_names)

            if is_memory_read(tool_call.name, tool_call.arguments):
                if memory_read_allowed is False:
                    return self._memory_topic_read_limit_outcome(step, tool_call)
                if memory_read_allowed is None:
                    with self._state_lock:
                        limit_reached = (
                            self._memory_topic_read_count
                            >= self.max_memory_topic_reads
                        )
                    if limit_reached:
                        return self._memory_topic_read_limit_outcome(step, tool_call)

            if tool_call.argument_parse_error is not None:
                return self._malformed_tool_arguments_outcome(step, tool_call)

            raw_arguments = dict(tool_call.arguments)
            try:
                admission = self.tools.admit(tool_call.name, raw_arguments)
            except ValidationError as exc:
                return self._invalid_tool_arguments_outcome(step, tool_call, exc)

            tool_call.arguments = deepcopy(admission.arguments)
            self.trace_writer.write_event(
                "tool_arguments_validated",
                step=step,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                argument_keys=sorted(admission.arguments),
                risk_level=admission.risk_level.value,
                command_category=(
                    admission.command_policy.category.value
                    if admission.command_policy is not None
                    else None
                ),
                requires_approval=admission.requires_approval,
                allowed=admission.allowed,
            )
            decision = self.hook_manager.emit(
                HookEvent(
                    name="pre_tool_use",
                    run_id=self.run_id,
                    task=self.task,
                    workspace=self.workspace,
                    step=step,
                    payload={
                        "loop": self.hook_owner,
                        "tool_call": tool_call,
                        "command_policy": admission.command_policy,
                        "available_tool_names": available_tool_names,
                    },
                )
            )
            hook_outcome = self._hook_outcome(step, tool_call, decision)
            if hook_outcome is not None:
                return hook_outcome

            tool_call.arguments = deepcopy(admission.arguments)
            repeated_failure = self._same_failed_command_outcome(
                step,
                tool_call,
                workspace_generation=workspace_generation,
            )
            if repeated_failure is not None:
                return repeated_failure
            reused = self._runtime_reuse_outcome(
                step,
                tool_call,
                workspace_generation=workspace_generation,
            )
            if reused is not None:
                return reused
            read_overlap_detected = self._reuse_tracker.overlap_detected(
                tool_call,
                workspace_generation=workspace_generation,
            )

            expected_file_sha256, stale_write = self._prepare_write_freshness(
                step,
                tool_call,
            )
            if stale_write is not None:
                return stale_write

            mutation_preview = (
                self.tools.preview_admitted(admission)
                if admission.name in {"edit", "write", "apply_patch"}
                else None
            )
            approval_decision = self._request_approval(
                step,
                tool_call,
                admission,
                preview=mutation_preview,
            )
            if approval_decision is not None:
                status = f"approval_{approval_decision.value}"
                content = (
                    f"Tool {tool_call.name} was not executed because approval decision was "
                    f"{approval_decision.value}. Choose another safe action or explain the blocker."
                )
                observation = build_simple_observation(
                    tool_call,
                    content,
                    status=status,
                    metadata={"approval_fingerprint": tool_call_fingerprint(tool_call)},
                )
                self.trace_writer.write_event(
                    "tool_result",
                    step=step,
                    tool_call_id=tool_call.id,
                    tool=tool_call.name,
                    status=status,
                    reason=approval_decision.value,
                )
                return ToolExecutionOutcome(
                    observation=observation,
                    stop_reason=(
                        "approval_aborted"
                        if approval_decision == ApprovalDecision.ABORT
                        else None
                    ),
                )

            self._snapshot_write_targets(tool_call)
            journal_entry = self._journal_prepared(
                step=step,
                tool_call=tool_call,
                admission=admission,
            )
            result = self.tools.execute_admitted(
                admission,
                approval_granted=True,
                expected_file_sha256=expected_file_sha256,
            )
            self._journal_completed(journal_entry, result)
            observation, compression_events = build_observation(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                artifact_dir=self.artifact_dir,
                tool_arguments=tool_call.arguments,
            )
            for event in compression_events:
                self.trace_writer.write_event(
                    "context_compressed",
                    step=step,
                    **event.model_dump(mode="json"),
                )
            if isinstance(result, FileReadResult) and result.content_sha256:
                observation.metadata["content_sha256"] = result.content_sha256
            if read_overlap_detected:
                observation.metadata["overlap_detected"] = True
            self._annotate_command_lifecycle(result, observation)
            self._annotate_mutation_diff(
                result,
                observation,
                preview=mutation_preview,
            )
            if admission.command_policy is not None:
                observation.metadata["command_category"] = (
                    admission.command_policy.category.value
                )
            write_strategy = write_strategy_from_result(tool_call.name, result)
            if write_strategy is not None:
                observation.metadata["write_strategy"] = write_strategy
            self._reuse_tracker.record(
                tool_call,
                observation,
                workspace_generation=workspace_generation,
            )
            self._record_command_attempt(
                tool_call,
                observation,
                workspace_generation=workspace_generation,
            )
            self._record_memory_read(tool_call, observation)
            status = str(
                observation.metadata.get("status") or tool_result_status(observation)
            )
            self.trace_writer.write_event(
                "tool_result",
                step=step,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                status=status,
                truncated=observation.is_truncated,
                preview=observation.output_preview,
                artifact_path=observation.artifact_path,
                write_strategy=write_strategy,
            )
            return ToolExecutionOutcome(
                observation=observation,
                modified_files=modified_files_from_result(tool_call.name, result),
            )
        except StaleWriteError as exc:
            return self._stale_write_error_outcome(step, tool_call, str(exc))
        except Exception as exc:
            self.hook_manager.emit(
                HookEvent(
                    name="error",
                    run_id=self.run_id,
                    task=self.task,
                    workspace=self.workspace,
                    step=step,
                    payload={
                        "loop": self.hook_owner,
                        "error": exc,
                        "tool_call": tool_call,
                        "kind": "tool_error",
                    },
                )
            )
            contract = classify_tool_exception(exc)
            content = bounded_tool_error_message(exc, fallback=contract["message"])
            observation = build_simple_observation(
                tool_call,
                content,
                status="error",
                metadata={
                    "error_type": contract["error_type"],
                    "retryable": contract["retryable"],
                    "retry_hint": contract["retry_hint"],
                    "side_effect": contract["side_effect"],
                    "exception_type": type(exc).__name__,
                },
            )
            self.trace_writer.write_event(
                "tool_result",
                step=step,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                status="error",
                error_type=contract["error_type"],
                exception_type=type(exc).__name__,
                retryable=contract["retryable"],
                side_effect=contract["side_effect"],
                error=content,
            )
            return ToolExecutionOutcome(observation=observation)

    def _journal_prepared(
        self,
        *,
        step: int,
        tool_call: NormalizedToolCall,
        admission: ToolAdmission,
    ) -> ExecutionJournalEvent | None:
        if self.execution_journal is None or tool_call.name not in {
            "edit",
            "write",
            "apply_patch",
            "run_command",
        }:
            return None
        target_paths = write_target_paths(tool_call)
        effect_kind = "command" if tool_call.name == "run_command" else "filesystem"
        before_hashes = (
            digest_workspace_files(self.workspace, target_paths)
            if target_paths
            else {}
        )
        expected_after_hashes = expected_after_hashes_for_tool(
            self.workspace,
            tool_call,
            target_paths,
        )
        return self.execution_journal.append_prepared(
            entry_id=f"{step}:{tool_call.id}",
            run_id=self.run_id,
            step=step,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            argument_fingerprint=tool_call_fingerprint(tool_call),
            effect_kind=effect_kind,
            target_paths=target_paths,
            before_hashes=before_hashes,
            expected_after_hashes=expected_after_hashes,
            command_category=(
                admission.command_policy.category.value
                if admission.command_policy is not None
                else None
            ),
        )

    def _journal_completed(
        self,
        prepared: ExecutionJournalEvent | None,
        result: Any,
    ) -> None:
        if self.execution_journal is None or prepared is None:
            return
        after_hashes = (
            digest_workspace_files(self.workspace, prepared.target_paths)
            if prepared.target_paths
            else {}
        )
        lifecycle_status = getattr(result, "lifecycle_status", None)
        self.execution_journal.append_completed(
            prepared,
            after_hashes=after_hashes,
            result_status=(
                str(lifecycle_status)
                if isinstance(lifecycle_status, str)
                else "completed"
            ),
            returncode=(
                int(result.returncode)
                if isinstance(getattr(result, "returncode", None), int)
                else None
            ),
        )

    def synchronize_reuse_with_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        observations: list[ContextObservation],
        run_state: RunState,
        workspace_generation: int,
    ) -> None:
        """Keep reuse entries only while their source result remains model-visible."""

        self._reuse_tracker.synchronize(
            messages,
            observations=observations,
            run_state=run_state,
            workspace_generation=workspace_generation,
        )

    def rollback_unfinished_changes(self, *, step: int, reason: str) -> RollbackOutcome:
        if not self.rollback_on_unfinished_stop or not self._write_snapshots:
            return RollbackOutcome(performed=False)
        restored: list[str] = []
        deleted: list[str] = []
        errors: list[str] = []
        for relative_path, snapshot in self._write_snapshots.items():
            path = safe_workspace_file(self.workspace, relative_path)
            if path is None:
                errors.append(f"unsafe:{relative_path}")
                continue
            try:
                if snapshot.existed:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(snapshot.content or b"")
                    restored.append(relative_path)
                else:
                    if path.exists() and path.is_file():
                        path.unlink()
                    deleted.append(relative_path)
            except Exception as exc:
                errors.append(f"{relative_path}: {type(exc).__name__}: {exc}")
        self.trace_writer.write_event(
            "workspace_rollback",
            step=step,
            reason=reason,
            restored=restored,
            deleted=deleted,
            errors=errors,
        )
        return RollbackOutcome(
            performed=True,
            restored=tuple(restored),
            deleted=tuple(deleted),
            errors=tuple(errors),
        )

    def emit_tool_call_started(self, step: int, tool_call: NormalizedToolCall) -> None:
        handler = getattr(self.output_sink, "tool_call_started", None)
        if handler is not None:
            handler(
                step=step,
                tool_name=tool_call.name,
                arguments=dict(tool_call.arguments or {}),
                tool_call_id=tool_call.id,
            )

    def _hook_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        decision: HookDecision,
    ) -> ToolExecutionOutcome | None:
        if decision.action == "allow" or decision.observation is None:
            return None
        observation = decision.observation
        status = str(observation.metadata.get("status") or decision.action)
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status=status,
            hook=decision.hook_name,
            reason=decision.reason,
            preview=observation.output_preview,
            artifact_path=observation.artifact_path,
        )
        return ToolExecutionOutcome(observation=observation)

    def _unavailable_tool_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        available_tool_names: tuple[str, ...],
    ) -> ToolExecutionOutcome:
        available = ", ".join(available_tool_names) if available_tool_names else "none"
        content = (
            f"Tool {tool_call.name} is not available in this turn. "
            f"Available tools: {available}. No tool was executed."
        )
        observation = build_simple_observation(
            tool_call,
            content,
            status="tool_unavailable",
            metadata={
                "error_type": "tool_unavailable",
                "retryable": False,
                "retry_hint": "Use one of the available tools or change the plan.",
                "side_effect": "none",
                "available_tools": list(available_tool_names),
            },
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status="tool_unavailable",
            available_tools=list(available_tool_names),
        )
        return ToolExecutionOutcome(observation=observation)

    def _memory_topic_read_limit_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
    ) -> ToolExecutionOutcome:
        content = (
            f"Repository-memory Topic read limit reached ({self.max_memory_topic_reads} per user turn). "
            "Use the Topics already read and continue the task; do not retry another Topic."
        )
        observation = build_simple_observation(
            tool_call,
            content,
            status="memory_topic_read_limit",
            metadata={
                "limit": self.max_memory_topic_reads,
                "successful_topic_reads": self._memory_topic_read_count,
            },
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status="memory_topic_read_limit",
            limit=self.max_memory_topic_reads,
            successful_topic_reads=self._memory_topic_read_count,
        )
        return ToolExecutionOutcome(observation=observation)

    def _malformed_tool_arguments_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
    ) -> ToolExecutionOutcome:
        raw_arguments = tool_call.raw_arguments or ""
        raw_argument_sha256 = hashlib.sha256(raw_arguments.encode("utf-8")).hexdigest()
        fingerprint = malformed_tool_call_fingerprint(tool_call)
        with self._state_lock:
            repeat_count = self._malformed_tool_call_counts.get(fingerprint, 0) + 1
            self._malformed_tool_call_counts[fingerprint] = repeat_count

            output_budget_raised = False
            if tool_call.arguments_likely_truncated and repeat_count == 1:
                if self.output_budget.current < self.output_budget.maximum:
                    previous = self.output_budget.current
                    self.output_budget.current = min(
                        self.output_budget.maximum,
                        max(previous * 2, previous + 1024),
                    )
                    output_budget_raised = True
                    self.trace_writer.write_event(
                        "model_recovery",
                        step=step,
                        action="increase_max_output_tokens_for_malformed_tool_call",
                        previous_max_output_tokens=previous,
                        max_output_tokens=self.output_budget.current,
                        tool=tool_call.name,
                    )

        lines = [
            f"Tool {tool_call.name} received malformed JSON arguments from the model provider.",
            "",
            f"JSON parse error: {tool_call.argument_parse_error or 'unknown parse error'}",
            f"Raw argument length: {len(raw_arguments)} characters.",
            "",
            "No Hook, approval, snapshot, or tool execution occurred.",
        ]
        if tool_call.arguments_likely_truncated:
            lines.append(
                "The provider response likely ended before the tool-call JSON completed."
            )
        if output_budget_raised:
            lines.append(
                f"The next model call may use up to {self.output_budget.current} output tokens."
            )
        if repeat_count == 1:
            lines.extend(
                [
                    "Retry with shorter valid JSON arguments.",
                    "Do not embed large binary, base64, or long escaped byte content in one tool call; "
                    "split the work into smaller calls or generate compact text fixtures instead.",
                ]
            )
        elif repeat_count < MAX_REPEATED_MALFORMED_TOOL_CALLS:
            lines.extend(
                [
                    f"This is repeated malformed attempt {repeat_count} for the same tool-call prefix.",
                    "Do not repeat the same call. Change strategy, substantially shorten the payload, "
                    "or use multiple smaller tool calls.",
                ]
            )
        else:
            lines.extend(
                [
                    f"The malformed tool-call limit ({MAX_REPEATED_MALFORMED_TOOL_CALLS}) was reached.",
                    "The run is stopping to prevent an invalid retry loop.",
                ]
            )

        status = (
            "malformed_tool_arguments_repeated"
            if repeat_count >= MAX_REPEATED_MALFORMED_TOOL_CALLS
            else "malformed_tool_arguments"
        )
        content = "\n".join(lines)
        observation = build_simple_observation(
            tool_call,
            content,
            status=status,
            metadata={
                "error_type": "malformed_tool_arguments",
                "retryable": repeat_count < MAX_REPEATED_MALFORMED_TOOL_CALLS,
                "retry_hint": (
                    "Retry with substantially shorter valid JSON arguments."
                    if repeat_count < MAX_REPEATED_MALFORMED_TOOL_CALLS
                    else "Do not retry the same malformed call."
                ),
                "side_effect": "none",
                "parse_error": tool_call.argument_parse_error,
                "raw_argument_chars": len(raw_arguments),
                "raw_argument_sha256": raw_argument_sha256,
                "likely_truncated": tool_call.arguments_likely_truncated,
                "repeat_count": repeat_count,
                "fingerprint": fingerprint,
                "output_budget_raised": output_budget_raised,
            },
        )
        self.trace_writer.write_event(
            "tool_arguments_malformed",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            parse_error=tool_call.argument_parse_error,
            raw_argument_chars=len(raw_arguments),
            raw_argument_sha256=raw_argument_sha256,
            likely_truncated=tool_call.arguments_likely_truncated,
            repeat_count=repeat_count,
            fingerprint=fingerprint,
            output_budget_raised=output_budget_raised,
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status=status,
        )
        return ToolExecutionOutcome(
            observation=observation,
            stop_reason=(
                "repeated_malformed_tool_call"
                if repeat_count >= MAX_REPEATED_MALFORMED_TOOL_CALLS
                else None
            ),
        )

    def _invalid_tool_arguments_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        error: ValidationError,
    ) -> ToolExecutionOutcome:
        errors = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        locations = [format_validation_location(item.get("loc")) for item in errors]
        expected = tool_argument_expectations(self.tools, tool_call.name)
        lines = [f"Tool {tool_call.name} received invalid arguments.", "", "Expected:"]
        lines.extend(f"- {item}" for item in expected)
        lines.extend(["", "Validation errors:"])
        lines.extend(
            f"- {location}: {item.get('msg', 'Invalid value')}"
            for location, item in zip(locations, errors)
        )
        lines.extend(
            [
                "",
                "No Hook, approval, snapshot, or tool execution occurred.",
                "Correct the arguments and call the tool again.",
            ]
        )
        content = "\n".join(lines)
        observation = build_simple_observation(
            tool_call,
            content,
            status="invalid_tool_arguments",
            metadata={
                "error_type": "invalid_tool_arguments",
                "retryable": True,
                "retry_hint": "Correct the reported fields and call the tool again.",
                "side_effect": "none",
                "error_count": len(errors),
                "error_locations": locations,
            },
        )
        self.trace_writer.write_event(
            "tool_arguments_invalid",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            error_count=len(errors),
            error_locations=locations,
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status="invalid_tool_arguments",
        )
        return ToolExecutionOutcome(observation=observation)

    def _prepare_write_freshness(
        self,
        step: int,
        tool_call: NormalizedToolCall,
    ) -> tuple[str | None, ToolExecutionOutcome | None]:
        if tool_call.name != "write" or not bool(tool_call.arguments.get("overwrite", False)):
            return None, None
        raw_path = str(tool_call.arguments.get("path") or "").strip()
        if not raw_path:
            return None, None
        target = WorkspaceGuard(self.workspace).resolve(raw_path)
        if not target.exists():
            return None, None
        expected = self._expected_overwrite_hash(raw_path)
        if expected is None:
            return None, self._stale_write_error_outcome(
                step,
                tool_call,
                "Existing file overwrite requires a successful full workspace read first.",
                reason="read_required_before_overwrite",
            )
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            return None, self._stale_write_error_outcome(
                step,
                tool_call,
                "Target file changed after it was read; read it again before overwriting.",
                reason="file_changed_since_read",
            )
        return expected, None

    def _expected_overwrite_hash(self, path: str) -> str | None:
        normalized = normalize_workspace_path(path)
        for inspected in reversed(self.run_state.inspected_files):
            if normalize_workspace_path(inspected.path) != normalized:
                continue
            if inspected.content_sha256 is None:
                return None
            if inspected.content_status not in {
                "full_content_available_in_context",
                "full_content_available_via_artifact",
            }:
                return None
            return inspected.content_sha256
        return None

    def _stale_write_error_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        message: str,
        *,
        reason: str = "stale_write_conflict",
    ) -> ToolExecutionOutcome:
        content = (
            f"Tool {tool_call.name} was not executed: {message} "
            "No workspace write occurred. Read the complete target file again before retrying."
        )
        observation = build_simple_observation(
            tool_call,
            content,
            status="stale_write",
            metadata={
                "error_type": "stale_write",
                "retryable": True,
                "retry_hint": "Read the complete target file again before retrying the overwrite.",
                "side_effect": "none",
                "reason": reason,
                "path": str(tool_call.arguments.get("path") or ""),
            },
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status="stale_write",
            reason=reason,
            path=str(tool_call.arguments.get("path") or ""),
        )
        return ToolExecutionOutcome(observation=observation)

    def _request_approval(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        admission: ToolAdmission,
        *,
        preview: dict[str, Any] | None = None,
    ) -> ApprovalDecision | None:
        if not admission.requires_approval:
            return None
        session_grant = self._command_session_grant(tool_call)
        if (
            session_grant is not None
            and self.session_memory is not None
            and self.session_memory.has_command_approval_grant(session_grant)
        ):
            self.trace_writer.write_event(
                "approval_session_grant_used",
                step=step,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                executable=self._command_session_executable(tool_call),
                grant_id=session_grant,
            )
            return None
        permission_decision = decide_permission(
            tool_name=tool_call.name,
            requires_approval=admission.requires_approval,
            is_command=admission.command_policy is not None,
            approval_policy=self.approval_policy,
            permission_mode=self.permission_mode,
        )
        self.trace_writer.write_event(
            "permission_decision",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            approval_policy=self.approval_policy.value,
            permission_mode=self.permission_mode.value,
            decision=permission_decision.value,
        )
        if permission_decision == PermissionDecision.ALLOW:
            return None
        if permission_decision == PermissionDecision.DENY:
            return ApprovalDecision.REJECT
        fingerprint = tool_call_fingerprint(tool_call)
        if fingerprint in self._denied_tool_calls:
            self.trace_writer.write_event(
                "approval_repeat_blocked",
                step=step,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
            )
            return ApprovalDecision.REJECT
        preview = preview or self.tools.preview_admitted(admission)
        if is_edit_tool(tool_call.name):
            if int(preview.get("matches") or 0) != 1:
                raise ValueError(
                    "edit requires old_text to match exactly once before approval; "
                    f"matched {preview.get('matches', 0)} location(s)."
                )
            if not str(preview.get("diff") or "").strip():
                raise ValueError("edit would not change the target file.")
        if tool_call.name == "run_command" and not preview.get("allowed", False):
            raise PermissionError(preview.get("reason") or "Command is not allowed.")
        request = ApprovalRequest(
            id=f"step_{step:04d}_{tool_call.id}",
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            risk_level=admission.risk_level.value,
            step=step,
            arguments=tool_call.arguments,
            preview=preview,
            can_approve_session=session_grant is not None,
        )
        self.approval_store.save_pending(request)
        self.trace_writer.write_event(
            "approval_required",
            step=step,
            approval_id=request.id,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            risk_level=request.risk_level,
            preview_summary=preview.get("summary", preview),
            pending_path=str(self.approval_store.pending_path),
        )
        response = self.approval_client.decide(request)
        self.trace_writer.write_event(
            "approval_resolved",
            step=step,
            approval_id=request.id,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            decision=response.decision.value,
            reason=response.reason,
        )
        self.approval_store.clear_pending()
        if response.decision == ApprovalDecision.APPROVE:
            return None
        if response.decision == ApprovalDecision.APPROVE_SESSION:
            if session_grant is None or self.session_memory is None:
                return ApprovalDecision.REJECT
            self.session_memory.grant_command_approval(session_grant)
            self.trace_writer.write_event(
                "approval_session_granted",
                step=step,
                approval_id=request.id,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                executable=self._command_session_executable(tool_call),
                grant_id=session_grant,
            )
            return None
        with self._state_lock:
            self._denied_tool_calls.add(fingerprint)
        return response.decision

    def _command_session_grant(self, tool_call: NormalizedToolCall) -> str | None:
        if tool_call.name != "run_command" or self.session_memory is None:
            return None
        argv = tool_call.arguments.get("argv")
        if not isinstance(argv, list) or not argv:
            return None
        return resolve_command_session_grant(self.workspace, argv)

    def _command_session_executable(self, tool_call: NormalizedToolCall) -> str | None:
        argv = tool_call.arguments.get("argv")
        if not isinstance(argv, list) or not argv:
            return None
        return resolve_command_executable_identity(self.workspace, str(argv[0]))

    def _same_failed_command_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        *,
        workspace_generation: int,
    ) -> ToolExecutionOutcome | None:
        if tool_call.name != "run_command":
            return None
        identity = command_identity(tool_call.arguments)
        if identity is None:
            return None
        with self._state_lock:
            previous = self._failed_command_attempts.get(identity)
        if previous is None:
            return None
        previous_generation, previous_returncode = previous
        if previous_generation != workspace_generation:
            return None
        command = render_argv(identity)
        content = (
            "The same command already failed and no workspace files changed afterward. "
            "The command was not executed again. Inspect the previous failure and modify the "
            "relevant code or test, or finish with the blocker before requesting this command again."
        )
        observation = build_simple_observation(
            tool_call,
            content,
            status="same_failed_command_without_workspace_change",
            metadata={
                "command": command,
                "argv": list(identity),
                "command_identity": list(identity),
                "previous_returncode": previous_returncode,
                "workspace_generation": workspace_generation,
            },
        )
        self.trace_writer.write_event(
            "command_execution_blocked",
            step=step,
            tool_call_id=tool_call.id,
            command=command,
            argv=list(identity),
            previous_returncode=previous_returncode,
            workspace_generation=workspace_generation,
            reason="same_failed_command_without_workspace_change",
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status="same_failed_command_without_workspace_change",
            reason="same_failed_command_without_workspace_change",
            preview=observation.output_preview,
        )
        return ToolExecutionOutcome(observation=observation)

    def _runtime_reuse_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        *,
        workspace_generation: int,
    ) -> ToolExecutionOutcome | None:
        match = self._reuse_tracker.lookup(
            tool_call,
            workspace_generation=workspace_generation,
        )
        if isinstance(match, ReadReuseMatch):
            return self._read_reuse_outcome(
                step,
                tool_call,
                match,
                workspace_generation=workspace_generation,
            )
        if isinstance(match, SearchReuseMatch):
            return self._search_reuse_outcome(
                step,
                tool_call,
                match,
                workspace_generation=workspace_generation,
            )
        return None

    def _read_reuse_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        match: ReadReuseMatch,
        *,
        workspace_generation: int,
    ) -> ToolExecutionOutcome:
        content = (
            "该源码范围已在本次运行中读取：\n"
            f"{match.path}:{match.covered_by[0]}-{match.covered_by[1]}\n\n"
            "当前请求：\n"
            f"{match.path}:{match.requested[0]}-{match.requested[1]}\n\n"
            "请使用已有消息中的结果，或读取尚未覆盖的范围。"
        )
        observation = build_simple_observation(
            tool_call,
            content,
            status="duplicate_reused",
            metadata={
                "path": match.path,
                "requested_range": list(match.requested),
                "covered_by": list(match.covered_by),
                "total_lines": match.total_lines,
                "source_tool_call_id": match.source_tool_call_id,
                "workspace_generation": workspace_generation,
            },
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status="duplicate_reused",
            path=match.path,
            requested_range=list(match.requested),
            covered_by=list(match.covered_by),
            source_tool_call_id=match.source_tool_call_id,
        )
        return ToolExecutionOutcome(observation=observation)

    def _search_reuse_outcome(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        match: SearchReuseMatch,
        *,
        workspace_generation: int,
    ) -> ToolExecutionOutcome:
        arguments = match.arguments
        content = (
            "相同搜索请求已在本次运行中执行。\n"
            f"source={arguments['source']!r}, query={arguments['query']!r}, "
            f"path={arguments['path']!r}\n"
            f"上次结果：{match.summary}\n"
            "请使用已有搜索结果，或调整搜索参数。"
        )
        observation = build_simple_observation(
            tool_call,
            content,
            status="duplicate_reused",
            metadata={
                "search_source": arguments["source"],
                "query": arguments["query"],
                "path": arguments["path"],
                "match_count": match.match_count,
                "files": list(match.files),
                "source_tool_call_id": match.source_tool_call_id,
                "workspace_generation": workspace_generation,
            },
        )
        self.trace_writer.write_event(
            "tool_result",
            step=step,
            tool_call_id=tool_call.id,
            tool=tool_call.name,
            status="duplicate_reused",
            source=arguments["source"],
            query=arguments["query"],
            path=arguments["path"],
            source_tool_call_id=match.source_tool_call_id,
        )
        return ToolExecutionOutcome(observation=observation)

    def _annotate_mutation_diff(
        self,
        result: Any,
        observation: ContextObservation,
        *,
        preview: dict[str, Any] | None,
    ) -> None:
        if observation.tool_name not in {"edit", "write", "apply_patch"}:
            return
        if preview is None or getattr(result, "changed", True) is False:
            return
        diff_preview, truncated = bounded_diff_preview(preview)
        if diff_preview is None:
            return
        observation.metadata["diff_preview"] = diff_preview
        observation.metadata["diff_truncated"] = truncated

    def _annotate_command_lifecycle(
        self,
        result: Any,
        observation: ContextObservation,
    ) -> None:
        if observation.tool_name != "run_command":
            return
        if isinstance(result, dict) and result.get("runtime_task_id"):
            observation.metadata["status"] = "background_started"
            observation.metadata["command_status"] = "background_started"
            observation.metadata["runtime_task_id"] = result.get("runtime_task_id")
            return
        lifecycle_status = getattr(result, "lifecycle_status", None)
        if lifecycle_status is None:
            return
        observation.metadata["command_status"] = lifecycle_status
        duration_seconds = getattr(result, "duration_seconds", None)
        if isinstance(duration_seconds, (int, float)):
            observation.metadata["duration_ms"] = max(0, round(float(duration_seconds) * 1000))
        if lifecycle_status == "completed":
            return
        status = {
            "failed": "command_failed",
            "timed_out": "command_timed_out",
            "cancelled": "command_cancelled",
        }.get(str(lifecycle_status), "command_failed")
        observation.metadata["status"] = status
        observation.metadata["error_type"] = status
        observation.metadata["side_effect"] = "possible"
        observation.metadata["retryable"] = lifecycle_status != "cancelled"
        observation.metadata["retry_hint"] = (
            "Do not retry unless the user asks to continue after cancellation."
            if lifecycle_status == "cancelled"
            else "Inspect the command output, change the workspace or command, then retry intentionally."
        )
        failure_text = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
        files = extract_source_paths(failure_text)
        if files:
            observation.metadata["failure_files"] = files
            observation.summary = (
                f"{observation.summary} Failure file(s): {', '.join(files[:5])}."
            )

    def _snapshot_write_targets(self, tool_call: NormalizedToolCall) -> None:
        if not self.rollback_on_unfinished_stop:
            return
        with self._state_lock:
            for relative_path in write_target_paths(tool_call):
                if relative_path in self._write_snapshots:
                    continue
                path = safe_workspace_file(self.workspace, relative_path)
                if path is None:
                    continue
                existed = path.exists()
                self._write_snapshots[relative_path] = WorkspaceFileSnapshot(
                    path=relative_path,
                    existed=existed,
                    content=path.read_bytes() if existed and path.is_file() else None,
                )
                self.trace_writer.write_event(
                    "rollback_snapshot_saved",
                    tool_call_id=tool_call.id,
                    tool=tool_call.name,
                    path=relative_path,
                    existed=existed,
                )

    def _restore_failed_command_attempts(
        self,
        observations: list[ContextObservation],
    ) -> None:
        for observation in observations:
            if observation.tool_name != "run_command":
                continue
            raw_identity = observation.metadata.get("command_identity")
            if not (
                isinstance(raw_identity, list)
                and raw_identity
                and all(isinstance(item, str) for item in raw_identity)
            ):
                continue
            generation = coerce_optional_int(
                observation.metadata.get("workspace_generation")
            )
            returncode = coerce_optional_int(observation.metadata.get("returncode"))
            if generation is None or returncode is None:
                continue
            identity = tuple(raw_identity)
            if returncode == 0:
                self._failed_command_attempts.pop(identity, None)
            else:
                self._failed_command_attempts[identity] = (generation, returncode)

    def _record_command_attempt(
        self,
        tool_call: NormalizedToolCall,
        observation: ContextObservation,
        *,
        workspace_generation: int,
    ) -> None:
        if tool_call.name != "run_command":
            return
        identity = command_identity(tool_call.arguments)
        if identity is None:
            return
        observation.metadata["command_identity"] = list(identity)
        returncode = coerce_optional_int(observation.metadata.get("returncode"))
        if returncode is None:
            return
        with self._state_lock:
            if returncode == 0:
                self._failed_command_attempts.pop(identity, None)
            else:
                self._failed_command_attempts[identity] = (
                    workspace_generation,
                    returncode,
                )

    def _record_memory_read(
        self,
        tool_call: NormalizedToolCall,
        observation: ContextObservation,
    ) -> None:
        status = str(
            observation.metadata.get("status") or tool_result_status(observation)
        )
        if is_memory_read(tool_call.name, tool_call.arguments) and status == "ok":
            with self._state_lock:
                self._memory_topic_read_count += 1


def bounded_diff_preview(preview: dict[str, Any]) -> tuple[str | None, bool]:
    """Return one small unified-diff preview for TUI display only."""

    raw = str(preview.get("diff") or preview.get("diff_summary") or "").strip()
    if not raw:
        return None, False
    lines = raw.splitlines()
    selected = lines[:MAX_TUI_DIFF_PREVIEW_LINES]
    text = "\n".join(selected)
    truncated = len(lines) > len(selected)
    if len(text) > MAX_TUI_DIFF_PREVIEW_CHARS:
        text = text[: MAX_TUI_DIFF_PREVIEW_CHARS - 1] + "…"
        truncated = True
    return text, truncated


def classify_tool_exception(exc: Exception) -> dict[str, Any]:
    """Map implementation exceptions to a small stable model-facing error contract."""

    if isinstance(exc, FileNotFoundError):
        return {
            "error_type": "not_found",
            "retryable": True,
            "retry_hint": "Check the exact path or discover the resource before retrying.",
            "side_effect": "none",
            "message": "The requested resource was not found.",
        }
    if isinstance(exc, IsADirectoryError):
        return {
            "error_type": "is_directory",
            "retryable": True,
            "retry_hint": "Target a file or use file search to inspect the directory.",
            "side_effect": "none",
            "message": "The requested path is a directory, not a file.",
        }
    if isinstance(exc, WorkspaceAccessError):
        return {
            "error_type": "workspace_access_denied",
            "retryable": True,
            "retry_hint": "Choose a non-sensitive path inside the current workspace.",
            "side_effect": "none",
            "message": "The requested path is outside the allowed workspace boundary.",
        }
    if isinstance(exc, PermissionError):
        return {
            "error_type": "permission_denied",
            "retryable": False,
            "retry_hint": "Choose an allowed action or report the permission blocker.",
            "side_effect": "none",
            "message": "The operation was denied by a permission boundary.",
        }
    if isinstance(exc, UnicodeDecodeError):
        return {
            "error_type": "invalid_text_encoding",
            "retryable": False,
            "retry_hint": "Use a text resource with supported UTF-8 encoding or another inspection path.",
            "side_effect": "none",
            "message": "The resource could not be decoded as UTF-8 text.",
        }
    if isinstance(exc, TimeoutError):
        return {
            "error_type": "timeout",
            "retryable": True,
            "retry_hint": "Narrow the operation before retrying.",
            "side_effect": "unknown",
            "message": "The tool operation timed out.",
        }
    if isinstance(exc, ValueError):
        return {
            "error_type": "invalid_request",
            "retryable": True,
            "retry_hint": "Correct the request using the error message and retry once.",
            "side_effect": "none",
            "message": "The tool request was invalid.",
        }
    if isinstance(exc, OSError):
        return {
            "error_type": "io_error",
            "retryable": True,
            "retry_hint": "Re-inspect the target state before retrying.",
            "side_effect": "unknown",
            "message": "The tool encountered an I/O error.",
        }
    return {
        "error_type": "execution_failed",
        "retryable": True,
        "retry_hint": "Inspect the reported failure, change the approach, and avoid repeating the same call unchanged.",
        "side_effect": "unknown",
        "message": "The tool failed during execution.",
    }


def bounded_tool_error_message(exc: Exception, *, fallback: str) -> str:
    """Keep model-facing execution errors informative but bounded."""

    message = str(exc).strip() or fallback
    if len(message) <= MAX_TOOL_ERROR_MESSAGE_CHARS:
        return message
    return message[: MAX_TOOL_ERROR_MESSAGE_CHARS - 1] + "…"


def build_simple_observation(
    tool_call: NormalizedToolCall,
    content: str,
    *,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> ContextObservation:
    return ContextObservation(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=content,
        output_preview=content,
        token_estimate=estimate_tokens(content),
        summary=content,
        is_important=True,
        is_truncated=False,
        metadata={"status": status, **(metadata or {})},
    )


def tool_result_status(observation: ContextObservation) -> str:
    if (
        observation.tool_name == "run_command"
        and observation.metadata.get("returncode") not in (0, None)
    ):
        return "command_failed"
    return "ok"


def command_identity(arguments: dict[str, Any]) -> tuple[str, ...] | None:
    argv = arguments.get("argv")
    if not (
        isinstance(argv, list)
        and argv
        and all(isinstance(item, str) for item in argv)
    ):
        return None
    return tuple(argv)


def format_validation_location(location: Any) -> str:
    if not isinstance(location, (list, tuple)) or not location:
        return "<root>"
    return ".".join(str(part) for part in location)


def tool_argument_expectations(tools: ToolRegistry, tool_name: str) -> list[str]:
    schemas = tools.schemas([tool_name])
    if not schemas:
        return ["arguments matching the registered tool schema"]
    parameters = schemas[0].get("function", {}).get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    required = set(parameters.get("required", [])) if isinstance(parameters, dict) else set()
    expectations: list[str] = []
    for name, schema in properties.items():
        if not isinstance(schema, dict):
            continue
        value_type = str(schema.get("type") or "value")
        suffix = " (required)" if name in required else ""
        expectations.append(f"{name}: {value_type}{suffix}")
    return expectations or ["arguments matching the registered tool schema"]



def write_strategy_from_result(tool_name: str, result: Any) -> str | None:
    if is_edit_tool(tool_name):
        return "exact_edit"
    if tool_name == "apply_patch":
        return "patch"
    if tool_name != "write":
        return None
    model_dump = getattr(result, "model_dump", None)
    payload = model_dump(mode="json") if callable(model_dump) else result
    if isinstance(payload, dict) and payload.get("created"):
        return "full_write_create"
    return "full_write_overwrite"


def modified_files_from_result(tool_name: str, result: Any) -> list[str]:
    model_dump = getattr(result, "model_dump", None)
    payload = model_dump(mode="json") if callable(model_dump) else result
    if not isinstance(payload, dict):
        return []
    if tool_name == "apply_patch":
        return [str(path) for path in payload.get("files") or []]
    if tool_name in {"edit", "write"}:
        if payload.get("changed", True) is False:
            return []
        path = payload.get("path")
        return [str(path)] if path else []
    return []


def write_target_paths(tool_call: NormalizedToolCall) -> list[str]:
    if tool_call.name in {"edit", "write"}:
        path = str(tool_call.arguments.get("path") or "").strip().replace("\\", "/")
        return [path] if path else []
    if tool_call.name != "apply_patch":
        return []
    paths: list[str] = []
    for line in str(tool_call.arguments.get("patch") or "").splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw = line[4:].split("\t", maxsplit=1)[0].strip().strip('"')
        if raw == "/dev/null":
            continue
        path = re.sub(r"^[ab]/", "", raw).replace("\\", "/")
        if path and path not in paths:
            paths.append(path)
    return paths


def expected_after_hashes_for_tool(
    workspace: str | Path,
    tool_call: NormalizedToolCall,
    target_paths: list[str],
) -> dict[str, str | None]:
    """Return exact post-state hashes only when the mutation is deterministic here."""

    if tool_call.name == "write" and len(target_paths) == 1:
        content = str(tool_call.arguments.get("content") or "")
        return {
            target_paths[0]: hashlib.sha256(content.encode("utf-8")).hexdigest()
        }
    if tool_call.name == "edit" and len(target_paths) == 1:
        target = WorkspaceGuard(workspace).resolve(target_paths[0])
        current = target.read_text(encoding="utf-8")
        old_text = str(tool_call.arguments.get("old_text") or "")
        new_text = str(tool_call.arguments.get("new_text") or "")
        if old_text and current.count(old_text) == 1:
            updated = current.replace(old_text, new_text, 1)
            return {
                target_paths[0]: hashlib.sha256(updated.encode("utf-8")).hexdigest()
            }
    return {}


def tool_call_fingerprint(tool_call: NormalizedToolCall) -> str:
    payload = json.dumps(
        {"name": tool_call.name, "arguments": tool_call.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def malformed_tool_call_fingerprint(tool_call: NormalizedToolCall) -> str:
    raw_arguments = tool_call.raw_arguments or ""
    stable_prefix = re.sub(r"\s+", " ", raw_arguments[:1024]).strip()
    payload = f"{tool_call.name}\0{stable_prefix}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_workspace_file(workspace: str | Path, relative_path: str) -> Path | None:
    if (
        not relative_path
        or relative_path.startswith(("/", "~"))
        or ".." in Path(relative_path).parts
    ):
        return None
    root = Path(workspace).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate
