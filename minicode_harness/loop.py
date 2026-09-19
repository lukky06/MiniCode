"""Canonical MiniCode model/tool loop.

The runtime follows a compact provider-native tool-calling architecture:

1. keep one provider-native message history for the user turn;
2. rebuild a compact dynamic system prompt for every model call;
3. append native assistant tool calls and tool results to that history;
4. compact only when the message budget requires it;
5. stop on a final answer or a deterministic safety boundary.

Repository maps, test-context packs, controller state, evidence ledgers, and
sub-agent digests are not parallel model-visible context channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Callable

from minicode_harness.context import (
    ContextBuilder,
    ContextObservation,
    ContextPreparer,
    RepositoryRuleLoader,
    RepositoryRulesSnapshot,
    RunState,
    SessionCompactionState,
    initialize_run_state,
    mark_verification_failed,
    mark_verification_not_run,
    mark_verification_passed,
    mark_verification_rolled_back,
    record_inspected_file,
    render_repository_structure_card,
    render_tool_result_message,
)
from minicode_harness.context.token import estimate_tokens
from minicode_harness.hooks import HookDecision, HookEvent, HookManager
from minicode_harness.mcp import MCPManager
from minicode_harness.memory import MemorySnapshotStore, RepositoryMemoryStore
from minicode_harness.state import (
    ReplSessionMemory,
    UserInputClient,
    UserInputOption,
    UserInputRequest,
)
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.output import OutputSink
from minicode_harness.policy import (
    ApprovalPolicy,
    DEFAULT_APPROVAL_POLICY,
    DEFAULT_PERMISSION_MODE,
    PermissionMode,
    render_argv,
)
from minicode_harness.runtime import (
    CollaborationMode,
    DEFAULT_COLLABORATION_MODE,
    ModelRecoveryPolicy,
)
from minicode_harness.runtime.agent_setup import build_agent_components
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.model_step import ModelStepOutcome, ModelStepRunner
from minicode_harness.runtime.run_lifecycle import RunSnapshot
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.runtime.tool_batch import ToolBatchExecutor
from minicode_harness.runtime.tool_reuse import coerce_optional_int
from minicode_harness.runtime.tool_runtime import (
    MAX_TUI_DIFF_PREVIEW_CHARS,
    ToolExecutionOutcome,
    build_simple_observation,
)
from minicode_harness.skills import SkillLoader
from minicode_harness.storage import HarnessDataStore
from minicode_harness.state import (
    ApprovalClient,
    ApprovalStore,
    CheckpointStore,
    TaskListState,
    UserTurnState,
)
from minicode_harness.subagent import ReadonlySubagentRunner, SubagentResult
from minicode_harness.tools import CommandExecutor
from minicode_harness.trace import TraceWriter
from minicode_harness.workspace import (
    WorkspaceProfile,
    preferred_verification_command_for_paths,
    scan_workspace_profile,
    workspace_profile_may_change,
)


@dataclass(frozen=True)
class AgentLoopConfig:
    """Small set of controls for one canonical model/tool loop."""

    max_steps: int = 50
    max_tool_calls: int = 60
    start_step: int = 0
    rollback_on_unfinished_stop: bool = True
    repository_memory_enabled: bool = True
    enable_subagents: bool = True
    max_subagent_calls: int = 4
    max_subagent_duration_seconds: float = 180.0
    enable_worktree_workers: bool = False
    enable_progress_guidance: bool = False
    enable_repository_rules: bool = True
    hidden_tool_names: tuple[str, ...] = ()
    reserve_final_step: bool = False
    max_memory_topic_reads: int = 2
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    """Terminal outcome returned by the runtime."""

    status: str
    final_text: str | None
    steps: int
    tool_calls: int
    stop_reason: str
    stop_summary: str | None = None


class AgentLoop:
    """One bounded user turn containing one or more model calls."""

    def __init__(
        self,
        *,
        task: str,
        workspace: Path | str,
        model_client: ModelClient,
        trace_writer: TraceWriter,
        config: AgentLoopConfig | None = None,
        skill_names: list[str] | None = None,
        no_skills: bool = False,
        skill_loader: SkillLoader | None = None,
        memory_store: HarnessDataStore | None = None,
        data_dir: Path | str | None = None,
        repository_memory: RepositoryMemoryStore | None = None,
        memory_snapshot_hash: str | None = None,
        memory_snapshot_path: str | None = None,
        enable_long_term_context: bool = True,
        long_term_context: str | None = None,
        context_builder: ContextBuilder | None = None,
        context_preparer: ContextPreparer | None = None,
        repository_rule_loader: RepositoryRuleLoader | None = None,
        enable_write: bool = False,
        enable_command: bool = True,
        approval_policy: ApprovalPolicy = DEFAULT_APPROVAL_POLICY,
        permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE,
        collaboration_mode: CollaborationMode = DEFAULT_COLLABORATION_MODE,
        approval_client: ApprovalClient | None = None,
        user_input_client: UserInputClient | None = None,
        approval_store: ApprovalStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        initial_observations: list[ContextObservation] | None = None,
        initial_modified_files: list[str] | None = None,
        initial_run_state: RunState | None = None,
        initial_task_state: TaskListState | None = None,
        initial_message_history: list[dict[str, Any]] | None = None,
        initial_compaction_state: SessionCompactionState | None = None,
        initial_tool_calls: int = 0,
        provider: str = "unknown",
        model: str | None = None,
        hook_manager: HookManager | None = None,
        session_memory: ReplSessionMemory | None = None,
        output_sink: OutputSink | None = None,
        stream_model: bool = False,
        cancellation_token: CancellationToken | None = None,
        recovery_policy: ModelRecoveryPolicy | None = None,
        mcp_manager: MCPManager | None = None,
        mcp_config: Path | str | None = None,
        command_executor: CommandExecutor | None = None,
        subagent_model_client_factory: Callable[[], ModelClient] | None = None,
        steering_queue: SteeringQueue | None = None,
    ) -> None:
        self.task = task.strip()
        self.workspace = str(Path(workspace).resolve())
        self.model_client = model_client
        self.trace_writer = trace_writer
        self.run_id = run_id or self.trace_writer.trace_path.parent.name
        self.config = config or AgentLoopConfig()
        self._last_repository_rules_hash: str | None = None
        self.enable_write = enable_write
        self.enable_command = enable_command
        self.approval_policy = ApprovalPolicy(approval_policy)
        self.permission_mode = PermissionMode(permission_mode)
        self.collaboration_mode = CollaborationMode(collaboration_mode)
        self.user_input_client = user_input_client
        self.provider = provider.strip().lower() or "unknown"
        self.model = model or getattr(model_client, "model", None)
        self.session_memory = session_memory
        self.stream_model = stream_model
        self.subagent_model_client_factory = subagent_model_client_factory
        self.steering_queue = (
            steering_queue if steering_queue is not None else SteeringQueue()
        )
        self.repository_memory = repository_memory
        self.memory_snapshot_hash = memory_snapshot_hash
        self.memory_snapshot_path = memory_snapshot_path
        self.memory_snapshot_store = MemorySnapshotStore(
            self.trace_writer.trace_path.parent
        )
        self._reactive_compactions = 0
        self._output_recoveries = 0
        self._force_final_only_next_call = False
        self._subagent_calls = 0
        self._subagent_call_lock = Lock()

        self.observations = list(initial_observations or [])
        self.modified_files = list(initial_modified_files or [])
        self.run_state = initialize_run_state(initial_run_state)
        self.tool_call_count = initial_tool_calls
        restored_generations = [
            int(item.workspace_generation)
            for item in self.run_state.inspected_files
        ]
        restored_generations.extend(
            generation
            for observation in self.observations
            if (
                generation := coerce_optional_int(
                    observation.metadata.get("workspace_generation")
                )
            )
            is not None
        )
        self.workspace_generation = max(
            restored_generations,
            default=1 if self.modified_files else 0,
        )
        self._workspace_profile_cache: WorkspaceProfile | None = None
        self._repository_structure_card_cache: str | None = None
        self.preferred_verification_command = (
            preferred_verification_command_for_paths(
                self.workspace,
                self.modified_files,
                profile=self._workspace_profile_snapshot(),
            )
            if self.modified_files
            else None
        )
        restoring = initial_message_history is not None
        prior_messages = (
            list(initial_message_history)
            if restoring
            else (self.session_memory.load_message_history() if self.session_memory else [])
        )
        self.compaction_state = (
            initial_compaction_state.model_copy(deep=True)
            if initial_compaction_state is not None
            else (
                self.session_memory.load_compaction_state()
                if self.session_memory is not None
                else SessionCompactionState()
            )
        )
        self.user_turn = UserTurnState.create(
            turn_id=self.run_id,
            task=self.task,
            messages=prior_messages,
            model_call_count=self.config.start_step,
            append_task=not restoring,
        )
        if long_term_context is not None:
            self.long_term_context = long_term_context
        elif (
            enable_long_term_context
            and self.config.repository_memory_enabled
            and self.repository_memory is not None
        ):
            self.long_term_context = self.repository_memory.render_index()
        else:
            self.long_term_context = ""

        components = build_agent_components(
            task=self.task,
            workspace=self.workspace,
            model_client=self.model_client,
            trace_writer=self.trace_writer,
            run_id=self.run_id,
            provider=self.provider,
            model=self.model,
            enable_repository_rules=self.config.enable_repository_rules,
            repository_rule_loader=repository_rule_loader,
            skill_names=skill_names,
            no_skills=no_skills,
            skill_loader=skill_loader,
            memory_store=memory_store,
            data_dir=data_dir,
            repository_memory=self.repository_memory,
            repository_memory_enabled=self.config.repository_memory_enabled,
            context_builder=context_builder,
            context_preparer=context_preparer,
            enable_write=self.enable_write,
            enable_command=self.enable_command,
            approval_policy=self.approval_policy,
            permission_mode=self.permission_mode,
            approval_client=approval_client,
            approval_store=approval_store,
            request_user_input_handler=(
                self._request_user_input
                if self.collaboration_mode == CollaborationMode.PLAN
                and self.user_input_client is not None
                else None
            ),
            checkpoint_store=checkpoint_store,
            hook_manager=hook_manager,
            hook_owner=self,
            session_memory=self.session_memory,
            output_sink=output_sink,
            cancellation_token=cancellation_token,
            recovery_policy=recovery_policy,
            mcp_manager=mcp_manager,
            mcp_config=mcp_config,
            command_executor=command_executor,
            subagent_handler=self._run_subagent,
            memory_topic_reader=(
                self._read_memory_topic if self.repository_memory is not None else None
            ),
            initial_task_state=initial_task_state,
            initial_observations=self.observations,
            run_state=self.run_state,
            workspace_generation=self.workspace_generation,
            max_memory_topic_reads=self.config.max_memory_topic_reads,
            rollback_on_unfinished_stop=self.config.rollback_on_unfinished_stop,
            enable_subagents=self.config.enable_subagents,
            enable_worktree_workers=self.config.enable_worktree_workers,
            enable_progress_guidance=self.config.enable_progress_guidance,
            max_steps=self.config.max_steps,
            start_step=self.config.start_step,
            emit_lifecycle_hook=lambda name, step, payload: self._emit_lifecycle_hook(
                name,
                step=step,
                payload=payload,
            ),
        )
        self.repository_rule_loader = components.repository_rule_loader
        self.output_sink = components.output_sink
        self.cancellation_token = components.cancellation_token
        self.hook_manager = components.hook_manager
        self.recovery_policy = components.recovery_policy
        self.model_capabilities = components.model_capabilities
        self.output_budget = components.output_budget
        self.mcp_manager = components.mcp_manager
        self.artifact_dir = components.artifact_dir
        self.runtime_task_registry = components.runtime_task_registry
        self.background_command_manager = components.background_command_manager
        self.worktree_worker_manager = components.worktree_worker_manager
        self.task_store = components.task_store
        self.skill_loader = components.skill_loader
        self.available_skills = components.available_skills
        self.tools = components.tools
        self.data_dir = components.data_dir
        self.context_builder = components.context_builder
        self.context_preparer = components.context_preparer
        self.checkpoint_store = components.checkpoint_store
        self.approval_store = components.approval_store
        self.approval_client = components.approval_client
        self.tool_runtime = components.tool_runtime
        self.progress_policy = components.progress_policy
        self.lifecycle = components.lifecycle
        self.model_step = ModelStepRunner()
        self.tool_batch = ToolBatchExecutor()

        self._last_tool_schema_hash: str | None = None
        self._current_stream_emitted_final_text = False

    def run(self) -> AgentRunResult:
        """Execute the bounded model/tool loop."""

        self._emit_lifecycle_hook(
            "session_start",
            step=None,
            payload={"restored": self.config.start_step > 0},
        )
        for step in range(self.config.start_step + 1, self.config.max_steps + 1):
            self._inject_runtime_notifications()
            if self.cancellation_token.is_cancelled:
                return self._finish_cancelled_run(step=max(self.config.start_step, step - 1))
            prepared = self.model_step.prepare(self, step)
            if isinstance(prepared, AgentRunResult):
                return prepared

            model_outcome = self.model_step.request(self, step, prepared)
            if model_outcome.result is not None:
                return model_outcome.result
            if model_outcome.should_continue:
                continue
            response = model_outcome.response
            assert response is not None

            if response.is_final():
                final_outcome = self._accept_final_response(step, response)
                if final_outcome.result is not None:
                    return final_outcome.result
                if final_outcome.should_continue:
                    continue

            tool_result = self.tool_batch.execute(
                self,
                step,
                response,
                tool_names=prepared.tool_names,
            )
            if tool_result is not None:
                return tool_result
            self._consume_one_steering_message(step)

        return self._finish_stopped_run(
            step=self.config.max_steps,
            reason="max_steps",
            final_text=None,
        )

    def enqueue_steering(self, text: str) -> None:
        """Queue one runtime message for the next safe Tool Batch boundary."""

        self.steering_queue.enqueue(text)

    def _consume_one_steering_message(self, step: int) -> None:
        """Append at most one queued message after a complete Tool Batch."""

        if step >= self.config.max_steps or self.cancellation_token.is_cancelled:
            return
        message = self.steering_queue.dequeue()
        if message is None:
            return
        self.user_turn.append_message({"role": "user", "content": message})
        self.trace_writer.write_event(
            "steering_message_consumed",
            step=step,
            content=message,
            remaining=len(self.steering_queue),
        )
        self._persist_and_checkpoint(step, reason="steering_message")

    def _accept_final_response(
        self,
        step: int,
        response: ModelResponse,
    ) -> ModelStepOutcome:
        final_text = response.final_text or ""
        incomplete_reason = _incomplete_final_text_reason(final_text)
        if incomplete_reason is not None:
            if self._recover_incomplete_final_text(
                step,
                final_text,
                reason=incomplete_reason,
            ):
                self._persist_and_checkpoint(
                    step,
                    reason="incomplete_final_text_recovery",
                )
                return ModelStepOutcome(should_continue=True)
            return ModelStepOutcome(
                result=self._finish_stopped_run(
                    step=step,
                    reason="incomplete_final_text",
                    final_text=final_text,
                    rollback=False,
                )
            )

        self.user_turn.append_final_text(final_text)
        self.lifecycle.finish_completed(
            self._run_snapshot(step),
            final_text=final_text,
            emit_final_text=lambda: self._emit_accepted_final_text(final_text),
        )
        return ModelStepOutcome(
            result=AgentRunResult(
                status="completed",
                final_text=final_text,
                steps=step,
                tool_calls=self.tool_call_count,
                stop_reason="final_text",
            )
        )

    def _emit_lifecycle_hook(
        self,
        name: str,
        *,
        step: int | None,
        payload: dict[str, Any] | None = None,
    ) -> HookDecision:
        """Emit one non-recursive harness lifecycle event."""

        return self.hook_manager.emit(
            HookEvent(
                name=name,
                run_id=self.run_id,
                task=self.task,
                workspace=self.workspace,
                step=step,
                payload={"loop": self, **(payload or {})},
            )
        )

    def _recover_truncated_output(self, step: int, response: ModelResponse) -> bool:
        partial = (response.final_text or response.assistant_commentary or "").strip()
        if self.stream_model and self._current_stream_emitted_final_text:
            return self._append_output_continuation(
                step,
                partial,
                action="continue_visible_truncated_output",
            )

        if self.output_budget.current < self.output_budget.maximum:
            previous = self.output_budget.current
            self.output_budget.current = min(
                self.output_budget.maximum,
                max(previous * 2, previous + 1024),
            )
            self.trace_writer.write_event(
                "model_recovery",
                step=step,
                action="increase_max_output_tokens",
                previous_max_output_tokens=previous,
                max_output_tokens=self.output_budget.current,
            )
            return True
        return self._append_output_continuation(
            step,
            partial,
            action="continue_truncated_output",
        )

    def _recover_incomplete_final_text(
        self,
        step: int,
        partial: str,
        *,
        reason: str,
    ) -> bool:
        if self.stream_model and self._current_stream_emitted_final_text:
            return False
        if step >= self.config.max_steps:
            return False
        if self._output_recoveries >= self.recovery_policy.config.max_output_recoveries:
            return False
        self.user_turn.append_message({"role": "assistant", "content": partial.strip()})
        self._append_runtime_recovery(
            "上一份最终答案结构不完整。不得再调用工具；请从头重写一份完整、紧凑、"
            "可独立阅读的最终答案，不要复述修复说明，也不要留下未闭合的代码块。"
        )
        self._output_recoveries += 1
        self._force_final_only_next_call = True
        self.trace_writer.write_event(
            "model_recovery",
            step=step,
            action="rewrite_incomplete_final_text",
            attempt=self._output_recoveries,
            reason=reason,
            partial_chars=len(partial),
        )
        return True

    def _append_output_continuation(
        self,
        step: int,
        partial: str,
        *,
        action: str,
    ) -> bool:
        if self._output_recoveries >= self.recovery_policy.config.max_output_recoveries:
            return False
        if partial:
            self.user_turn.append_message({"role": "assistant", "content": partial})
        self._append_runtime_recovery(
            "Output token limit was reached. Continue directly from the previous text without "
            "recap or apology, and finish the remaining work in a compact form."
        )
        self._output_recoveries += 1
        self.trace_writer.write_event(
            "model_recovery",
            step=step,
            action=action,
            attempt=self._output_recoveries,
            partial_chars=len(partial),
        )
        return True

    def _run_subagent(self, task: str) -> SubagentResult:
        call_number = self._reserve_subagent_call()
        subagent_id = f"call_{call_number:02d}"
        started = time.monotonic()
        self.trace_writer.write_event(
            "subagent_started",
            subagent_id=subagent_id,
            task=task,
        )
        try:
            result = self._run_reserved_subagent(
                call_number,
                task,
                model_client=self._create_subagent_model_client(),
                mcp_manager=self.mcp_manager,
            )
        except Exception as exc:
            self.trace_writer.write_event(
                "subagent_finished",
                subagent_id=subagent_id,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            raise
        self.trace_writer.write_event(
            "subagent_finished",
            subagent_id=subagent_id,
            status=result.status,
            duration_ms=int((time.monotonic() - started) * 1000),
            trace_path=f"subagents/{subagent_id}/trace.jsonl",
        )
        return result

    def _reserve_subagent_call(self) -> int:
        with self._subagent_call_lock:
            if self._subagent_calls >= self.config.max_subagent_calls:
                raise RuntimeError(
                    f"Subagent call limit reached ({self.config.max_subagent_calls})."
                )
            self._subagent_calls += 1
            return self._subagent_calls

    def _create_subagent_model_client(self) -> ModelClient:
        if self.subagent_model_client_factory is not None:
            return self.subagent_model_client_factory()
        return self.model_client

    def _run_reserved_subagent(
        self,
        call_number: int,
        task: str,
        *,
        model_client: ModelClient,
        mcp_manager: MCPManager | None,
    ) -> SubagentResult:
        subagent_artifacts = self.artifact_dir / "subagents" / f"call_{call_number:02d}"
        subagent_artifacts.mkdir(parents=True, exist_ok=True)
        runner = ReadonlySubagentRunner(
            workspace=self.workspace,
            model_client=model_client,
            trace_writer=TraceWriter(subagent_artifacts / "trace.jsonl"),
            artifact_dir=subagent_artifacts,
            recovery_policy=self.recovery_policy,
            mcp_manager=mcp_manager,
            cancellation_token=self.cancellation_token,
            max_duration_seconds=self.config.max_subagent_duration_seconds,
        )
        return runner.run(task)

    def _inject_runtime_notifications(self) -> None:
        for notification in self.runtime_task_registry.drain_notifications():
            self.user_turn.append_message(
                {"role": "assistant", "content": notification}
            )
            self.trace_writer.write_event(
                "runtime_task_notification",
                content=notification,
            )

    def _record_invalid_model_response(self, step: int, response: ModelResponse) -> None:
        reason = response.invalid_reason or "invalid_model_response"
        content = f"Model response rejected by output protocol: {reason}."
        self.observations.append(
            ContextObservation(
                tool_call_id=f"invalid_model_response_step_{step}",
                tool_name="agent_loop",
                content=content,
                output_preview=content,
                token_estimate=estimate_tokens(content),
                summary=content,
                is_important=True,
                metadata={"status": reason, "step": step},
            )
        )
        self.trace_writer.write_event(
            "invalid_model_response",
            step=step,
            status=reason,
        )

    def _record_modified_files(self, modified_files: list[str]) -> None:
        if not modified_files:
            return
        if workspace_profile_may_change(modified_files):
            self._workspace_profile_cache = None
            self._repository_structure_card_cache = None
        self.workspace_generation += 1
        mark_verification_not_run(self.run_state)
        for path in modified_files:
            if path not in self.modified_files:
                self.modified_files.append(path)
        preferred = preferred_verification_command_for_paths(
            self.workspace,
            modified_files,
            profile=self._workspace_profile_snapshot(),
        )
        if preferred:
            self.preferred_verification_command = preferred
            self.trace_writer.write_event(
                "preferred_verification_command_locked",
                command=preferred,
                modified_files=list(modified_files),
            )

    def _advance_verification_state(
        self,
        tool_call: NormalizedToolCall,
        outcome: ToolExecutionOutcome,
    ) -> None:
        if tool_call.name != "run_command":
            return
        observation = outcome.observation
        argv = tool_call.arguments.get("argv")
        fallback_command = (
            render_argv(argv)
            if isinstance(argv, list) and all(isinstance(item, str) for item in argv)
            else ""
        )
        command = str(observation.metadata.get("command") or fallback_command)
        returncode = observation.metadata.get("returncode")
        if returncode == 0:
            mark_verification_passed(
                self.run_state,
                command=command,
                returncode=0,
            )
        elif returncode is not None:
            mark_verification_failed(
                self.run_state,
                command=command,
                returncode=int(returncode),
                reason=(
                    str(observation.metadata.get("command_status"))
                    if observation.metadata.get("command_status")
                    else None
                ),
            )

    def _record_run_state(
        self,
        step: int,
        tool_call: NormalizedToolCall,
        outcome: ToolExecutionOutcome,
    ) -> None:
        outcome.observation.metadata["workspace_generation"] = self.workspace_generation
        record_inspected_file(
            state=self.run_state,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            arguments=tool_call.arguments,
            observation=outcome.observation,
            step=step,
            workspace_generation=self.workspace_generation,
        )
        self.trace_writer.write_event(
            "run_state_updated",
            step=step,
            tool=tool_call.name,
            tool_call_id=tool_call.id,
            modified_files=len(self.modified_files),
            verification_status=self.run_state.verification.status,
            verification_returncode=self.run_state.verification.returncode,
            inspected_files=len(self.run_state.inspected_files),
            tool_calls=self.tool_call_count,
        )

    def _rollback_unfinished_changes(self, *, step: int, reason: str) -> None:
        rolled_back = self.tool_runtime.rollback_unfinished_changes(
            step=step,
            reason=reason,
        )
        if not rolled_back:
            return
        self.modified_files = []
        mark_verification_rolled_back(self.run_state, reason=reason)

    def _append_cancelled_tool_results(
        self,
        step: int,
        tool_calls: list[NormalizedToolCall],
    ) -> None:
        for tool_call in tool_calls:
            self.tool_call_count += 1
            self.output_sink.tool_call_started(
                step=step,
                tool_name=tool_call.name,
                arguments=dict(tool_call.arguments),
                tool_call_id=tool_call.id,
            )
            observation = self._simple_observation(
                tool_call,
                f"Tool {tool_call.name} was not executed because the run was cancelled.",
                status="cancelled",
            )
            self._emit_tool_call_finished(step, observation)
            self.observations.append(observation)
            self._append_tool_result_message(tool_call, observation)
            self.trace_writer.write_event(
                "tool_result",
                step=step,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                status="cancelled",
            )

    def _finish_cancelled_run(self, *, step: int) -> AgentRunResult:
        reason = "cancelled"
        self.lifecycle.finish_cancelled(self._run_snapshot(step))
        return AgentRunResult(
            status="cancelled",
            final_text=None,
            steps=step,
            tool_calls=self.tool_call_count,
            stop_reason=reason,
        )

    def _finish_stopped_run(
        self,
        *,
        step: int,
        reason: str,
        final_text: str | None,
        rollback: bool = True,
    ) -> AgentRunResult:
        stop_summary = self.lifecycle.stop_run(
            snapshot_factory=lambda: self._run_snapshot(step),
            reason=reason,
            final_text=final_text,
            rollback=(
                (lambda: self._rollback_unfinished_changes(step=step, reason=reason))
                if rollback
                else None
            ),
        )
        return AgentRunResult(
            status="stopped",
            final_text=final_text,
            steps=step,
            tool_calls=self.tool_call_count,
            stop_reason=reason,
            stop_summary=stop_summary,
        )

    def _persist_and_checkpoint(
        self,
        step: int,
        *,
        reason: str,
        status: str = "running",
    ) -> None:
        self.lifecycle.checkpoint_progress(
            self._run_snapshot(step),
            status=status,
            reason=reason,
        )

    def _run_snapshot(self, step: int) -> RunSnapshot:
        digest_paths = list(self.modified_files)
        for inspected in self.run_state.inspected_files:
            if inspected.path not in digest_paths:
                digest_paths.append(inspected.path)
        return RunSnapshot(
            step=step,
            messages=self.user_turn.snapshot_messages(),
            compaction_state=self.compaction_state,
            run_state=self.run_state,
            task_state=self.task_store.snapshot(),
            observations=list(self.observations),
            modified_files=list(self.modified_files),
            workspace_digest_paths=digest_paths,
            user_turn_id=self.user_turn.turn_id,
            model_call_count=self.user_turn.model_call_count,
            tool_calls=self.tool_call_count,
            memory_snapshot_hash=self.memory_snapshot_hash,
            memory_snapshot_path=self.memory_snapshot_path,
        )

    def _request_user_input(
        self,
        question: str,
        options: list[dict[str, str | None]],
    ) -> dict[str, object]:
        if self.user_input_client is None:
            raise RuntimeError("Structured user input is unavailable for this Run.")
        request = UserInputRequest(
            question=question,
            options=[UserInputOption.model_validate(option) for option in options],
        )
        self.trace_writer.write_event(
            "user_input_requested",
            request_id=request.id,
            question=request.question,
            option_labels=[option.label for option in request.options],
        )
        response = self.user_input_client.choose(request)
        if response.selected_index >= len(request.options):
            raise ValueError("User input response selected an option outside the request range.")
        selected = request.options[response.selected_index]
        self.trace_writer.write_event(
            "user_input_resolved",
            request_id=request.id,
            selected_index=response.selected_index,
            selected_label=selected.label,
        )
        return {
            "selected_index": response.selected_index,
            "selected_label": selected.label,
        }

    def _read_memory_topic(self, topic: str) -> dict[str, str]:
        if self.repository_memory is None:
            raise RuntimeError("Repository Memory is not configured.")
        snapshot = self.memory_snapshot_store.load(
            expected_hash=self.memory_snapshot_hash,
        )
        if snapshot is None:
            raise RuntimeError("Repository Memory Run snapshot is missing.")
        if topic not in snapshot.topic_hashes:
            raise FileNotFoundError(
                f"Memory Topic is not registered in the Run snapshot: {topic}"
            )
        payload = self.memory_snapshot_store.read_topic(topic)
        status = "run_start_snapshot"
        self.trace_writer.write_event(
            "memory_topic_snapshot_used",
            topic=topic,
            status=status,
            content_hash=payload.get("content_hash", ""),
        )
        return payload

    def _repository_rules_snapshot(self) -> RepositoryRulesSnapshot:
        if self.repository_rule_loader is None:
            return RepositoryRulesSnapshot()
        paths = [item.path for item in self.run_state.inspected_files]
        paths.extend(self.modified_files)
        snapshot = self.repository_rule_loader.load(paths=paths)
        if snapshot.content_hash != self._last_repository_rules_hash:
            self.trace_writer.write_event(
                "repository_rules_loaded",
                sources=list(snapshot.sources),
                skipped_sources=list(snapshot.skipped_sources),
                size_bytes=snapshot.size_bytes,
                truncated=snapshot.truncated,
                content_hash=snapshot.content_hash,
            )
            self._last_repository_rules_hash = snapshot.content_hash
        return snapshot

    def _workspace_profile_snapshot(self) -> WorkspaceProfile:
        if self._workspace_profile_cache is None:
            self._workspace_profile_cache = scan_workspace_profile(self.workspace)
        return self._workspace_profile_cache

    def _repository_structure_card(self) -> str:
        if self._repository_structure_card_cache is None:
            self._repository_structure_card_cache = render_repository_structure_card(
                self.workspace,
                profile=self._workspace_profile_snapshot(),
            )
        return self._repository_structure_card_cache

    def _append_assistant_tool_calls(self, response: ModelResponse) -> None:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    },
                }
                for call in response.tool_calls
            ],
        }
        if response.reasoning_content is not None:
            message["reasoning_content"] = response.reasoning_content
        self.user_turn.append_message(message)

    def _append_tool_result_message(
        self,
        tool_call: NormalizedToolCall,
        observation: ContextObservation,
    ) -> None:
        self.user_turn.append_message(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": render_tool_result_message(observation),
            }
        )

    def _append_runtime_recovery(self, content: str) -> None:
        if content.strip():
            self.user_turn.append_message(
                {
                    "role": "user",
                    "content": "[Runtime recovery]\n" + content.strip(),
                }
            )

    def _trace_tool_schema_if_changed(
        self,
        step: int,
        tool_schemas: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        payload = json.dumps(
            tool_schemas,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if digest == self._last_tool_schema_hash:
            return digest, []
        self._last_tool_schema_hash = digest
        self.trace_writer.write_event(
            "tool_schema_registered",
            step=step,
            schema_hash=digest,
            tool_count=len(tool_schemas),
            tools=tool_schemas,
        )
        return digest, tool_schemas

    def _emit_accepted_final_text(self, final_text: str) -> None:
        if self.stream_model and final_text and not self._current_stream_emitted_final_text:
            self.output_sink.model_text_delta(final_text)

    def _emit_tool_call_finished(self, step: int, observation: ContextObservation) -> None:
        status = str(observation.metadata.get("status") or "ok")
        self.output_sink.tool_call_finished(
            step=step,
            tool_name=observation.tool_name,
            status=status,
            tool_call_id=observation.tool_call_id,
            summary=_tool_ui_summary(observation),
            metadata=_tool_ui_metadata(observation, status=status),
        )

    def _append_tool_budget_results(
        self,
        step: int,
        tool_calls: list[NormalizedToolCall],
    ) -> None:
        """Pair unexecuted tool calls with deterministic budget-exhaustion results."""

        if not tool_calls:
            return
        self.trace_writer.write_event(
            "tool_call_budget_exhausted",
            step=step,
            max_tool_calls=self.config.max_tool_calls,
            tool_calls_used=self.tool_call_count,
            skipped_tool_call_ids=[tool_call.id for tool_call in tool_calls],
        )
        for tool_call in tool_calls:
            content = (
                f"Tool {tool_call.name} was not executed because the tool-call budget "
                "was exhausted. Do not request another tool; return the best final answer "
                "from the current evidence and state any remaining verification blocker."
            )
            observation = self._simple_observation(
                tool_call,
                content,
                status="tool_budget_exhausted",
                metadata={
                    "max_tool_calls": self.config.max_tool_calls,
                    "tool_calls_used": self.tool_call_count,
                },
            )
            self.trace_writer.write_event(
                "tool_result",
                step=step,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                status="tool_budget_exhausted",
                reason="max_tool_calls",
            )
            self._emit_lifecycle_hook(
                "post_tool_use",
                step=step,
                payload={"tool_call": tool_call, "observation": observation},
            )
            self._emit_tool_call_finished(step, observation)
            self.observations.append(observation)
            self._append_tool_result_message(tool_call, observation)

    def _simple_observation(
        self,
        tool_call: NormalizedToolCall,
        content: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContextObservation:
        return build_simple_observation(
            tool_call,
            content,
            status=status,
            metadata=metadata,
        )


_TOOL_UI_METADATA_KEYS = {
    "path",
    "root",
    "pattern",
    "query",
    "command",
    "returncode",
    "command_status",
    "duration_ms",
    "runtime_task_id",
    "timed_out",
    "cancelled",
    "files",
    "truncated",
    "truncation_reason",
    "scanned_entries",
    "created",
    "overwritten",
    "overwrite",
    "changed",
    "bytes_written",
    "write_strategy",
    "diff_preview",
    "diff_truncated",
    "error_type",
    "total_lines",
    "start_line",
    "end_line",
    "match_count",
}


def _tool_ui_summary(observation: ContextObservation, limit: int = 240) -> str | None:
    text = " ".join((observation.summary or observation.output_preview or "").split())
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _tool_ui_metadata(
    observation: ContextObservation,
    *,
    status: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _TOOL_UI_METADATA_KEYS:
        if key not in observation.metadata:
            continue
        value = observation.metadata[key]
        if isinstance(value, str):
            if key == "diff_preview":
                metadata[key] = (
                    value
                    if len(value) <= MAX_TUI_DIFF_PREVIEW_CHARS
                    else value[: MAX_TUI_DIFF_PREVIEW_CHARS - 1] + "…"
                )
            else:
                metadata[key] = value if len(value) <= 240 else value[:237] + "..."
        elif isinstance(value, list):
            metadata[key] = value[:20]
        else:
            metadata[key] = value
    if observation.artifact_path:
        metadata["artifact_path"] = observation.artifact_path
    return metadata


def _incomplete_final_text_reason(text: str) -> str | None:
    active_fence: tuple[str, int] | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        match = re.match(r"(`{3,}|~{3,})", stripped)
        if match is None:
            continue
        marker = match.group(1)
        fence = (marker[0], len(marker))
        if active_fence is None:
            active_fence = fence
        elif fence[0] == active_fence[0] and fence[1] >= active_fence[1]:
            active_fence = None
    return "unclosed_markdown_fence" if active_fence is not None else None
