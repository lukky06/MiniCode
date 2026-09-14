"""Public execution boundary for one MiniCode run."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from minicode_harness.context import (
    ContextPreparer,
    LLMSemanticHistoryCompactor,
    TokenBudget,
)
from minicode_harness.context.token import estimate_tokens
from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.memory import MemorySnapshotStore, RepositoryMemoryStore
from minicode_harness.memory.repository_id import RepositoryIdentityUnavailable
from minicode_harness.state import ReplSessionMemory
from minicode_harness.models import create_model_client
from minicode_harness.output import OutputSink
from minicode_harness.policy import (
    ApprovalPolicy,
    DEFAULT_APPROVAL_POLICY,
    DEFAULT_PERMISSION_MODE,
    PermissionMode,
)
from minicode_harness.runtime.collaboration import (
    CollaborationMode,
    DEFAULT_COLLABORATION_MODE,
)
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.request_orchestrator import RequestOrchestrator
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.state import ApprovalClient, RunStore, UserInputClient
from minicode_harness.skills import SkillLoader
from minicode_harness.subagent import ReadonlySubagentRunner
from minicode_harness.tools import (
    SandboxMode,
    create_command_executor,
    inspect_git_diff,
)
from minicode_harness.trace import TraceWriter


class RunExecutionRequest(BaseModel):
    """Configuration required to execute one user task."""

    task: str
    workspace: Path = Path(".")
    provider: str = "qwen"
    model: str | None = None
    dry_run: bool = False
    write_enabled: bool = True
    approval_policy: ApprovalPolicy = DEFAULT_APPROVAL_POLICY
    permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE
    sandbox_mode: SandboxMode = SandboxMode.LOCAL
    sandbox_image: str | None = None
    collaboration_mode: CollaborationMode = DEFAULT_COLLABORATION_MODE
    skills: list[str] | None = None
    skills_enabled: bool = True
    repository_memory_enabled: bool = True
    subagents_enabled: bool = True
    worktree_workers_enabled: bool = True
    mcp_config: Path | None = None
    debug_trace: bool = False
    stream_model: bool = True


class RunExecutionResult(BaseModel):
    """Structured result returned to CLI and terminal callers."""

    run_id: str
    run_path: Path
    status: str
    conversation_session_id: str | None = None
    stop_reason: str | None = None
    final_text: str = ""
    stop_summary: str = ""
    steps: int = 0
    tool_calls: int = 0
    modified_files: list[str] = Field(default_factory=list)
    inspected_files: int = 0
    verification_status: str | None = None
    memory_review_status: str | None = None
    memory_reviewed_turns: int = 0
    memory_candidate_count: int = 0
    memory_auto_published_count: int = 0
    memory_pending_candidates: int = 0
    dry_run: bool = False


class ReviewExecutionResult(BaseModel):
    """Result of one read-only review subagent Run."""

    status: str
    summary: str
    run_id: str | None = None
    run_path: Path | None = None


class SessionCompactionResult(BaseModel):
    """Result of one explicit REPL session compaction."""

    changed: bool
    before_tokens: int
    after_tokens: int
    removed_groups: int = 0
    reason: str = ""
    focus: str = ""


def _emit_optional(output_sink: OutputSink, name: str, *args, **kwargs) -> None:
    handler = getattr(output_sink, name, None)
    if callable(handler):
        handler(*args, **kwargs)


class RunExecutor:
    """Create and execute one Run without owning terminal presentation."""

    def __init__(
        self,
        *,
        run_store: RunStore | None = None,
        session_memory: ReplSessionMemory | None = None,
    ) -> None:
        self.run_store = run_store or RunStore()
        self.session_memory = session_memory

    def compact_session(
        self,
        *,
        provider: str,
        model: str | None = None,
        focus: str = "",
    ) -> SessionCompactionResult:
        """Compact the active canonical session with the normal semantic policy."""

        if self.session_memory is None:
            raise ValueError("No active conversation session is available.")
        messages = self.session_memory.load_message_history()
        compaction_state = self.session_memory.load_compaction_state()
        before_tokens = estimate_tokens(
            json.dumps(
                messages,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
        if not messages:
            return SessionCompactionResult(
                changed=False,
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                reason="empty_session",
                focus=" ".join(focus.split()),
            )

        model_client = create_model_client(provider=provider, model=model)
        capabilities = model_client.capabilities
        budget = TokenBudget(
            context_budget=int(getattr(capabilities, "context_window", 32_000)),
            reserved_output=int(
                getattr(capabilities, "reserved_output_tokens", 6_000)
            ),
            soft_limit=0.80,
            semantic_limit=0.88,
            hard_limit=0.95,
        )
        preparer = ContextPreparer(
            budget,
            semantic_compactor=LLMSemanticHistoryCompactor(model_client),
        )
        updated_state, event = preparer.manual_compact(
            messages,
            compaction_state=compaction_state,
            focus=focus,
        )
        if event is None:
            return SessionCompactionResult(
                changed=False,
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                reason="no_compressible_history",
                focus=" ".join(focus.split()),
            )

        changed = (
            bool(event.details.get("changed"))
            and updated_state != compaction_state
        )
        if changed:
            self.session_memory.replace_compaction_state(updated_state)
        return SessionCompactionResult(
            changed=changed,
            before_tokens=event.before_tokens,
            after_tokens=event.after_tokens,
            removed_groups=int(event.details.get("removed_groups") or 0),
            reason=(
                "semantic"
                if changed
                else str(event.details.get("failure_reason") or event.reason)
            ),
            focus=" ".join(focus.split())[:500],
        )

    def review_current_diff(
        self,
        *,
        workspace: Path | str,
        provider: str,
        model: str | None = None,
        focus: str = "",
        cancellation_token: CancellationToken | None = None,
    ) -> ReviewExecutionResult:
        """Review the current Git diff with the built-in review Skill and read-only subagent."""

        resolved_workspace = Path(workspace).expanduser().resolve()
        diff = inspect_git_diff(resolved_workspace)
        if diff.returncode != 0:
            raise RuntimeError(diff.stderr.strip() or "git diff failed.")
        if not diff.diff.strip():
            return ReviewExecutionResult(
                status="no_changes",
                summary="No git diff to review.",
            )

        normalized_focus = " ".join(focus.split())
        if len(normalized_focus) > 500:
            raise ValueError("Review focus must not exceed 500 characters.")
        skill = SkillLoader().load("review")
        task = "\n\n".join(
            part
            for part in [
                skill.content.strip(),
                (
                    "Review the current Git diff. Inspect relevant surrounding code and tests "
                    "before judging a finding. Keep the review read-only."
                ),
                f"Additional focus: {normalized_focus}" if normalized_focus else "",
            ]
            if part
        )
        session = self.run_store.create_run(
            task="Review current Git diff",
            workspace=resolved_workspace,
            provider=provider,
            model=model,
            conversation_session_id=(
                self.session_memory.session_id
                if self.session_memory is not None
                else None
            ),
            no_write=True,
            approval_policy=ApprovalPolicy.NEVER.value,
            permission_mode=PermissionMode.READ_ONLY.value,
            collaboration_mode=CollaborationMode.PLAN.value,
            skills="review",
            no_skills=False,
            repository_memory_enabled=False,
            subagents_enabled=False,
        )
        run_path = self.run_store.path_for(session.run_id)
        trace_writer = TraceWriter(run_path / "trace.jsonl")
        trace_writer.write_event(
            "review_started",
            run_id=session.run_id,
            focus=normalized_focus,
        )
        model_client = create_model_client(provider=provider, model=model)
        runner = ReadonlySubagentRunner(
            workspace=resolved_workspace,
            model_client=model_client,
            trace_writer=trace_writer,
            artifact_dir=run_path / "artifacts" / "review",
            cancellation_token=cancellation_token,
        )
        try:
            result = runner.run(task)
        except Exception as exc:
            self.run_store.update_session_state(
                session.run_id,
                status="failed",
                current_step=0,
            )
            trace_writer.write_event(
                "review_finished",
                run_id=session.run_id,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self.run_store.update_session_state(
            session.run_id,
            status=result.status,
            current_step=0,
        )
        trace_writer.write_event(
            "review_finished",
            run_id=session.run_id,
            status=result.status,
        )
        return ReviewExecutionResult(
            status=result.status,
            summary=result.summary,
            run_id=session.run_id,
            run_path=run_path,
        )

    def execute(
        self,
        request: RunExecutionRequest,
        *,
        output_sink: OutputSink,
        approval_client: ApprovalClient,
        user_input_client: UserInputClient | None = None,
        cancellation_token: CancellationToken | None = None,
        steering_queue: SteeringQueue | None = None,
    ) -> RunExecutionResult:
        """Execute one run and return facts for the caller to render."""

        workspace = request.workspace.expanduser().resolve()
        skills_csv = ",".join(request.skills) if request.skills else None
        session = self.run_store.create_run(
            task=request.task,
            workspace=workspace,
            provider=request.provider,
            model=request.model,
            conversation_session_id=(
                self.session_memory.session_id
                if self.session_memory is not None
                else None
            ),
            no_write=not request.write_enabled,
            approval_policy=request.approval_policy.value,
            permission_mode=request.permission_mode.value,
            sandbox_mode=request.sandbox_mode.value,
            sandbox_image=request.sandbox_image,
            collaboration_mode=request.collaboration_mode.value,
            skills=skills_csv,
            no_skills=not request.skills_enabled,
            repository_memory_enabled=request.repository_memory_enabled,
            mcp_config=str(request.mcp_config) if request.mcp_config is not None else None,
            subagents_enabled=request.subagents_enabled,
        )
        run_path = self.run_store.path_for(session.run_id)
        _emit_optional(output_sink, "run_started", session.run_id)
        trace_writer = TraceWriter(run_path / "trace.jsonl")
        trace_writer.write_event(
            "run_started",
            run_id=session.run_id,
            task=session.task,
            workspace=session.workspace,
            conversation_session_id=session.conversation_session_id,
            provider=request.provider,
            model=request.model,
            dry_run=request.dry_run,
            no_write=not request.write_enabled,
            approval_policy=request.approval_policy.value,
            permission_mode=request.permission_mode.value,
            sandbox_mode=request.sandbox_mode.value,
            sandbox_image=request.sandbox_image,
            collaboration_mode=request.collaboration_mode.value,
            repository_memory_enabled=request.repository_memory_enabled,
            context_architecture="canonical_messages",
            mcp_config=str(request.mcp_config) if request.mcp_config is not None else None,
            subagents_enabled=request.subagents_enabled,
            debug_trace=request.debug_trace,
        )

        if request.dry_run:
            _emit_optional(
                output_sink,
                "run_finished",
                status="created",
                run_id=session.run_id,
                stop_reason="dry_run",
            )
            return RunExecutionResult(
                run_id=session.run_id,
                run_path=run_path,
                status="created",
                conversation_session_id=session.conversation_session_id,
                stop_reason="dry_run",
                dry_run=True,
            )

        model_client = create_model_client(
            provider=request.provider,
            model=request.model,
        )
        command_executor = create_command_executor(
            request.sandbox_mode,
            image=request.sandbox_image,
        )
        repository_memory = None
        request_orchestrator = None
        memory_source = None
        memory_snapshot = None
        if request.repository_memory_enabled:
            try:
                repository_memory = RepositoryMemoryStore(workspace)
                request_orchestrator = RequestOrchestrator(
                    repository_memory=repository_memory,
                    trace_writer=trace_writer,
                    review_model_client=model_client,
                )
                memory_source = request_orchestrator.recall_snapshot()
                memory_snapshot = MemorySnapshotStore(run_path).save(
                    repository_id=repository_memory.repository_id,
                    rendered_index=memory_source.rendered_index,
                    topic_payloads=memory_source.topic_payloads,
                )
            except RepositoryIdentityUnavailable as exc:
                repository_memory = None
                request_orchestrator = None
                memory_source = None
                memory_snapshot = None
                trace_writer.write_event(
                    "memory_initialization_skipped",
                    reason="repository_identity_unavailable",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        long_term_context = (
            memory_source.rendered_index if memory_source is not None else ""
        )
        if memory_snapshot is not None:
            trace_writer.write_event(
                "memory_snapshot_created",
                repository_id=memory_snapshot.repository_id,
                index_hash=memory_snapshot.index_hash,
                path=MemorySnapshotStore(run_path).checkpoint_path,
            )

        loop = AgentLoop(
            task=request.task,
            workspace=workspace,
            model_client=model_client,
            trace_writer=trace_writer,
            config=AgentLoopConfig(
                repository_memory_enabled=request.repository_memory_enabled,
                enable_subagents=request.subagents_enabled,
                enable_worktree_workers=(
                    request.write_enabled and request.worktree_workers_enabled
                ),
            ),
            skill_names=request.skills,
            no_skills=not request.skills_enabled,
            data_dir=(repository_memory.data_dir if repository_memory is not None else None),
            long_term_context=long_term_context,
            repository_memory=repository_memory,
            memory_snapshot_hash=(memory_snapshot.index_hash if memory_snapshot else None),
            memory_snapshot_path=(
                MemorySnapshotStore(run_path).checkpoint_path
                if memory_snapshot is not None
                else None
            ),
            enable_write=request.write_enabled,
            approval_policy=request.approval_policy,
            permission_mode=request.permission_mode,
            collaboration_mode=request.collaboration_mode,
            approval_client=approval_client,
            user_input_client=user_input_client,
            provider=request.provider,
            model=request.model or getattr(model_client, "model", None),
            session_memory=self.session_memory,
            output_sink=output_sink,
            stream_model=request.stream_model,
            cancellation_token=cancellation_token,
            mcp_config=request.mcp_config,
            command_executor=command_executor,
            subagent_model_client_factory=lambda: create_model_client(
                provider=request.provider,
                model=request.model,
            ),
            steering_queue=steering_queue,
        )
        agent_result = loop.run()
        self.run_store.update_session_state(
            session.run_id,
            status=agent_result.status,
            current_step=agent_result.steps,
        )
        trace_writer.write_event(
            "run_finished",
            run_id=session.run_id,
            status=agent_result.status,
            stop_reason=agent_result.stop_reason,
            steps=agent_result.steps,
            tool_calls=agent_result.tool_calls,
        )

        final_text = agent_result.final_text or ""
        if self.session_memory is not None:
            self.session_memory.add_user_turn(request.task, run_id=session.run_id)
            self.session_memory.add_assistant_turn(
                final_text
                or (
                    f"Run finished with status {agent_result.status}: "
                    f"{agent_result.stop_reason}"
                ),
                run_id=session.run_id,
                history_length=len(self.session_memory.load_message_history()),
            )

        memory_finalization = None
        if (
            agent_result.status == "completed"
            and final_text
            and request_orchestrator is not None
        ):
            memory_finalization = request_orchestrator.finalize_completed_run(
                run_id=session.run_id,
                user_input=request.task,
                assistant_text=final_text,
                observations=list(getattr(loop, "observations", [])),
                modified_files=list(loop.modified_files),
                verification=loop.run_state.verification,
            )

        _emit_optional(
            output_sink,
            "run_finished",
            status=agent_result.status,
            run_id=session.run_id,
            stop_reason=agent_result.stop_reason,
        )
        return RunExecutionResult(
            run_id=session.run_id,
            run_path=run_path,
            status=agent_result.status,
            conversation_session_id=session.conversation_session_id,
            stop_reason=agent_result.stop_reason,
            final_text=final_text,
            stop_summary=agent_result.stop_summary or "",
            steps=agent_result.steps,
            tool_calls=agent_result.tool_calls,
            modified_files=list(loop.modified_files),
            inspected_files=len(
                getattr(getattr(loop, "run_state", None), "inspected_files", [])
            ),
            verification_status=loop.run_state.verification.status,
            memory_review_status=(
                getattr(memory_finalization, "review_status", None)
                if memory_finalization is not None
                else None
            ),
            memory_reviewed_turns=(
                int(getattr(memory_finalization, "reviewed_turns", 0))
                if memory_finalization is not None
                else 0
            ),
            memory_candidate_count=(
                int(getattr(memory_finalization, "candidate_count", 0))
                if memory_finalization is not None
                else 0
            ),
            memory_auto_published_count=(
                int(getattr(memory_finalization, "auto_published_count", 0))
                if memory_finalization is not None
                else 0
            ),
            memory_pending_candidates=(
                int(getattr(memory_finalization, "pending_candidates", 0))
                if memory_finalization is not None
                else 0
            ),
        )
