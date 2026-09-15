"""Lightweight dependency composition for one AgentLoop instance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from minicode_harness.context import (
    ContextBuilder,
    ContextObservation,
    ContextPreparer,
    ContextSkill,
    LLMSemanticHistoryCompactor,
    RepositoryRuleLoader,
    RunState,
    TokenBudget,
)
from minicode_harness.hooks import HookManager, default_hook_manager
from minicode_harness.mcp import MCPManager
from minicode_harness.memory import RepositoryMemoryStore
from minicode_harness.models import ModelClient
from minicode_harness.output import NullOutputSink, OutputSink
from minicode_harness.policy import ApprovalPolicy, PermissionMode
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.progress_policy import (
    BoundedProgressPolicy,
    DisabledProgressPolicy,
)
from minicode_harness.runtime.recovery import ModelRecoveryPolicy
from minicode_harness.runtime.run_lifecycle import RunLifecycle
from minicode_harness.runtime.runtime_tasks import (
    BackgroundCommandManager,
    RuntimeTaskRegistry,
)
from minicode_harness.runtime.tool_runtime import ModelOutputBudget, ToolRuntime
from minicode_harness.skills import SkillLoader
from minicode_harness.state import (
    ApprovalClient,
    ApprovalStore,
    CheckpointStore,
    ExecutionJournal,
    InteractiveApprovalClient,
    ReplSessionMemory,
    TaskListState,
    TaskStore,
)
from minicode_harness.storage import HarnessDataStore, default_data_dir
from minicode_harness.tools import CommandExecutor, LocalCommandExecutor, ToolRegistry
from minicode_harness.trace import TraceWriter
from minicode_harness.worktree_worker import WorktreeWorkerManager


@dataclass
class AgentComponents:
    """Constructed runtime dependencies; intentionally contains no control methods."""

    repository_rule_loader: RepositoryRuleLoader | None
    output_sink: OutputSink
    cancellation_token: CancellationToken
    hook_manager: HookManager
    recovery_policy: ModelRecoveryPolicy
    model_capabilities: Any
    output_budget: ModelOutputBudget
    mcp_manager: MCPManager | None
    artifact_dir: Path
    runtime_task_registry: RuntimeTaskRegistry
    background_command_manager: BackgroundCommandManager
    worktree_worker_manager: WorktreeWorkerManager
    task_store: TaskStore
    skill_loader: SkillLoader
    available_skills: list[ContextSkill]
    tools: ToolRegistry
    data_dir: Path
    context_builder: ContextBuilder
    context_preparer: ContextPreparer
    checkpoint_store: CheckpointStore
    approval_store: ApprovalStore
    approval_client: ApprovalClient
    tool_runtime: ToolRuntime
    progress_policy: BoundedProgressPolicy | DisabledProgressPolicy
    lifecycle: RunLifecycle


def build_agent_components(
    *,
    task: str,
    workspace: str,
    model_client: ModelClient,
    trace_writer: TraceWriter,
    run_id: str,
    provider: str,
    model: str | None,
    enable_repository_rules: bool,
    repository_rule_loader: RepositoryRuleLoader | None,
    skill_names: list[str] | None,
    no_skills: bool,
    skill_loader: SkillLoader | None,
    memory_store: HarnessDataStore | None,
    data_dir: Path | str | None,
    repository_memory: RepositoryMemoryStore | None,
    repository_memory_enabled: bool,
    context_builder: ContextBuilder | None,
    context_preparer: ContextPreparer | None,
    enable_write: bool,
    enable_command: bool,
    approval_policy: ApprovalPolicy,
    permission_mode: PermissionMode,
    approval_client: ApprovalClient | None,
    approval_store: ApprovalStore | None,
    request_user_input_handler: Callable[[str, list[dict[str, str | None]]], Any] | None,
    checkpoint_store: CheckpointStore | None,
    hook_manager: HookManager | None,
    hook_owner: Any,
    session_memory: ReplSessionMemory | None,
    output_sink: OutputSink | None,
    cancellation_token: CancellationToken | None,
    recovery_policy: ModelRecoveryPolicy | None,
    mcp_manager: MCPManager | None,
    mcp_config: Path | str | None,
    command_executor: CommandExecutor | None,
    subagent_handler: Callable[[str], Any] | None,
    memory_topic_reader: Callable[[str], dict[str, str]] | None,
    initial_task_state: TaskListState | None,
    initial_observations: list[ContextObservation],
    run_state: RunState,
    workspace_generation: int,
    max_memory_topic_reads: int,
    rollback_on_unfinished_stop: bool,
    enable_subagents: bool,
    enable_worktree_workers: bool,
    enable_progress_guidance: bool,
    max_steps: int,
    start_step: int,
    emit_lifecycle_hook: Callable[[str, int | None, dict[str, Any]], None],
) -> AgentComponents:
    """Build dependencies without taking ownership of AgentLoop control flow."""

    resolved_rule_loader = (
        repository_rule_loader
        if repository_rule_loader is not None
        else (RepositoryRuleLoader(workspace) if enable_repository_rules else None)
    )
    resolved_output_sink = output_sink or NullOutputSink()
    resolved_cancellation = cancellation_token or CancellationToken()
    resolved_hook_manager = hook_manager or default_hook_manager(trace_writer=trace_writer)
    resolved_recovery = recovery_policy or ModelRecoveryPolicy()

    model_capabilities = getattr(model_client, "capabilities", None)
    model_max_output_tokens = int(
        getattr(model_capabilities, "max_output_tokens", 4096)
    )
    output_budget = ModelOutputBudget(
        current=min(4096, model_max_output_tokens),
        maximum=model_max_output_tokens,
    )

    configured_mcp = mcp_config
    if configured_mcp is None:
        import os

        configured_mcp = os.environ.get("MINICODE_MCP_CONFIG")
    resolved_mcp_manager = mcp_manager or (
        MCPManager.from_config_file(configured_mcp) if configured_mcp else None
    )

    artifact_dir = trace_writer.trace_path.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runtime_task_registry = RuntimeTaskRegistry(max_active=4)
    effective_command_executor = command_executor or LocalCommandExecutor()
    background_command_manager = BackgroundCommandManager(
        workspace=workspace,
        artifact_dir=artifact_dir,
        command_executor=effective_command_executor,
        registry=runtime_task_registry,
    )
    worktree_worker_manager = WorktreeWorkerManager(
        parent_workspace=workspace,
        run_id=run_id,
        model_client=model_client,
        registry=runtime_task_registry,
        artifact_dir=artifact_dir,
        command_executor=effective_command_executor,
        recovery_policy=resolved_recovery,
    )
    task_store = TaskStore(initial_task_state)
    resolved_skill_loader = skill_loader or SkillLoader()
    skill_summaries = (
        [] if no_skills else resolved_skill_loader.list_summaries(skill_names)
    )
    available_skills = [
        ContextSkill(
            name=summary.name,
            description=summary.description,
            source=summary.path,
        )
        for summary in skill_summaries
    ]

    try:
        tools = ToolRegistry(
            workspace,
            enable_write=enable_write,
            enable_command=enable_command,
            artifact_dir=str(artifact_dir),
            skill_loader=resolved_skill_loader if available_skills else None,
            skill_names=[skill.name for skill in available_skills],
            mcp_manager=resolved_mcp_manager,
            subagent_handler=subagent_handler if enable_subagents else None,
            memory_topic_reader=(
                memory_topic_reader
                if repository_memory is not None and repository_memory_enabled
                else None
            ),
            task_create_handler=task_store.create,
            task_update_handler=task_store.update,
            task_list_handler=task_store.list_tasks,
            request_user_input_handler=request_user_input_handler,
            cancellation_token=resolved_cancellation,
            command_executor=effective_command_executor,
            runtime_task_registry=runtime_task_registry,
            background_command_manager=background_command_manager,
            worktree_worker_handler=(
                worktree_worker_manager.start
                if enable_write and enable_worktree_workers
                else None
            ),
        )
    except Exception:
        if resolved_mcp_manager is not None:
            resolved_mcp_manager.close()
        raise

    resolved_data_dir = Path(
        data_dir
        or (
            repository_memory.data_dir
            if repository_memory is not None
            else (memory_store.data_dir if memory_store is not None else default_data_dir())
        )
    )
    capabilities_budget = TokenBudget(
        context_budget=int(getattr(model_capabilities, "context_window", 32_000)),
        reserved_output=int(
            getattr(model_capabilities, "reserved_output_tokens", 6_000)
        ),
        soft_limit=0.80,
        semantic_limit=0.88,
        hard_limit=0.95,
    )
    resolved_context_builder = context_builder or ContextBuilder(
        budget=capabilities_budget,
    )
    resolved_context_preparer = context_preparer or ContextPreparer(
        resolved_context_builder.budget,
        semantic_compactor=LLMSemanticHistoryCompactor(
            model_client,
            trace_writer=trace_writer,
        ),
    )
    resolved_checkpoint_store = checkpoint_store or CheckpointStore(
        trace_writer.trace_path.parent / "checkpoints"
    )
    resolved_approval_store = approval_store or ApprovalStore(
        trace_writer.trace_path.parent / "approvals"
    )
    resolved_approval_client = approval_client or InteractiveApprovalClient()
    execution_journal = ExecutionJournal(
        trace_writer.trace_path.parent / "execution-journal.jsonl"
    )

    tool_runtime = ToolRuntime(
        workspace=workspace,
        task=task,
        run_id=run_id,
        tools=tools,
        trace_writer=trace_writer,
        hook_manager=resolved_hook_manager,
        hook_owner=hook_owner,
        approval_client=resolved_approval_client,
        approval_store=resolved_approval_store,
        approval_policy=approval_policy,
        permission_mode=permission_mode,
        execution_journal=execution_journal,
        session_memory=session_memory,
        artifact_dir=artifact_dir,
        output_sink=resolved_output_sink,
        initial_observations=initial_observations,
        run_state=run_state,
        initial_workspace_generation=workspace_generation,
        max_memory_topic_reads=max_memory_topic_reads,
        rollback_on_unfinished_stop=rollback_on_unfinished_stop,
        output_budget=output_budget,
    )
    progress_policy = (
        BoundedProgressPolicy(
            max_steps=max_steps,
            start_step=start_step,
            observations=initial_observations,
        )
        if enable_progress_guidance and enable_write
        else DisabledProgressPolicy()
    )

    resource_closers = [
        ("worktree_workers", worktree_worker_manager.shutdown),
        ("background_commands", background_command_manager.shutdown),
        ("runtime_tasks", runtime_task_registry.shutdown),
    ]
    if resolved_mcp_manager is not None:
        resource_closers.append(("mcp", resolved_mcp_manager.close))
    lifecycle = RunLifecycle(
        run_id=run_id,
        task=task,
        workspace=workspace,
        trace_writer=trace_writer,
        checkpoint_store=resolved_checkpoint_store,
        session_memory=session_memory,
        emit_hook=emit_lifecycle_hook,
        resource_closers=resource_closers,
    )

    return AgentComponents(
        repository_rule_loader=resolved_rule_loader,
        output_sink=resolved_output_sink,
        cancellation_token=resolved_cancellation,
        hook_manager=resolved_hook_manager,
        recovery_policy=resolved_recovery,
        model_capabilities=model_capabilities,
        output_budget=output_budget,
        mcp_manager=resolved_mcp_manager,
        artifact_dir=artifact_dir,
        runtime_task_registry=runtime_task_registry,
        background_command_manager=background_command_manager,
        worktree_worker_manager=worktree_worker_manager,
        task_store=task_store,
        skill_loader=resolved_skill_loader,
        available_skills=available_skills,
        tools=tools,
        data_dir=resolved_data_dir,
        context_builder=resolved_context_builder,
        context_preparer=resolved_context_preparer,
        checkpoint_store=resolved_checkpoint_store,
        approval_store=resolved_approval_store,
        approval_client=resolved_approval_client,
        tool_runtime=tool_runtime,
        progress_policy=progress_policy,
        lifecycle=lifecycle,
    )
