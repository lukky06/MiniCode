"""Command line entry points for MiniCodeHarness."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .config import UserConfig, load_user_config as _load_user_config
from .benchmark import (
    BenchmarkRunner,
    BenchmarkRunnerConfig,
    BenchmarkScenarioRunner,
    BenchmarkScenarioRunnerConfig,
)
from .history import (
    format_conversation_sessions,
    format_run_detail,
    format_run_history,
)
from .memory import (
    MemoryManualReviewError,
    RepositoryMemoryCandidateService,
    RepositoryMemoryPublisher,
    RepositoryMemoryStore,
    TOPIC_NAMES,
)
from .models import ModelClientConfigurationError, create_model_client
from .output import TextOutputSink
from .policy import (
    ApprovalPolicy,
    DEFAULT_APPROVAL_POLICY,
    DEFAULT_PERMISSION_MODE,
    PermissionMode,
)
from .report import format_run_trace, generate_run_report
from .resume import latest_recoverable_run_id, resume_run
from .runtime import CollaborationMode, DEFAULT_COLLABORATION_MODE
from .runtime.request_orchestrator import RequestOrchestrator
from .runtime.run_executor import RunExecutionRequest, RunExecutor
from .state import (
    NonInteractiveApprovalClient,
    RunStore,
    default_run_root,
)
from .swebench import (
    SweBenchBudget,
    SweBenchDockerCommandExecutor,
    SweBenchEvaluator,
    SweBenchRunner,
    SweBenchRunnerConfig,
)
from .terminal import (
    SessionLaunchMode,
    TerminalApprovalClient,
    TerminalOutputSink,
)
from .tui_bridge.launcher import launch_tui
from .terminal.rendering import TerminalRenderer
from .tools import SandboxMode
from .worktrees import GitWorktreeManager


app = typer.Typer(
    name="minicode",
    help="Local general-purpose Coding Agent Harness for software repositories.",
    no_args_is_help=True,
    epilog=(
        'Examples: minicode "fix the failing tests" | '
        "minicode --no-write \"explain this repo\" | "
        "minicode resume | minicode runs | minicode memory"
    ),
)
bench_app = typer.Typer(help="Benchmark commands.", no_args_is_help=True)
worktree_app = typer.Typer(help="Managed Git worktree commands.", no_args_is_help=True)

KNOWN_COMMANDS = {
    "exec", "resume", "recover", "trace", "report", "runs", "sessions",
    "memory", "bench", "worktree"
}
GLOBAL_OPTIONS = {
    "--help",
    "-h",
    "--version",
    "--install-completion",
    "--show-completion",
}
INTERACTIVE_OPTIONS_WITH_VALUES = {
    "--workspace",
    "-w",
    "--provider",
    "--model",
    "--skills",
    "--mcp-config",
    "--worktree",
    "--approval-policy",
    "--permission-mode",
    "--sandbox",
    "--sandbox-image",
    "--mode",
}
INTERACTIVE_FLAG_OPTIONS = {
    "--no-write",
    "--no-skills",
    "--no-repository-memory",
    "--no-subagents",
    "--no-color",
}
EXEC_ONLY_OPTIONS = {"--dry-run", "--debug-trace", "--plain"}


@dataclass(frozen=True)
class InteractiveLaunch:
    task: str | None = None
    workspace: Path = Path(".")
    worktree: str | None = None
    provider: str | None = None
    model: str | None = None
    no_write: bool = False
    approval_policy: ApprovalPolicy | None = None
    permission_mode: PermissionMode | None = None
    sandbox_mode: SandboxMode | None = None
    sandbox_image: str | None = None
    collaboration_mode: CollaborationMode = DEFAULT_COLLABORATION_MODE
    skills: str | None = None
    no_skills: bool = False
    mcp_config: Path | None = None
    no_subagents: bool = False
    no_repository_memory: bool = False
    no_color: bool = False
    session_mode: SessionLaunchMode = SessionLaunchMode.NEW
    session_id: str | None = None


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"minicode {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        help="Show the MiniCodeHarness version and exit.",
        is_eager=True,
    ),
) -> None:
    """MiniCodeHarness command group."""
    _ = version


@app.command("exec")
def exec_command(
    task: str = typer.Argument(..., help="Coding task to execute."),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Target workspace path.",
    ),
    worktree: Optional[str] = typer.Option(
        None,
        "--worktree",
        help="Open or create an isolated managed Worktree Session.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Model provider name. Defaults to MINICODE_PROVIDER or qwen.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name. Defaults to MINICODE_MODEL when set.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Create no external side effects."),
    no_write: bool = typer.Option(False, "--no-write", help="Disable write tools."),
    approval_policy: Optional[ApprovalPolicy] = typer.Option(
        None,
        "--approval-policy",
        help="Approval prompting policy: on-request or never.",
    ),
    permission_mode: Optional[PermissionMode] = typer.Option(
        None,
        "--permission-mode",
        help="Automatic side-effect scope: read-only, workspace-write, or full-access.",
    ),
    sandbox: Optional[SandboxMode] = typer.Option(
        None,
        "--sandbox",
        help="Command execution environment: local or docker.",
    ),
    sandbox_image: Optional[str] = typer.Option(
        None,
        "--sandbox-image",
        help="Docker image used when --sandbox docker is selected.",
    ),
    mode: CollaborationMode = typer.Option(
        DEFAULT_COLLABORATION_MODE,
        "--mode",
        help="Collaboration mode: default or plan.",
    ),
    skills: Optional[str] = typer.Option(
        None,
        "--skills",
        help="Comma-separated skill names exposed to the model catalog.",
    ),
    no_skills: bool = typer.Option(
        False,
        "--no-skills",
        help="Disable the skill catalog and read(source=skill) capability.",
    ),
    debug_trace: bool = typer.Option(False, "--debug-trace", help="Persist debug trace artifacts."),
    no_repository_memory: bool = typer.Option(
        False,
        "--no-repository-memory",
        help="Disable Repository Memory index, topic reads, snapshots, and review capture.",
    ),
    mcp_config: Optional[Path] = typer.Option(
        None,
        "--mcp-config",
        help="Explicit MCP JSON configuration file.",
    ),
    no_subagents: bool = typer.Option(
        False,
        "--no-subagents",
        help="Disable the bounded read-only delegate_task tool.",
    ),
    plain: bool = typer.Option(False, "--plain", help="Force plain non-ANSI output."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable terminal colors."),
) -> None:
    """Execute one standalone non-interactive Run."""
    try:
        user_config = _load_user_config()
        resolved_provider = _resolve_provider(provider, config=user_config)
        resolved_model = _resolve_model(model, config=user_config)
        resolved_approval_policy = _resolve_approval_policy(
            approval_policy,
            config=user_config,
        )
        resolved_permission_mode = _resolve_permission_mode(
            permission_mode,
            config=user_config,
        )
        resolved_sandbox = _resolve_sandbox_mode(sandbox, config=user_config)
        resolved_sandbox_image = _resolve_sandbox_image(
            resolved_sandbox,
            sandbox_image,
        )
    except ValueError as exc:
        typer.echo(f"Runtime configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    resolved_mcp_config = _resolve_mcp_config(mcp_config)
    resolved_workspace = _resolve_worktree_workspace(workspace, worktree)
    executor = RunExecutor(run_store=RunStore())
    request = RunExecutionRequest(
        task=task,
        workspace=resolved_workspace,
        provider=resolved_provider,
        model=resolved_model,
        dry_run=dry_run,
        write_enabled=not no_write,
        approval_policy=resolved_approval_policy,
        permission_mode=resolved_permission_mode,
        sandbox_mode=resolved_sandbox,
        sandbox_image=resolved_sandbox_image,
        collaboration_mode=mode,
        skills=_parse_skill_names(skills),
        skills_enabled=not no_skills,
        repository_memory_enabled=not no_repository_memory,
        subagents_enabled=not no_subagents,
        mcp_config=resolved_mcp_config,
        debug_trace=debug_trace,
        stream_model=True,
    )
    sink, approval_client, terminal_renderer = _execution_io(
        plain=plain,
        no_color=no_color,
    )
    try:
        result = executor.execute(
            request,
            output_sink=sink,
            approval_client=approval_client,
        )
    except ModelClientConfigurationError as exc:
        typer.echo(f"Model client configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(f"Runtime configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    finalize_answer = getattr(sink, "finalize_answer", None)
    if callable(finalize_answer):
        finalize_answer()
    flush = getattr(sink, "flush", None)
    if callable(flush):
        flush()
    if terminal_renderer is not None and not result.dry_run:
        if result.final_text:
            terminal_renderer.ensure_trailing_newline()
        terminal_renderer.show_run_summary(result)
    if result.dry_run:
        typer.echo(f"Created run {result.run_id} at {result.run_path}")
        typer.echo("Dry run: agent execution skipped.")
    elif result.status != "completed":
        if terminal_renderer is None and result.stop_summary:
            typer.echo(result.stop_summary)
        raise typer.Exit(code=1)


@app.command()
def resume(
    session_id: Optional[str] = typer.Argument(
        None,
        help="Conversation Session ID. Defaults to the latest Workspace Session.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace whose Session should be resumed.",
    ),
    worktree: Optional[str] = typer.Option(
        None,
        "--worktree",
        help="Open or create an isolated Git Worktree Session.",
    ),
    provider: Optional[str] = typer.Option(None, "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    no_write: bool = typer.Option(False, "--no-write"),
    approval_policy: Optional[ApprovalPolicy] = typer.Option(
        None,
        "--approval-policy",
    ),
    permission_mode: Optional[PermissionMode] = typer.Option(
        None,
        "--permission-mode",
    ),
    sandbox: Optional[SandboxMode] = typer.Option(
        None,
        "--sandbox",
    ),
    sandbox_image: Optional[str] = typer.Option(None, "--sandbox-image"),
    mode: CollaborationMode = typer.Option(
        DEFAULT_COLLABORATION_MODE,
        "--mode",
    ),
    skills: Optional[str] = typer.Option(None, "--skills"),
    no_skills: bool = typer.Option(False, "--no-skills"),
    mcp_config: Optional[Path] = typer.Option(None, "--mcp-config"),
    no_subagents: bool = typer.Option(False, "--no-subagents"),
    no_repository_memory: bool = typer.Option(
        False,
        "--no-repository-memory",
    ),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Resume a conversation Session in the interactive terminal."""

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo("Interactive sessions require a TTY.", err=True)
        raise typer.Exit(code=2)
    launch = InteractiveLaunch(
        workspace=workspace,
        worktree=worktree,
        provider=provider,
        model=model,
        no_write=no_write,
        approval_policy=approval_policy,
        permission_mode=permission_mode,
        sandbox_mode=sandbox,
        sandbox_image=sandbox_image,
        collaboration_mode=mode,
        skills=skills,
        no_skills=no_skills,
        mcp_config=mcp_config,
        no_subagents=no_subagents,
        no_repository_memory=no_repository_memory,
        no_color=no_color,
        session_mode=(
            SessionLaunchMode.EXACT
            if session_id is not None
            else SessionLaunchMode.CONTINUE
        ),
        session_id=session_id,
    )
    try:
        _run_terminal(launch)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Session launch failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def recover(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID to recover. Defaults to the latest recoverable Workspace Run.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace used when no Run ID is supplied.",
    ),
    force_rebuild_context: bool = typer.Option(
        False,
        "--force-rebuild-context",
        help="Rebuild context from the current workspace before recovery.",
    ),
    plain: bool = typer.Option(False, "--plain", help="Force plain non-ANSI output."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable terminal colors."),
) -> None:
    """Recover an interrupted Run from its Checkpoint."""
    stream_model = True
    sink, approval_client, terminal_renderer = _execution_io(
        plain=plain,
        no_color=no_color,
    )
    try:
        resolved_run_id = (
            run_id
            if run_id is not None
            else latest_recoverable_run_id(workspace=workspace)
        )
        result = resume_run(
            resolved_run_id,
            force_rebuild_context=force_rebuild_context,
            approval_client=approval_client,
            output_sink=sink,
            stream_model=stream_model,
        )
    except ModelClientConfigurationError as exc:
        typer.echo(f"Model client configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(f"Recovery failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if result.status == "blocked":
        typer.echo(f"Recovery blocked: {result.reason}", err=True)
        if result.conflicts:
            typer.echo("Conflicting files:", err=True)
            for conflict in result.conflicts:
                typer.echo(f"- {conflict}", err=True)
        raise typer.Exit(code=1)

    finalize_answer = getattr(sink, "finalize_answer", None)
    if callable(finalize_answer):
        finalize_answer()
    flush = getattr(sink, "flush", None)
    if callable(flush):
        flush()
    if terminal_renderer is not None and result.final_text:
        terminal_renderer.ensure_trailing_newline()
    typer.echo(f"Recovery finished with status {result.status}: {result.reason}")
    if result.final_text and not stream_model:
        typer.echo(result.final_text)
    if result.stop_summary:
        typer.echo(result.stop_summary)
    if result.status not in {"completed", "continued"}:
        raise typer.Exit(code=1)


@app.command()
def trace(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID whose trace should be displayed. Defaults to the latest run.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace used when no Run ID is supplied.",
    ),
) -> None:
    """Display a run trace."""
    resolved_run_id = _resolve_run_id(run_id, workspace=workspace)
    try:
        typer.echo(format_run_trace(resolved_run_id))
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Trace failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def report(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID whose report should be generated. Defaults to the latest run.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace used when no Run ID is supplied.",
    ),
) -> None:
    """Generate or display a run report."""
    resolved_run_id = _resolve_run_id(run_id, workspace=workspace)
    try:
        result = generate_run_report(resolved_run_id)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Report failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Report generated:")
    typer.echo(result.report_path)
    typer.echo("Final diff generated:")
    typer.echo(result.final_diff_path)


@app.command()
def sessions(
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace whose conversation Sessions should be listed.",
    ),
    limit: int = typer.Option(10, "--limit", "-n", min=1),
) -> None:
    """List conversation Sessions for one Workspace."""

    typer.echo(format_conversation_sessions(workspace, limit=limit))


@app.command()
def runs(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID to inspect. Omit it to list recent Workspace Runs.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace used when no Run ID is supplied.",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        min=1,
        help="Maximum Runs to list when no Run ID is provided.",
    ),
    observations: int = typer.Option(
        5,
        "--observations",
        min=0,
        help="Recent observations to show for one run.",
    ),
) -> None:
    """Display Run history or one Run."""

    _show_runs(
        run_id=run_id,
        workspace=workspace,
        limit=limit,
        observations=observations,
    )


def _show_runs(
    *,
    run_id: str | None,
    workspace: Path,
    limit: int,
    observations: int,
) -> None:
    try:
        if run_id:
            typer.echo(format_run_detail(run_id, observation_limit=observations))
        else:
            typer.echo(format_run_history(limit=limit, workspace=workspace))
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Runs failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def memory(
    action: Optional[str] = typer.Argument(
        None,
        help=(
            "Repository Memory action: list, remember, review, candidates, approve, reject, "
            "show, or forget."
        ),
    ),
    value: Optional[str] = typer.Argument(
        None,
        help=(
            "Text for remember, candidate ID for approve/reject, or Topic/entry ID "
            "for show and forget."
        ),
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace whose repository memory should be managed.",
    ),
    topic: str = typer.Option(
        "instructions",
        "--topic",
        help="Repository Memory Topic used by remember.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Provider used by memory review.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model used by memory review.",
    ),
) -> None:
    """Manage repository Memory."""

    resolved_workspace = workspace.resolve()
    normalized_action = (action or "list").strip().lower()
    repository = RepositoryMemoryStore(resolved_workspace)
    try:
        if normalized_action == "list":
            typer.echo("Repository Memory")
            typer.echo(
                repository.index_store.ensure().content
                or "No durable memory Topics are registered."
            )
            typer.echo(
                "Pending reviews: "
                f"{len(repository.workflow_store.pending_reviews())}; "
                "pending candidates: "
                f"{len(repository.workflow_store.list_candidates(status='pending'))}."
            )
            for candidate_topic in repository.topic_store.registered_topics():
                warning = repository.topic_store.capacity_status(candidate_topic).warning
                if warning:
                    typer.echo(f"Warning: {warning}")
        elif normalized_action == "remember":
            if not value or not value.strip():
                raise ValueError("memory remember requires text.")
            if topic not in TOPIC_NAMES:
                raise ValueError(f"Unknown memory Topic: {topic}")
            result = RepositoryMemoryPublisher(repository).publish(
                topic=topic,
                text=value,
                evidence_ids=["cli"],
            )
            typer.echo(
                f"Memory {result.status}: topic={result.topic}; "
                f"entry={result.entry_id}; index={result.index_status}."
            )
            if result.capacity_warning:
                typer.echo(f"Warning: {result.capacity_warning}")
        elif normalized_action == "review":
            client = create_model_client(
                provider=_resolve_provider(provider),
                model=_resolve_model(model),
            )
            status, reviewed, created, auto_published = RequestOrchestrator(
                repository_memory=repository,
                review_model_client=client,
            ).review_pending(force=True)
            typer.echo(
                f"Memory review {status or 'no_reviews'}; reviewed={reviewed}; "
                f"created={created}; auto_published={auto_published}; "
                f"pending_reviews={len(repository.workflow_store.pending_reviews())}; "
                "pending_candidates="
                f"{len(repository.workflow_store.list_candidates(status='pending'))}."
            )
        elif normalized_action == "candidates":
            candidates = repository.workflow_store.list_candidates()
            if not candidates:
                typer.echo("No memory candidates.")
            for candidate in candidates:
                sources = ",".join(str(seq) for seq in candidate.source_review_seqs)
                typer.echo(
                    f"[{candidate.status}] {candidate.candidate_id} "
                    f"{candidate.topic}: {candidate.text}"
                )
                typer.echo(f"  reason: {candidate.reason}")
                typer.echo(f"  auto publish proposed: {candidate.auto_publish}")
                typer.echo(f"  source reviews: {sources}")
                for quote in candidate.source_quotes:
                    typer.echo(f"  source quote: {quote}")
                if candidate.reviewed_entry_ids:
                    typer.echo(
                        "  reviewed entries: "
                        + ", ".join(candidate.reviewed_entry_ids)
                    )
                if candidate.conflicts_with_entry_ids:
                    typer.echo(
                        "  conflicts with: "
                        + ", ".join(candidate.conflicts_with_entry_ids)
                    )
                for evidence in candidate.evidence:
                    verification = evidence.verification
                    verification_status = (
                        verification.status if verification is not None else "unknown"
                    )
                    typer.echo(
                        f"  evidence: review={evidence.review_seq}; "
                        f"run={evidence.source_run_id}; "
                        f"verification={verification_status}"
                    )
                    if verification is not None and verification.command:
                        typer.echo(
                            f"    verification command: {verification.command}; "
                            f"returncode={verification.returncode}"
                        )
                    failure = evidence.resolved_failure
                    if failure is not None:
                        typer.echo(
                            f"    failure: {failure.failed_command} -> "
                            f"{failure.passed_command}"
                        )
                        typer.echo(
                            "    modified files: "
                            + (", ".join(failure.modified_files) or "none")
                        )
        elif normalized_action == "approve":
            if not value:
                raise ValueError("memory approve requires a candidate ID.")
            result = RepositoryMemoryCandidateService(repository).approve(value)
            typer.echo(
                f"Memory candidate {result.status}: {result.candidate_id}; "
                f"status={result.candidate_status}; entry={result.entry_id}."
            )
            if result.capacity_warning:
                typer.echo(f"Warning: {result.capacity_warning}")
        elif normalized_action == "reject":
            if not value:
                raise ValueError("memory reject requires a candidate ID.")
            result = RepositoryMemoryCandidateService(repository).reject(value)
            typer.echo(
                f"Memory candidate {result.status}: {result.candidate_id}; "
                f"status={result.candidate_status}."
            )
        elif normalized_action == "show":
            if not value:
                raise ValueError("memory show requires a Topic or entry ID.")
            if value in TOPIC_NAMES:
                document = repository.topic_store.read(value)
                typer.echo(repository.topic_store.path_for(value).read_text(encoding="utf-8") if document.entries else f"Topic {value} has no entries.")
            else:
                found = None
                found_topic = None
                for candidate_topic in TOPIC_NAMES:
                    entry = repository.topic_store.get_entry(candidate_topic, value)
                    if entry is not None:
                        found = entry
                        found_topic = candidate_topic
                        break
                if found is None:
                    raise KeyError(f"Memory entry not found: {value}")
                typer.echo(f"Topic: {found_topic}")
                typer.echo(found.model_dump_json(indent=2))
        elif normalized_action == "forget":
            if not value:
                raise ValueError("memory forget requires an entry ID.")
            found = None
            found_topic = None
            for candidate_topic in TOPIC_NAMES:
                entry = repository.topic_store.get_entry(candidate_topic, value)
                if entry is not None:
                    found = entry
                    found_topic = candidate_topic
                    break
            if found is None or found_topic is None:
                raise KeyError(f"Memory entry not found: {value}")
            repository.topic_store.deactivate_entry(
                topic=found_topic,
                entry_id=value,
            )
            repository.refresh_index()
            repository.event_store.append(
                "memory_action_applied",
                operation_id=f"explicit_forget:{value}",
                action="DEACTIVATE",
                topic=found_topic,
                source_note_seqs=[],
                entry_ids=[value],
            )
            typer.echo(f"Forgot memory entry {value}.")
        else:
            raise ValueError(
                "Repository Memory action must be list, remember, review, candidates, approve, "
                "reject, show, or forget."
            )
    except (
        OSError,
        ValueError,
        KeyError,
        MemoryManualReviewError,
        ModelClientConfigurationError,
    ) as exc:
        typer.echo(f"Memory failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@bench_app.command("run")
def bench_run(
    suite: Path = typer.Argument(..., help="Benchmark suite path."),
    output: Path = typer.Option(
        default_run_root() / "bench_demo",
        "--output",
        "-o",
        help="Benchmark output directory.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Model provider name. Defaults to MINICODE_PROVIDER or qwen.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name. Defaults to MINICODE_MODEL when set.",
    ),
    max_steps: int = typer.Option(20, "--max-steps", min=1, help="Maximum agent steps per task."),
    max_tool_calls: int = typer.Option(
        30,
        "--max-tool-calls",
        min=1,
        help="Maximum tool calls per task.",
    ),
    no_repository_memory: bool = typer.Option(
        False,
        "--no-repository-memory",
        help="Disable Repository Memory for benchmark runs.",
    ),
) -> None:
    """Run a benchmark suite."""
    resolved_provider = _resolve_provider(provider)
    resolved_model = _resolve_model(model)
    try:
        summary = BenchmarkRunner(
            BenchmarkRunnerConfig(
                provider=resolved_provider,
                model=resolved_model,
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
                repository_memory_enabled=not no_repository_memory,
            )
        ).run_suite(suite, output)
    except (FileNotFoundError, ValueError, ModelClientConfigurationError) as exc:
        typer.echo(f"Benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Benchmark Suite: {summary.suite}")
    typer.echo(f"Total Tasks: {summary.total_tasks}")
    typer.echo(f"Resolved: {summary.resolved}")
    typer.echo(f"Resolve Rate: {summary.resolve_rate:.0%}")
    typer.echo(f"Report saved to {Path(output) / 'benchmark_report.md'}")


@bench_app.command("scenario")
def bench_scenario(
    suite: Path = typer.Argument(..., help="Multi-turn benchmark scenario suite path."),
    output: Path = typer.Option(
        default_run_root() / "bench_scenarios",
        "--output",
        "-o",
        help="Benchmark output directory.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Model provider name. Defaults to MINICODE_PROVIDER or qwen.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name. Defaults to MINICODE_MODEL when set.",
    ),
    max_steps: int = typer.Option(50, "--max-steps", min=1),
    max_tool_calls: int = typer.Option(60, "--max-tool-calls", min=1),
    memory_mode: Optional[str] = typer.Option(
        None,
        "--memory-mode",
        help="Override scenario memory mode: off, index_only, or index_topic.",
    ),
    context_compaction_mode: Optional[str] = typer.Option(
        None,
        "--context-compaction-mode",
        help="Override context compaction mode: deterministic or llm_hard.",
    ),
    ablation: bool = typer.Option(
        False,
        "--ablation",
        help="Run the fixed memory-recall ablation matrix.",
    ),
    scenario_id: Optional[str] = typer.Option(
        None,
        "--scenario-id",
        help="Run only one scenario from the suite.",
    ),
    context_budget: Optional[int] = typer.Option(
        None,
        "--context-budget",
        min=2000,
        help="Override the selected scenario context budget.",
    ),
    baseline: bool = typer.Option(
        False,
        "--baseline",
        help="Treat required trace events as observational so low-pressure baselines can pass without compaction.",
    ),
) -> None:
    """Run shared-session memory and context-compaction scenarios."""

    resolved_provider = _resolve_provider(provider)
    resolved_model = _resolve_model(model)
    try:
        resolved_memory_mode = _benchmark_mode(
            memory_mode,
            allowed={"off", "index_only", "index_topic"},
            label="memory mode",
        )
        resolved_context_compaction_mode = _benchmark_mode(
            context_compaction_mode,
            allowed={"deterministic", "llm_hard"},
            label="context compaction mode",
        )
        summary = BenchmarkScenarioRunner(
            BenchmarkScenarioRunnerConfig(
                provider=resolved_provider,
                model=resolved_model,
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
                ablation_matrix=ablation,
                memory_mode=resolved_memory_mode,
                context_compaction_mode=resolved_context_compaction_mode,
                scenario_id=scenario_id,
                context_budget_override=context_budget,
                baseline_mode=baseline,
            )
        ).run_suite(suite, output)
    except (FileNotFoundError, ValueError, ModelClientConfigurationError) as exc:
        typer.echo(f"Scenario benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Scenario Suite: {summary.suite}")
    typer.echo(f"Scenario Runs: {summary.total_runs}")
    typer.echo(f"Resolved Runs: {summary.resolved_runs}")
    typer.echo(f"Resolve Rate: {summary.resolve_rate:.0%}")
    typer.echo(f"Report saved to {Path(output) / 'benchmark_report.md'}")



@bench_app.command("swebench")
def bench_swebench(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Local JSON/JSONL path or Hugging Face SWE-bench dataset name.",
    ),
    output: Path = typer.Option(
        default_run_root() / "swebench_verified",
        "--output",
        "-o",
        help="Suite output directory.",
    ),
    instance_ids: Optional[list[str]] = typer.Option(
        None,
        "--instance-id",
        "--instance-ids",
        help="Instance ID to select; repeat the option for multiple IDs.",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
    shard_id: Optional[int] = typer.Option(None, "--shard-id", min=0),
    num_shards: Optional[int] = typer.Option(None, "--num-shards", min=1),
    dataset_split: str = typer.Option("test", "--dataset-split"),
    dataset_revision: Optional[str] = typer.Option(None, "--dataset-revision"),
    dataset_cache: Optional[Path] = typer.Option(None, "--dataset-cache"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    max_steps: int = typer.Option(40, "--max-steps", min=1),
    max_tool_calls: int = typer.Option(120, "--max-tool-calls", min=1),
    max_elapsed_seconds: Optional[int] = typer.Option(
        None,
        "--max-elapsed-seconds",
        min=1,
    ),
    attempts: int = typer.Option(1, "--attempts", min=1),
    repo_cache: Optional[Path] = typer.Option(None, "--repo-cache"),
    no_agent_verification: bool = typer.Option(False, "--no-agent-verification"),
    no_skills: bool = typer.Option(False, "--no-skills"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(
        True,
        "--skip-completed/--no-skip-completed",
    ),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    docker_image: Optional[str] = typer.Option(
        None,
        "--docker-image",
        help="Run agent verification commands inside this fixed image.",
    ),
    evaluator_python: Optional[Path] = typer.Option(
        None,
        "--evaluator-python",
        help="Python executable from an environment containing official SWE-bench.",
    ),
    evaluator_dataset_name: Optional[str] = typer.Option(
        None,
        "--evaluator-dataset-name",
        help="Official dataset name used by the evaluator.",
    ),
    evaluation_workers: int = typer.Option(1, "--evaluation-workers", min=1),
    evaluation_timeout: int = typer.Option(3600, "--evaluation-timeout", min=1),
    docker_cache_level: Optional[str] = typer.Option(None, "--docker-cache-level"),
) -> None:
    """Run MiniCode through the external SWE-bench adapter."""

    resolved_provider = _resolve_provider(provider)
    resolved_model = _resolve_model(model)
    command_executor_factory = None
    if docker_image:
        command_executor_factory = lambda instance, manifest: SweBenchDockerCommandExecutor(
            image=docker_image
        )

    evaluator = None
    official_dataset = evaluator_dataset_name
    if evaluator_python is not None:
        if official_dataset is None:
            if Path(dataset).expanduser().is_file():
                typer.echo(
                    "SWE-bench failed: --evaluator-dataset-name is required for a local dataset file.",
                    err=True,
                )
                raise typer.Exit(code=1)
            official_dataset = dataset
        evaluator = SweBenchEvaluator(
            evaluator_python=evaluator_python,
            output_dir=output / "evaluation",
            dataset_split=dataset_split,
            evaluation_workers=evaluation_workers,
            evaluation_timeout=evaluation_timeout,
            docker_cache_level=docker_cache_level,
        )

    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider=resolved_provider,
            model=resolved_model,
            budget=SweBenchBudget(
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
                max_elapsed_seconds=max_elapsed_seconds,
            ),
            agent_verification_enabled=not no_agent_verification,
            enable_subagents=False,
            enable_skills=not no_skills,
            attempts=attempts,
            resume=resume,
            skip_completed=skip_completed,
            retry_failed=retry_failed,
        ),
        command_executor_factory=command_executor_factory,
    )
    try:
        summary = runner.run_dataset(
            dataset,
            output,
            instance_ids=instance_ids,
            offset=offset,
            limit=limit,
            shard_id=shard_id,
            num_shards=num_shards,
            dataset_split=dataset_split,
            dataset_revision=dataset_revision,
            dataset_cache=dataset_cache,
            repository_cache=repo_cache,
            evaluator=evaluator,
            evaluator_dataset_name=official_dataset,
        )
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        ModelClientConfigurationError,
    ) as exc:
        typer.echo(f"SWE-bench failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"SWE-bench Dataset: {summary.dataset}")
    typer.echo(f"Total Instances: {summary.total_instances}")
    typer.echo(f"Patches Generated: {summary.patches_generated}")
    typer.echo(f"Resolved: {summary.resolved}")
    typer.echo(f"Resolve Rate: {summary.resolve_rate:.2%}")
    typer.echo(f"Predictions: {output / 'predictions.jsonl'}")
    typer.echo(f"Report: {output / 'swebench_report.md'}")


@worktree_app.command("create")
def worktree_create(
    name: str = typer.Argument(..., help="Managed Worktree Session name."),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w"),
    base_ref: str = typer.Option("HEAD", "--base-ref"),
) -> None:
    """Create or reopen one isolated Worktree Session."""

    manifest = GitWorktreeManager().open_or_create_session(
        workspace,
        name,
        base_ref=base_ref,
    )
    typer.echo(manifest.workspace_root)


@worktree_app.command("list")
def worktree_list(
    workspace: Path = typer.Option(Path("."), "--workspace", "-w"),
) -> None:
    """List managed worktrees for one project."""

    statuses = GitWorktreeManager().list(workspace)
    if not statuses:
        typer.echo("No managed worktrees.")
        return
    for status in statuses:
        marker = "dirty" if status.dirty else "clean"
        typer.echo(
            f"{status.manifest.kind}\t{status.manifest.name}\t{marker}\t"
            f"{status.manifest.workspace_root}"
        )


@worktree_app.command("remove")
def worktree_remove(
    name: str = typer.Argument(..., help="Managed worktree name."),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w"),
    force: bool = typer.Option(False, "--force", help="Discard changes and commits."),
) -> None:
    """Remove a managed worktree after a deterministic dirty-state check."""

    GitWorktreeManager().remove(workspace, name, force=force)
    typer.echo(f"Removed {name}")


app.add_typer(bench_app, name="bench")
app.add_typer(worktree_app, name="worktree")


def main() -> None:
    """Console script entry point."""
    raw_args = sys.argv[1:]
    if raw_args in (["--help"], ["-h"]):
        typer.echo(_main_help(Path(sys.argv[0]).name or "minicode"))
        return
    if raw_args and (
        raw_args[0] in KNOWN_COMMANDS or raw_args[0] in GLOBAL_OPTIONS
    ):
        app(
            args=raw_args,
            prog_name=Path(sys.argv[0]).name or "minicode",
        )
        return
    try:
        launch = _parse_interactive_task_args(raw_args)
    except ValueError as exc:
        typer.echo(f"Session launch failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        if launch.task:
            typer.echo("Interactive MiniCode requires a TTY.", err=True)
            typer.echo(
                'Use `minicode exec "<task>"` for non-interactive execution.',
                err=True,
            )
            raise typer.Exit(code=2)
        if raw_args:
            typer.echo("Interactive sessions require a TTY.", err=True)
            raise typer.Exit(code=2)
        app(args=["--help"], prog_name=Path(sys.argv[0]).name or "minicode")
        return
    try:
        _run_terminal(launch)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Session launch failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _run_terminal(launch: InteractiveLaunch) -> None:
    user_config = _load_user_config()
    terminal_workspace = _resolve_worktree_workspace(
        launch.workspace,
        launch.worktree,
    )
    approval_policy = _resolve_approval_policy(
        launch.approval_policy,
        config=user_config,
    )
    permission_mode = _resolve_permission_mode(
        launch.permission_mode,
        config=user_config,
    )
    sandbox_mode = _resolve_sandbox_mode(
        launch.sandbox_mode,
        config=user_config,
    )
    launch_tui(
        workspace=terminal_workspace,
        provider=_resolve_provider(launch.provider, config=user_config),
        model=_resolve_model(launch.model, config=user_config),
        write_enabled=not launch.no_write,
        approval_policy=approval_policy.value,
        permission_mode=permission_mode.value,
        sandbox_mode=sandbox_mode.value,
        sandbox_image=_resolve_sandbox_image(
            sandbox_mode,
            launch.sandbox_image,
        ),
        collaboration_mode=launch.collaboration_mode.value,
        skills=_parse_skill_names(launch.skills),
        skills_enabled=not launch.no_skills,
        repository_memory_enabled=not launch.no_repository_memory,
        subagents_enabled=not launch.no_subagents,
        mcp_config=_resolve_mcp_config(launch.mcp_config),
        session_mode=launch.session_mode.value,
        session_id=launch.session_id,
        initial_task=launch.task,
        no_color=launch.no_color,
    )


def _benchmark_mode(
    value: str | None,
    *,
    allowed: set[str],
    label: str,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {label}: {value}. Expected one of: {choices}.")
    return normalized


def _execution_io(
    *,
    plain: bool,
    no_color: bool,
):
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not plain
    if not interactive:
        return TextOutputSink(), NonInteractiveApprovalClient(), None
    console = Console(no_color=no_color)
    renderer = TerminalRenderer(console)
    return (
        TerminalOutputSink(console=console),
        TerminalApprovalClient(console=console),
        renderer,
    )


def _parse_skill_names(skills: str | None) -> list[str] | None:
    if skills is None:
        return None
    return [skill.strip() for skill in skills.split(",") if skill.strip()]


def _main_help(program_name: str) -> str:
    return "\n".join(
        [
            "MiniCodeHarness - local general-purpose coding agent for software repositories.",
            "",
            "Usage:",
            f"  {program_name}",
            f"  {program_name} --continue",
            f'  {program_name} "<task>" [options]',
            f"  {program_name} resume [session_id] [options]",
            f'  {program_name} exec "<task>" [options]',
            f"  {program_name} recover [run_id] [--workspace <dir>]",
            f"  {program_name} trace [run_id]",
            f"  {program_name} report [run_id]",
            f"  {program_name} sessions [--workspace <dir>]",
            f"  {program_name} runs [run_id] [--workspace <dir>]",
            f"  {program_name} memory [--workspace <dir>]",
            f"  {program_name} worktree create|list|remove",
            f"  {program_name} bench run <suite> --output <dir>",
            "",
            "Interactive session options:",
            "  --continue, -c         Continue the latest workspace session",
            "  --worktree <name>      Open/create an isolated Worktree Session",
            "",
            "Session and Run recovery:",
            "  resume [session_id]   Resume a conversation Session (TTY required)",
            "  recover [run_id]      Recover an interrupted Run Checkpoint",
            "",
            "Task options:",
            "  --no-write             Disable write tools",
            "  --permission-mode <m>  read-only, workspace-write, or full-access",
            "  --approval-policy <p>  on-request or never",
            "  --sandbox <m>          local or docker",
            "  --sandbox-image <img>  Docker image for --sandbox docker",
            "  --mode <m>             default or plan",
            "  --provider <name>      qwen, kimi, openai, anthropic, deepseek, or ollama",
            "  --model <name>         Model name",
            "  --no-subagents         Disable bounded read-only delegation",
            "  --no-color             Disable terminal colors",
            "  --mcp-config <file>    Load explicit local MCP stdio tools",
            "  --workspace, -w <dir>  Override the current directory workspace",
            "  --worktree <name>      Run in an isolated Worktree Session",
            "  exec only: --dry-run, --debug-trace, --plain",
            "",
            "User defaults:",
            "  ~/.minicode/config.toml  provider, model, permission_mode, approval_policy, sandbox",
            "  precedence: CLI > existing environment defaults > user config > code defaults",
            "",
            "Environment defaults:",
            "  MINICODE_PROVIDER",
            "  MINICODE_MODEL",
            "  MINICODE_MCP_CONFIG    explicit local MCP JSON configuration file",
            "  MINICODE_SANDBOX_IMAGE Docker image used by --sandbox docker",
            "  MINICODE_HOME          user data root; Sessions live under <MINICODE_HOME>/sessions",
            "  MINICODE_RUNS_DIR      standalone Run root and benchmark default base",
            "",
            f"Run `{program_name} exec --help` for all execution options.",
        ]
    )


def _parse_approval_policy(value: str | None) -> ApprovalPolicy:
    if value is None:
        return DEFAULT_APPROVAL_POLICY
    try:
        return ApprovalPolicy(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in ApprovalPolicy)
        raise ValueError(
            f"Invalid approval policy: {value}. Expected one of: {choices}."
        ) from exc


def _parse_permission_mode(value: str | None) -> PermissionMode:
    if value is None:
        return DEFAULT_PERMISSION_MODE
    try:
        return PermissionMode(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in PermissionMode)
        raise ValueError(
            f"Invalid permission mode: {value}. Expected one of: {choices}."
        ) from exc


def _parse_sandbox_mode(value: str | None) -> SandboxMode:
    if value is None:
        return SandboxMode.LOCAL
    try:
        return SandboxMode(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in SandboxMode)
        raise ValueError(
            f"Invalid sandbox mode: {value}. Expected one of: {choices}."
        ) from exc


def _parse_collaboration_mode(value: str | None) -> CollaborationMode:
    if value is None:
        return DEFAULT_COLLABORATION_MODE
    try:
        return CollaborationMode(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in CollaborationMode)
        raise ValueError(
            f"Invalid collaboration mode: {value}. Expected one of: {choices}."
        ) from exc


def _resolve_approval_policy(
    value: ApprovalPolicy | str | None,
    *,
    config: UserConfig | None = None,
) -> ApprovalPolicy:
    configured = value if value is not None else getattr(config, "approval_policy", None)
    if isinstance(configured, ApprovalPolicy):
        return configured
    return _parse_approval_policy(configured)


def _resolve_permission_mode(
    value: PermissionMode | str | None,
    *,
    config: UserConfig | None = None,
) -> PermissionMode:
    configured = value if value is not None else getattr(config, "permission_mode", None)
    if isinstance(configured, PermissionMode):
        return configured
    return _parse_permission_mode(configured)


def _resolve_sandbox_mode(
    value: SandboxMode | str | None,
    *,
    config: UserConfig | None = None,
) -> SandboxMode:
    configured = value if value is not None else getattr(config, "sandbox", None)
    if isinstance(configured, SandboxMode):
        return configured
    return _parse_sandbox_mode(configured)


def _resolve_provider(
    provider: str | None,
    *,
    config: UserConfig | None = None,
) -> str:
    configured = getattr(config, "provider", None)
    return (
        provider
        or os.environ.get("MINICODE_PROVIDER")
        or configured
        or "qwen"
    ).strip()


def _resolve_model(
    model: str | None,
    *,
    config: UserConfig | None = None,
) -> str | None:
    resolved_model = (
        model
        or os.environ.get("MINICODE_MODEL")
        or getattr(config, "model", None)
    )
    if resolved_model is None:
        return None
    resolved_model = resolved_model.strip()
    return resolved_model or None


def _resolve_sandbox_image(
    mode: SandboxMode,
    value: str | None,
) -> str | None:
    if mode == SandboxMode.LOCAL:
        return None
    resolved = value or os.environ.get("MINICODE_SANDBOX_IMAGE")
    if resolved is None or not resolved.strip():
        raise ValueError(
            "Docker sandbox requires --sandbox-image or MINICODE_SANDBOX_IMAGE."
        )
    return resolved.strip()


def _resolve_mcp_config(value: Path | None) -> Path | None:
    configured = value
    if configured is None and os.environ.get("MINICODE_MCP_CONFIG"):
        configured = Path(os.environ["MINICODE_MCP_CONFIG"])
    return configured.expanduser().resolve() if configured is not None else None


def _resolve_run_id(
    run_id: str | None,
    *,
    workspace: Path | str | None = None,
) -> str:
    if run_id:
        return run_id
    return RunStore().latest_run_id(workspace=workspace)


def _resolve_worktree_workspace(workspace: Path, name: str | None) -> Path:
    resolved = workspace.expanduser().resolve()
    if name is None:
        return resolved
    manifest = GitWorktreeManager().open_or_create_session(resolved, name)
    return Path(manifest.workspace_root)


def _parse_interactive_task_args(args: list[str]) -> InteractiveLaunch:
    values: dict[str, str] = {}
    flags: set[str] = set()
    task_parts: list[str] = []
    session_mode = SessionLaunchMode.NEW
    session_id: str | None = None
    task_only = False
    index = 0
    while index < len(args):
        token = args[index]
        if task_only:
            task_parts.append(token)
            index += 1
            continue
        if token == "--":
            task_only = True
            index += 1
            continue
        if token in {"--continue", "-c"}:
            session_mode = SessionLaunchMode.CONTINUE
            index += 1
            continue
        if token in INTERACTIVE_OPTIONS_WITH_VALUES:
            if (
                index + 1 >= len(args)
                or not args[index + 1].strip()
                or args[index + 1].startswith("-")
            ):
                raise ValueError(f"{token} requires a value.")
            option_name = "--workspace" if token == "-w" else token
            values[option_name] = args[index + 1]
            index += 2
            continue
        if token in INTERACTIVE_FLAG_OPTIONS:
            flags.add(token)
            index += 1
            continue
        if token in EXEC_ONLY_OPTIONS:
            raise ValueError(f"{token} is available only with `minicode exec`.")
        if token.startswith("-"):
            raise ValueError(f"Unknown interactive option: {token}")
        task_parts.append(token)
        index += 1

    if session_mode != SessionLaunchMode.NEW and task_parts:
        raise ValueError("Continued Sessions do not accept an initial task.")
    task = " ".join(task_parts) if task_parts else None
    user_config = _load_user_config()
    return InteractiveLaunch(
        task=task,
        workspace=Path(values.get("--workspace", ".")),
        worktree=values.get("--worktree"),
        provider=_resolve_provider(values.get("--provider"), config=user_config),
        model=_resolve_model(values.get("--model"), config=user_config),
        no_write="--no-write" in flags,
        approval_policy=_resolve_approval_policy(
            values.get("--approval-policy"),
            config=user_config,
        ),
        permission_mode=_resolve_permission_mode(
            values.get("--permission-mode"),
            config=user_config,
        ),
        sandbox_mode=_resolve_sandbox_mode(
            values.get("--sandbox"),
            config=user_config,
        ),
        sandbox_image=values.get("--sandbox-image"),
        collaboration_mode=_parse_collaboration_mode(values.get("--mode")),
        skills=values.get("--skills"),
        no_skills="--no-skills" in flags,
        mcp_config=(
            Path(values["--mcp-config"])
            if "--mcp-config" in values
            else None
        ),
        no_subagents="--no-subagents" in flags,
        no_repository_memory="--no-repository-memory" in flags,
        no_color="--no-color" in flags,
        session_mode=session_mode,
        session_id=session_id,
    )
