"""Slash command parsing and execution outside the Typer command graph."""

from __future__ import annotations

from dataclasses import dataclass
import shlex

from minicode_harness.history import (
    format_conversation_sessions,
    format_project_memory,
    format_run_detail,
    format_run_history,
)
from minicode_harness.report import (
    format_context_usage,
    format_run_trace,
    generate_run_report,
)
from minicode_harness.resume import latest_recoverable_run_id, resume_run
from minicode_harness.state import UserInputRequest
from minicode_harness.policy import ApprovalPolicy, PermissionMode
from minicode_harness.runtime import CollaborationMode
from minicode_harness.terminal.status import (
    build_terminal_status,
    display_model_name,
    render_terminal_status,
)
from minicode_harness.terminal.types import CommandResult, SlashCommand, TerminalContext
from minicode_harness.tools import inspect_git_diff


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    argument_hint: str | None = None
    argument_choices: tuple[str, ...] = ()
    availability: str = "idle"


_COMMAND_SPECS = (
    CommandSpec(
        "permissions",
        "View or change runtime permissions",
        "[mode <value> | approval <value>]",
        (
            "mode read-only",
            "mode workspace-write",
            "mode full-access",
            "approval on-request",
            "approval never",
        ),
    ),
    CommandSpec("plan", "Set planning mode", "[on|off|status]", ("on", "off", "status")),
    CommandSpec("diff", "Show current workspace diff"),
    CommandSpec("review", "Review current Git diff", "[focus]"),
    CommandSpec("context", "Inspect latest context budget", "[run_id]"),
    CommandSpec("compact", "Compact current Session context", "[focus]"),
    CommandSpec("status", "Show workspace and latest Run"),
    CommandSpec("sessions", "Show recent workspace Sessions"),
    CommandSpec("runs", "Show Runs in the current Session", "[run_id]"),
    CommandSpec("rename", "Rename the current Session", "<name>"),
    CommandSpec("fork", "Fork the current Session", "[run]"),
    CommandSpec("recover", "Recover a Run in the current Session", "[run_id]"),
    CommandSpec("memory", "Show Repository Memory index"),
    CommandSpec("trace", "Show latest Run trace", "[run_id]"),
    CommandSpec("report", "Generate a Run report", "[run_id]"),
    CommandSpec("model", "Show current provider and model"),
    CommandSpec("help", "Show command reference"),
    CommandSpec("exit", "Exit MiniCode"),
)

SLASH_COMMANDS = {spec.name for spec in _COMMAND_SPECS}


def command_catalog() -> tuple[CommandSpec, ...]:
    """Return the ordered terminal command catalog used by help and TUI completion."""

    return _COMMAND_SPECS


def parse_slash_command(text: str) -> SlashCommand:
    """Parse only slash-prefixed input; normal tasks never use this parser."""

    if not text.startswith("/"):
        raise ValueError("Slash commands must start with '/'.")
    try:
        tokens = shlex.split(text, posix=False)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not tokens:
        raise ValueError("Slash command is empty.")
    name = tokens[0][1:].strip().lower()
    arguments = [_strip_matching_quotes(token) for token in tokens[1:]]
    return SlashCommand(name=name, arguments=arguments)


class SlashCommandRouter:
    """Execute terminal commands directly without re-entering Typer."""

    def execute(
        self,
        command: SlashCommand,
        context: TerminalContext,
    ) -> CommandResult:
        try:
            return self._execute(command, context)
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            return CommandResult(status="failed", content=str(exc))

    def _execute(
        self,
        command: SlashCommand,
        context: TerminalContext,
    ) -> CommandResult:
        if command.name not in SLASH_COMMANDS:
            return CommandResult(
                status="failed",
                content=f"Unknown command: /{command.name}. Use /help to list commands.",
            )
        if command.name == "exit":
            return CommandResult(status="exit")
        if command.name == "help":
            return CommandResult(status="completed", content=_help_text())
        if command.name == "status":
            status = build_terminal_status(
                context.workspace,
                run_store=context.run_store,
                provider=context.provider,
                model=context.model,
                session_id=context.session_id,
            )
            return CommandResult(status="completed", content=render_terminal_status(status))
        if command.name == "model":
            return CommandResult(
                status="completed",
                content="\n".join(
                    [
                        f"Provider: {context.provider}",
                        f"Model: {display_model_name(context.provider, context.model)}",
                        "Streaming: enabled",
                        "Tool calling: enabled",
                    ]
                ),
            )
        if command.name == "plan":
            return self._plan_mode(command, context)
        if command.name == "permissions":
            return self._permissions(command, context)
        if command.name == "rename":
            if context.executor is None or context.executor.session_memory is None:
                return CommandResult(
                    status="failed",
                    content="No active conversation Session is available.",
                )
            name = " ".join(command.arguments).strip()
            if not name:
                raise ValueError("/rename requires a Session name.")
            context.executor.session_memory.rename(name)
            return CommandResult(
                status="completed",
                content=f"Session renamed: {context.executor.session_memory.name}",
            )
        if command.name == "fork":
            if len(command.arguments) > 1:
                raise ValueError("/fork accepts at most one completed Run number.")
            if (
                context.session_store is None
                or context.executor is None
                or context.executor.session_memory is None
            ):
                return CommandResult(
                    status="failed",
                    content="No active persisted Session is available for fork.",
                )
            through_turn = None
            if command.arguments:
                try:
                    through_turn = int(command.arguments[0])
                except ValueError as exc:
                    raise ValueError("/fork Run number must be an integer.") from exc
            forked = context.session_store.fork(
                context.executor.session_memory,
                through_turn=through_turn,
            )
            context.executor.session_memory = forked
            suffix = f" through Run {through_turn}" if through_turn is not None else ""
            return CommandResult(
                status="completed",
                content=f"Forked Session{suffix}: {forked.session_id}",
                session_id=forked.session_id,
            )
        if command.name == "review":
            if context.executor is None:
                return CommandResult(
                    status="failed",
                    content="No active session executor is available.",
                )
            if not context.subagents_enabled:
                return CommandResult(
                    status="failed",
                    content="Review requires read-only subagents to be enabled.",
                )
            result = context.executor.review_current_diff(
                workspace=context.workspace,
                provider=context.provider,
                model=context.model,
                focus=" ".join(command.arguments),
                cancellation_token=context.cancellation_token,
            )
            run_line = f"Review Run: {result.run_id}\n" if result.run_id else ""
            return CommandResult(
                status="completed",
                content=f"{run_line}{result.summary}",
            )
        if command.name == "sessions":
            if command.arguments:
                return CommandResult(
                    status="failed",
                    content="/sessions does not accept arguments.",
                )
            content = format_conversation_sessions(
                context.workspace,
                session_store=context.session_store,
            )
            return CommandResult(status="completed", content=content)
        if command.name == "runs":
            content = self._run_history(command, context)
            return CommandResult(status="completed", content=content)
        if command.name == "memory":
            if command.arguments:
                return CommandResult(
                    status="failed",
                    content="Interactive /memory currently displays the Repository Memory index only.",
                )
            return CommandResult(
                status="completed",
                content=format_project_memory(context.workspace),
            )
        if command.name == "context":
            run_id = self._session_run_id(command, context)
            return CommandResult(
                status="completed",
                content=format_context_usage(run_id, context.run_store),
            )
        if command.name == "compact":
            if context.executor is None:
                return CommandResult(
                    status="failed",
                    content="No active session executor is available.",
                )
            result = context.executor.compact_session(
                provider=context.provider,
                model=context.model,
                focus=" ".join(command.arguments),
            )
            if not result.changed:
                return CommandResult(
                    status="completed",
                    content=f"Session unchanged: {result.reason}",
                )
            return CommandResult(
                status="completed",
                content=(
                    "Session compacted: "
                    f"{result.before_tokens} -> {result.after_tokens} tokens; "
                    f"removed groups: {result.removed_groups}"
                ),
            )
        if command.name == "trace":
            run_id = self._session_run_id(command, context)
            return CommandResult(
                status="completed",
                content=format_run_trace(run_id, context.run_store),
            )
        if command.name == "report":
            run_id = self._session_run_id(command, context)
            report = generate_run_report(run_id, context.run_store)
            return CommandResult(
                status="completed",
                content=(
                    f"Report: {report.report_path}\n"
                    f"Final diff: {report.final_diff_path}"
                ),
            )
        if command.name == "diff":
            diff = inspect_git_diff(context.workspace)
            if diff.returncode != 0:
                return CommandResult(
                    status="failed",
                    content=diff.stderr.strip() or "git diff failed.",
                )
            return CommandResult(
                status="completed",
                content=diff.diff.rstrip() or "No git diff.",
            )
        if command.name == "recover":
            if len(command.arguments) > 1:
                raise ValueError("/recover accepts at most one Run ID.")
            if command.arguments:
                run_id = self._session_run_id(command, context)
            else:
                run_id = latest_recoverable_run_id(
                    run_store=context.run_store,
                    conversation_session_id=context.session_id,
                )
            resumed = resume_run(
                run_id,
                approval_client=context.approval_client,
                output_sink=context.output_sink,
                stream_model=True,
                cancellation_token=context.cancellation_token,
                steering_queue=context.steering_queue,
            )
            if resumed.status == "blocked":
                conflicts = "\n".join(
                    f"- {path}" for path in (resumed.conflicts or [])
                )
                content = f"Recovery blocked: {resumed.reason}"
                if conflicts:
                    content += "\nConflicting files:\n" + conflicts
                return CommandResult(status="failed", content=content)
            return CommandResult(
                status="completed",
                content=f"Recovery {resumed.status}: {resumed.reason}",
            )
        raise AssertionError(f"Unhandled slash command: {command.name}")

    @staticmethod
    def _run_history(command: SlashCommand, context: TerminalContext) -> str:
        if len(command.arguments) > 1:
            raise ValueError(f"/{command.name} accepts at most one Run ID.")
        if command.arguments:
            return format_run_detail(
                command.arguments[0],
                run_store=context.run_store,
                conversation_session_id=context.session_id,
            )
        return format_run_history(
            limit=10,
            run_store=context.run_store,
            conversation_session_id=context.session_id,
        )

    @staticmethod
    def _permissions(
        command: SlashCommand,
        context: TerminalContext,
    ) -> CommandResult:
        settings = context.session_settings
        if not command.arguments and context.user_input_client is not None:
            mode_response = context.user_input_client.choose(
                UserInputRequest(
                    question="Choose the permission mode for subsequent Runs.",
                    options=[
                        {
                            "label": "Read only",
                            "description": "Read and inspect; side effects require approval or stay unavailable.",
                        },
                        {
                            "label": "Workspace write",
                            "description": "Allow admitted workspace edits without per-edit approval.",
                        },
                        {
                            "label": "Full access",
                            "description": "Allow admitted side effects with fewer approval prompts.",
                        },
                    ],
                )
            )
            selected_permission_mode = (
                PermissionMode.READ_ONLY,
                PermissionMode.WORKSPACE_WRITE,
                PermissionMode.FULL_ACCESS,
            )[mode_response.selected_index]
            approval_response = context.user_input_client.choose(
                UserInputRequest(
                    question="Choose the approval policy for subsequent Runs.",
                    options=[
                        {
                            "label": "On request",
                            "description": "Ask when the active permission mode requires approval.",
                        },
                        {
                            "label": "Never",
                            "description": "Do not prompt; denied operations remain blocked.",
                        },
                    ],
                )
            )
            selected_approval_policy = (
                ApprovalPolicy.ON_REQUEST,
                ApprovalPolicy.NEVER,
            )[approval_response.selected_index]
            settings.permission_mode = selected_permission_mode
            settings.approval_policy = selected_approval_policy
            return CommandResult(
                status="completed",
                content=(
                    f"Permission mode: {settings.permission_mode.value}\n"
                    f"Approval policy: {settings.approval_policy.value}\n"
                    "Applied to subsequent Runs."
                ),
            )
        if not command.arguments or command.arguments == ["status"]:
            return CommandResult(
                status="completed",
                content="\n".join(
                    [
                        f"Workspace: {context.workspace}",
                        f"Write tools enabled: {context.write_enabled}",
                        f"Permission mode: {settings.permission_mode.value}",
                        f"Approval policy: {settings.approval_policy.value}",
                        f"Command sandbox: {context.sandbox_mode.value}",
                        *(
                            [f"Sandbox image: {context.sandbox_image}"]
                            if context.sandbox_image
                            else []
                        ),
                        f"Repository Memory: {context.repository_memory_enabled}",
                        f"Subagents enabled: {context.subagents_enabled}",
                        f"MCP config: {context.mcp_config or 'none'}",
                    ]
                ),
            )
        if len(command.arguments) != 2:
            raise ValueError(
                "/permissions accepts: status, mode <value>, or approval <value>."
            )
        field, value = (item.strip().lower() for item in command.arguments)
        if field == "mode":
            try:
                settings.permission_mode = PermissionMode(value)
            except ValueError as exc:
                raise ValueError(
                    "Permission mode must be read-only, workspace-write, or full-access."
                ) from exc
            return CommandResult(
                status="completed",
                content=f"Permission mode set to {settings.permission_mode.value} for subsequent Runs.",
            )
        if field == "approval":
            try:
                settings.approval_policy = ApprovalPolicy(value)
            except ValueError as exc:
                raise ValueError("Approval policy must be on-request or never.") from exc
            return CommandResult(
                status="completed",
                content=f"Approval policy set to {settings.approval_policy.value} for subsequent Runs.",
            )
        raise ValueError(
            "/permissions accepts: status, mode <value>, or approval <value>."
        )

    @staticmethod
    def _plan_mode(
        command: SlashCommand,
        context: TerminalContext,
    ) -> CommandResult:
        if len(command.arguments) > 1:
            raise ValueError("/plan accepts at most one argument: on, off, or status.")
        if not command.arguments and context.user_input_client is not None:
            response = context.user_input_client.choose(
                UserInputRequest(
                    question="Choose the working mode for subsequent Runs.",
                    options=[
                        {
                            "label": "Default",
                            "description": "Allow the normal tool surface under the current permissions.",
                        },
                        {
                            "label": "Plan",
                            "description": "Use read-only exploration and planning tools.",
                        },
                    ],
                )
            )
            context.session_settings.collaboration_mode = (
                CollaborationMode.DEFAULT,
                CollaborationMode.PLAN,
            )[response.selected_index]
            enabled = (
                context.session_settings.collaboration_mode == CollaborationMode.PLAN
            )
            return CommandResult(
                status="completed",
                content=f"Plan mode: {'on' if enabled else 'off'} for subsequent Runs.",
            )
        action = command.arguments[0].strip().lower() if command.arguments else "status"
        if action == "status":
            enabled = context.session_settings.collaboration_mode == CollaborationMode.PLAN
            return CommandResult(
                status="completed",
                content=(
                    f"Plan mode: {'on' if enabled else 'off'}\n"
                    f"Next Run mode: {context.session_settings.collaboration_mode.value}"
                ),
            )
        if action == "on":
            context.session_settings.collaboration_mode = CollaborationMode.PLAN
            return CommandResult(
                status="completed",
                content="Plan mode enabled for subsequent Runs.",
            )
        if action == "off":
            context.session_settings.collaboration_mode = CollaborationMode.DEFAULT
            return CommandResult(
                status="completed",
                content="Plan mode disabled for subsequent Runs.",
            )
        raise ValueError("/plan accepts: on, off, or status.")

    @staticmethod
    def _session_run_id(command: SlashCommand, context: TerminalContext) -> str:
        if len(command.arguments) > 1:
            raise ValueError(f"/{command.name} accepts at most one Run ID.")
        if command.arguments:
            run_id = command.arguments[0]
            run = context.run_store.load_session(run_id)
            if run.conversation_session_id != context.session_id:
                if command.name == "recover":
                    raise ValueError(
                        "Run belongs to another Session. Use "
                        "`minicode recover <run_id>` outside the terminal."
                    )
                raise ValueError("Run belongs to another Session.")
            return run_id
        return context.run_store.latest_run_id(
            conversation_session_id=context.session_id,
        )


def _strip_matching_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _help_text() -> str:
    lines = ["Commands:"]
    for spec in command_catalog():
        usage = f"/{spec.name}"
        if spec.argument_hint:
            usage += f" {spec.argument_hint}"
        lines.append(f"  {usage:<34} {spec.description}")
    lines.extend(
        [
            "",
            "Type / in the interactive terminal to browse commands.",
            "Enter submits a task. Esc+Enter or Ctrl+O inserts a newline.",
        ]
    )
    return "\n".join(lines)
