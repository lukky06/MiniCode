"""Slash command parsing and execution outside the Typer command graph."""

from __future__ import annotations

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
from minicode_harness.runtime import CollaborationMode
from minicode_harness.terminal.status import (
    build_terminal_status,
    display_model_name,
    render_terminal_status,
)
from minicode_harness.terminal.types import CommandResult, SlashCommand, TerminalContext
from minicode_harness.tools import inspect_git_diff


SLASH_COMMANDS = {
    "status",
    "sessions",
    "runs",
    "context",
    "compact",
    "memory",
    "trace",
    "recover",
    "report",
    "diff",
    "model",
    "permissions",
    "plan",
    "rename",
    "fork",
    "review",
    "help",
    "exit",
}


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
            return CommandResult(
                status="completed",
                content="\n".join(
                    [
                        f"Workspace: {context.workspace}",
                        f"Write tools enabled: {context.write_enabled}",
                        f"Permission mode: {context.permission_mode.value}",
                        f"Approval policy: {context.approval_policy.value}",
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
    def _plan_mode(
        command: SlashCommand,
        context: TerminalContext,
    ) -> CommandResult:
        if len(command.arguments) > 1:
            raise ValueError("/plan accepts at most one argument: on, off, or status.")
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
    return "\n".join(
        [
            "Commands:",
            "  /status        workspace and latest run",
            "  /sessions      recent workspace Sessions",
            "  /runs          Runs in the current Session",
            "  /context       latest context budget breakdown",
            "  /compact [focus] compact session semantics within normal safety rules",
            "  /memory        repository memory",
            "  /trace         latest run trace",
            "  /recover       recover a Run in the current Session",
            "  /report        generate a run report",
            "  /diff          current git diff",
            "  /model         current provider and model",
            "  /permissions   runtime permissions",
            "  /plan [on|off|status] set the collaboration mode for subsequent Runs",
            "  /rename <name> name the current Session",
            "  /fork [run]    fork the current Session, optionally at a completed Run boundary",
            "  /review [focus] review the current Git diff with a read-only subagent",
            "  /exit          exit MiniCode",
            "",
            "Enter submits a task. Esc+Enter or Ctrl+O inserts a newline.",
        ]
    )
