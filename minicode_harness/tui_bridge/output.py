"""OutputSink implementation that exposes runtime facts as JSONL events."""

from __future__ import annotations

from typing import Any

from minicode_harness.output import ContextUsage
from minicode_harness.terminal.commands import CommandSpec
from minicode_harness.terminal.types import TerminalSessionSettings
from minicode_harness.tools.semantics import read_target

from .protocol import (
    AssistantDelta,
    CommandCatalogEvent,
    ContextCompactionEvent,
    ContextEvent,
    ErrorEvent,
    ExitRequested,
    PanelEvent,
    ReasoningDelta,
    RunFinished,
    RunStarted,
    SessionSettingsEvent,
    SessionStarted,
    ToolFinished,
    ToolStarted,
)
from .writer import JsonlEventWriter


class JsonlOutputSink:
    """Translate the existing OutputSink callbacks without changing runtime semantics."""

    def __init__(self, writer: JsonlEventWriter | None = None) -> None:
        self.writer = writer or JsonlEventWriter()
        self.current_run_id: str | None = None

    def session_started(self, session_id: str) -> None:
        self.writer.emit(SessionStarted(session_id=session_id))

    def command_catalog(self, commands: tuple[CommandSpec, ...]) -> None:
        self.writer.emit(
            CommandCatalogEvent(
                commands=[
                    {
                        "name": command.name,
                        "description": command.description,
                        "argument_hint": command.argument_hint,
                        "argument_choices": list(command.argument_choices),
                        "availability": command.availability,
                    }
                    for command in commands
                ]
            )
        )

    def session_settings(self, settings: TerminalSessionSettings) -> None:
        self.writer.emit(
            SessionSettingsEvent(
                permission_mode=settings.permission_mode.value,
                approval_policy=settings.approval_policy.value,
                collaboration_mode=settings.collaboration_mode.value,
            )
        )

    def run_started(self, run_id: str) -> None:
        self.current_run_id = run_id
        self.writer.emit(RunStarted(run_id=run_id))

    def run_finished(
        self,
        *,
        status: str,
        run_id: str | None = None,
        stop_reason: str | None = None,
    ) -> None:
        resolved_run_id = run_id or self.current_run_id
        self.writer.emit(
            RunFinished(
                status=status,
                run_id=resolved_run_id,
                stop_reason=stop_reason,
            )
        )
        self.current_run_id = None

    def error(self, message: str, *, fatal: bool = False) -> None:
        self.writer.emit(ErrorEvent(message=message, fatal=fatal))

    def panel(self, *, name: str, title: str, content: str) -> None:
        self.writer.emit(PanelEvent(name=name, title=title, content=content))

    def exit_requested(self) -> None:
        self.writer.emit(ExitRequested())

    def context_built(self, *, usage: ContextUsage) -> None:
        self.writer.emit(
            ContextEvent(
                used=usage.token_estimate,
                window=usage.context_window,
                prompt_budget=usage.prompt_budget,
                reserved_output=usage.reserved_output,
            )
        )

    def context_compaction_started(self, *, kind: str) -> None:
        if kind != "semantic":
            return
        self.writer.emit(
            ContextCompactionEvent(
                kind="semantic",
                phase="started",
            )
        )

    def context_compaction_finished(
        self,
        *,
        kind: str,
        duration_ms: int,
        success: bool,
    ) -> None:
        if kind != "semantic":
            return
        self.writer.emit(
            ContextCompactionEvent(
                kind="semantic",
                phase="completed" if success else "failed",
                duration_ms=max(0, duration_ms),
            )
        )

    def model_stream_started(self) -> None:
        pass

    def model_text_delta(self, text: str) -> None:
        if text:
            self.writer.emit(AssistantDelta(text=text))

    def model_reasoning_delta(self, text: str) -> None:
        if text:
            self.writer.emit(ReasoningDelta(text=text))

    def tool_call_started(
        self,
        *,
        step: int,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> None:
        self.writer.emit(
            ToolStarted(
                id=tool_call_id or f"fallback:{step}:{tool_name}",
                step=step,
                tool=tool_name,
                target=_tool_target(tool_name, arguments),
            )
        )

    def tool_call_finished(
        self,
        *,
        step: int,
        tool_name: str,
        status: str,
        tool_call_id: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        safe_metadata = metadata or {}
        command_status = None
        returncode = None
        duration_ms = None
        runtime_task_id = None
        diff_preview = None
        diff_truncated = False
        if tool_name == "run_command":
            raw_status = safe_metadata.get("command_status")
            if isinstance(raw_status, str):
                command_status = _bounded_line(raw_status, limit=80)
            raw_returncode = safe_metadata.get("returncode")
            if isinstance(raw_returncode, int):
                returncode = raw_returncode
            raw_duration_ms = safe_metadata.get("duration_ms")
            if isinstance(raw_duration_ms, int) and raw_duration_ms >= 0:
                duration_ms = raw_duration_ms
            raw_task_id = safe_metadata.get("runtime_task_id")
            if isinstance(raw_task_id, str):
                runtime_task_id = _bounded_line(raw_task_id, limit=80)
        raw_diff = safe_metadata.get("diff_preview")
        if isinstance(raw_diff, str) and tool_name in {"edit", "write", "apply_patch"}:
            diff_preview = raw_diff
            diff_truncated = bool(safe_metadata.get("diff_truncated", False))
        self.writer.emit(
            ToolFinished(
                id=tool_call_id or f"fallback:{step}:{tool_name}",
                step=step,
                tool=tool_name,
                status=status,
                summary=_bounded_line(summary),
                command_status=command_status,
                returncode=returncode,
                duration_ms=duration_ms,
                runtime_task_id=runtime_task_id,
                diff_preview=diff_preview,
                diff_truncated=diff_truncated,
            )
        )


def _tool_target(tool_name: str, arguments: dict[str, Any]) -> str | None:
    target = read_target(tool_name, arguments)
    if target is not None:
        return _bounded_line(target)

    if tool_name == "search":
        return _bounded_line(arguments.get("query"))
    if tool_name in {"edit", "write", "apply_patch"}:
        return _bounded_line(arguments.get("path"))
    if tool_name == "run_command":
        argv = arguments.get("argv")
        if isinstance(argv, list):
            return _bounded_line(" ".join(str(part) for part in argv))
        return None
    if tool_name in {"delegate_task", "delegate_worktree"}:
        return _bounded_line(arguments.get("task"))
    return None


def _bounded_line(value: object | None, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"
