"""Compact Rich output sink for interactive minicode exec runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console

from minicode_harness.output import ContextUsage
from minicode_harness.policy import render_argv
from minicode_harness.tools.semantics import read_source, read_target, search_kind
from minicode_harness.terminal.markdown_delta import MarkdownDeltaRenderer
from minicode_harness.terminal.transient_status import (
    TransientStatusLine,
    format_transient_status,
)
from minicode_harness.terminal.types import TerminalRunState


@dataclass(frozen=True)
class _PendingToolCall:
    step: int
    tool_name: str
    arguments: dict[str, Any]


class TerminalOutputSink:
    """Render compact status and streamed Markdown for the Python exec path."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        debug_trace: bool = False,
        run_state: TerminalRunState | None = None,
        transient_status: TransientStatusLine | None = None,
    ) -> None:
        self.console = console or Console()
        self.debug_trace = debug_trace
        self.run_state = run_state
        self.transient_status = transient_status or TransientStatusLine(
            self.console.file,
            enabled=self.console.is_terminal,
        )
        self._pending: dict[str, _PendingToolCall] = {}
        self._markdown = MarkdownDeltaRenderer(self.console)
        self._answer_started = False
        self._last_model_char: str | None = None

    def reset(self) -> None:
        """Reset presentation state before a new Run without changing runtime data."""
        self.transient_status.clear()
        self._pending.clear()
        self._markdown.reset()
        self._answer_started = False
        self._last_model_char = None

    def begin_run(self) -> None:
        self._refresh_status()

    def suspend_status(self) -> None:
        self.transient_status.suspend()

    def resume_status(self) -> None:
        if not self._answer_started:
            self.transient_status.resume()

    def context_built(self, *, usage: ContextUsage) -> None:
        if self.run_state is not None:
            self.run_state.phase = "thinking"
            self.run_state.activity = "thinking"
            self.run_state.activity_target = None
            self.run_state.context_token_estimate = usage.token_estimate
            self.run_state.context_window = usage.context_window
            self.run_state.prompt_budget = usage.prompt_budget
            self.run_state.reserved_output = usage.reserved_output
            self.run_state.context_remaining = usage.context_remaining
        self._refresh_status()
        if not self.debug_trace:
            return
        token_text = _compact_token_count(usage.token_estimate)
        remaining_text = _compact_token_count(usage.context_remaining)
        self.suspend_status()
        try:
            self.console.print(
                f"context {token_text} | {usage.build_duration_ms}ms | {remaining_text} free",
                style="dim",
                markup=False,
                highlight=False,
            )
        finally:
            self.resume_status()

    def model_stream_started(self) -> None:
        if self.run_state is not None:
            self.run_state.phase = "thinking"
            self.run_state.activity = "thinking"
            self.run_state.activity_target = None
        self._refresh_status()

    def model_reasoning_delta(self, text: str) -> None:
        del text

    def model_text_delta(self, text: str) -> None:
        if not text:
            return
        if not self._answer_started:
            self._answer_started = True
            if self.run_state is not None:
                self.run_state.phase = "answering"
                self.run_state.activity = "answering"
                self.run_state.activity_target = None
            self.transient_status.clear()
        self._write_model_text(text)

    def finalize_answer(self) -> None:
        self.transient_status.clear()
        self._markdown.finalize()

    def ensure_trailing_newline(self) -> None:
        if not self._answer_started or self._last_model_char == "\n":
            return
        self._write_model_text("\n")

    def tool_call_started(
        self,
        *,
        step: int,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
    ) -> None:
        self._pending[tool_call_id] = _PendingToolCall(
            step=step,
            tool_name=tool_name,
            arguments=dict(arguments),
        )
        if self.run_state is not None:
            self.run_state.tool_count += 1
        self._update_tool_status()
        self._refresh_status()

    def tool_call_finished(
        self,
        *,
        step: int,
        tool_name: str,
        status: str,
        tool_call_id: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del step, tool_name, status, summary, metadata
        self._pending.pop(tool_call_id, None)
        self._update_tool_status()
        self._refresh_status()

    def flush(self) -> None:
        """Retained for the OutputSink lifecycle contract."""

    def _update_tool_status(self) -> None:
        if self.run_state is None:
            return
        pending = list(self._pending.values())
        if not pending:
            self.run_state.active_tool = None
            self.run_state.phase = "thinking"
            self.run_state.activity = "thinking"
            self.run_state.activity_target = None
            return

        activities = [
            _tool_activity(call.tool_name, call.arguments)
            for call in pending
        ]
        self.run_state.phase = "tool_running"
        if len(activities) == 1:
            activity, target = activities[0]
            self.run_state.active_tool = pending[0].tool_name
            self.run_state.activity = activity
            self.run_state.activity_target = target
            return

        activity_names = {activity for activity, _ in activities}
        if len(activity_names) == 1:
            activity = activities[0][0]
            self.run_state.active_tool = (
                pending[0].tool_name
                if len({call.tool_name for call in pending}) == 1
                else None
            )
            self.run_state.activity = activity
            self.run_state.activity_target = _aggregate_activity_target(
                activity,
                len(activities),
            )
            return

        self.run_state.active_tool = None
        self.run_state.activity = "running"
        self.run_state.activity_target = f"{len(activities)} tools"

    def _write_model_text(self, text: str) -> None:
        self._markdown.feed(text)
        self._last_model_char = text[-1]

    def _refresh_status(self) -> None:
        if (
            self.run_state is None
            or not self.transient_status.enabled
            or self._answer_started
        ):
            return
        text = format_transient_status(
            self.run_state,
            width=self.console.width,
        )
        self.transient_status.update(text)


def _compact_token_count(value: int) -> str:
    if value >= 1000:
        rounded = value / 1000
        return f"{rounded:.1f}k" if rounded < 10 else f"{rounded:.0f}k"
    return str(value)


def _tool_activity(
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, str | None]:
    source = read_source(tool_name, arguments)
    if source is not None:
        target = read_target(tool_name, arguments)
        return "reading", _bounded_target(target)

    kind = search_kind(tool_name, arguments)
    if kind == "text":
        return "searching", _bounded_target(arguments.get("query"))
    if kind == "files":
        return "searching", _bounded_target(
            arguments.get("query") or arguments.get("path")
        )

    path = arguments.get("path")
    if tool_name in {"edit", "write"}:
        return "editing", _bounded_target(path)
    if tool_name == "apply_patch":
        return "editing", "files"
    if tool_name == "run_command":
        argv = arguments.get("argv")
        command = (
            render_argv(argv)
            if isinstance(argv, list) and all(isinstance(item, str) for item in argv)
            else None
        )
        return "running", _bounded_target(command)
    if tool_name in {"delegate_task", "delegate_worktree"}:
        return "delegating", _bounded_target(
            arguments.get("task") or arguments.get("subject")
        )
    return "running", _bounded_target(tool_name)


def _bounded_target(value: object, *, limit: int = 80) -> str | None:
    text = str(value or "").strip().replace("\n", " ")
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _aggregate_activity_target(activity: str, count: int) -> str:
    noun = {
        "reading": "files",
        "searching": "queries",
        "editing": "files",
        "running": "commands",
        "delegating": "tasks",
    }.get(activity, "tools")
    return f"{count} {noun}"