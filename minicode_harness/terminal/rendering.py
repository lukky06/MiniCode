"""Rich rendering helpers for the scrolling terminal."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from minicode_harness.runtime.run_executor import RunExecutionResult
from minicode_harness.terminal.status import TerminalStatus
from minicode_harness.terminal.types import CommandResult


class TerminalRenderer:
    """Render terminal-only presentation without affecting runtime state."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        _configure_utf8_terminal_stream(self.console.file)

    def show_welcome(
        self,
        *,
        status: TerminalStatus,
        write_enabled: bool,
    ) -> None:
        permission = "write" if write_enabled else "read-only"
        project_name = Path(status.workspace).name or status.project
        first = (
            f"MiniCode · {project_name} · "
            f"{_welcome_model_name(status.model)} · {permission}"
        )
        second = f"Session {_welcome_session(status.session)} · {status.project}"
        self.console.print(
            _truncate_visible(first, self.console.width),
            markup=False,
            highlight=False,
        )
        self.console.print(
            _truncate_visible(second, self.console.width),
            style="dim",
            markup=False,
            highlight=False,
        )
        self.console.print()

    def show_command_result(self, result: CommandResult) -> None:
        if result.content:
            self.console.print(result.content, markup=False, highlight=False)

    def show_error(self, message: str) -> None:
        self.console.print(f"Error: {message}", markup=False, highlight=False, style="bold red")

    def show_goodbye(self) -> None:
        self.console.print("Exited MiniCode.")

    def show_cancel_requested(self) -> None:
        self.console.print(
            "Cancellation requested; stopping at the next safe boundary.",
            style="yellow",
            markup=False,
            highlight=False,
        )

    def show_run_summary(self, result: RunExecutionResult) -> None:
        if result.status == "completed":
            label = "Completed"
        elif result.status == "cancelled":
            label = "Cancelled"
        else:
            label = "Stopped"
        parts = [label]
        if result.inspected_files:
            parts.append(f"{result.inspected_files} inspected")
        if result.modified_files:
            parts.append(f"{len(result.modified_files)} changed")
        else:
            parts.append("no changes")
        verification = (result.verification_status or "").strip()
        if verification and verification != "not_run":
            parts.append(f"verification {verification}")
        elif result.stop_reason and result.status != "completed":
            parts.append(result.stop_reason)
        self.console.print(Rule(" · ".join(parts)))
        if result.stop_summary:
            self.console.print(
                result.stop_summary,
                markup=False,
                highlight=False,
                style="yellow",
            )

    def ensure_trailing_newline(self) -> None:
        self.console.print()

    @property
    def output_stream(self):
        return self.console.file


def _configure_utf8_terminal_stream(stream) -> None:
    """Keep interactive Windows terminal glyphs intact."""

    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass


def _welcome_model_name(model: str) -> str:
    return model.split(" / ", 1)[-1].strip() or "model"


def _welcome_session(session: str) -> str:
    compact = session.removeprefix("session_")
    return compact[:6] if compact else "none"


def _visible_width(text: str) -> int:
    return sum(max(0, get_cwidth(char)) for char in text)


def _truncate_visible(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _visible_width(text) <= width:
        return text
    if width <= 3:
        return "." * width
    target = width - 3
    visible: list[str] = []
    cells = 0
    for char in text:
        char_width = max(0, get_cwidth(char))
        if cells + char_width > target:
            break
        visible.append(char)
        cells += char_width
    return "".join(visible) + "..."


class DiffRenderer:
    """Render bounded unified diffs for approval previews."""

    def __init__(self, console: Console | None = None, *, max_lines: int = 80) -> None:
        self.console = console or Console()
        self.max_lines = max(1, max_lines)

    def render(self, diff: str, *, full: bool = False) -> None:
        lines = diff.splitlines()
        visible = lines
        truncated = False
        if not full and len(lines) > self.max_lines:
            head = self.max_lines // 2
            tail = self.max_lines - head
            visible = [
                *lines[:head],
                "... <diff preview truncated> ...",
                *lines[-tail:],
            ]
            truncated = True
        text = "\n".join(visible)
        self.console.print(
            Panel(
                Syntax(text, "diff", word_wrap=False),
                title="Diff" + (" · truncated" if truncated else ""),
                expand=False,
            )
        )
