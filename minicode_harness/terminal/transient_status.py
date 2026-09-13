"""Single-line transient status for interactive terminal runs."""

from __future__ import annotations

from threading import RLock
from typing import TextIO

from prompt_toolkit.utils import get_cwidth

from minicode_harness.terminal.types import TerminalRunState


class TransientStatusLine:
    """Own one in-place terminal line without retaining transcript output."""

    def __init__(
        self,
        stream: TextIO,
        *,
        enabled: bool | None = None,
    ) -> None:
        self.stream = stream
        self.enabled = bool(stream.isatty()) if enabled is None else enabled
        self._lock = RLock()
        self._desired_text: str | None = None
        self._visible = False
        self._rendered_width = 0
        self._suspended = False

    def show(self, text: str) -> None:
        with self._lock:
            self._desired_text = text
            if not self.enabled or self._suspended:
                return
            self._render(text)

    def update(self, text: str) -> None:
        self.show(text)

    def clear(self) -> None:
        with self._lock:
            self._desired_text = None
            self._clear_visible()

    def suspend(self) -> None:
        with self._lock:
            self._suspended = True
            self._clear_visible()

    def resume(self) -> None:
        with self._lock:
            self._suspended = False
            if self.enabled and self._desired_text:
                self._render(self._desired_text)

    def _render(self, text: str) -> None:
        width = _cell_width(text)
        padding = max(0, self._rendered_width - width)
        self.stream.write("\r" + text + (" " * padding))
        self.stream.flush()
        self._rendered_width = max(width, self._rendered_width)
        self._visible = True

    def _clear_visible(self) -> None:
        if not self.enabled or not self._visible:
            return
        self.stream.write("\r" + (" " * self._rendered_width) + "\r")
        self.stream.flush()
        self._visible = False
        self._rendered_width = 0

def format_transient_status(
    state: TerminalRunState,
    *,
    width: int,
) -> str:
    """Render one compact status line only when runtime state changes."""

    activity = state.activity or "thinking"
    activity_text = _activity_text(activity, state.activity_target)
    remaining = state.context_remaining

    wide = f"· {activity_text}"
    if remaining is not None:
        wide += f" · ctx {_compact_tokens(remaining)}"

    medium = f"· {activity}"
    if remaining is not None:
        medium += f" · {_compact_tokens(remaining)} free"
    narrow = f"· {activity}"

    available = max(1, width)
    for candidate in (wide, medium, narrow):
        if _cell_width(candidate) <= available:
            return candidate
    return _truncate_cells(narrow, available)


def _activity_text(activity: str, target: str | None) -> str:
    if not target:
        return activity
    if activity == "searching":
        return f'{activity} "{target}"'
    return f"{activity} {target}"

def _compact_tokens(value: int) -> str:
    if value >= 1000:
        thousands = value / 1000
        return f"{thousands:.1f}k" if thousands < 10 else f"{thousands:.0f}k"
    return str(value)


def _cell_width(text: str) -> int:
    return sum(max(0, get_cwidth(char)) for char in text)


def _truncate_cells(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _cell_width(text) <= width:
        return text
    if width <= 3:
        return "." * width
    target = width - 3
    cells = 0
    visible: list[str] = []
    for char in text:
        char_width = max(0, get_cwidth(char))
        if cells + char_width > target:
            break
        visible.append(char)
        cells += char_width
    return "".join(visible) + "..."
