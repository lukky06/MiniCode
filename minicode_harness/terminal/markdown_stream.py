"""Block-aware streaming Markdown rendering for the scrolling terminal."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Heading, Markdown


class _TerminalHeading(Heading):
    """Rich Markdown heading rendered left-aligned without an oversized panel."""

    def __rich_console__(self, console, options):
        self.text.justify = "left"
        yield self.text


class _TerminalMarkdown(Markdown):
    elements = {
        **Markdown.elements,
        "heading_open": _TerminalHeading,
    }


class MarkdownStreamRenderer:
    """Render only complete Markdown blocks and never replay prior output."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.buffer = ""

    def feed(self, text: str) -> None:
        if not text:
            return
        self.buffer += text
        self._render_complete_blocks()

    def finalize(self) -> None:
        """Render the remaining incomplete tail at the final-answer boundary."""

        block = self.buffer.strip("\n")
        self.buffer = ""
        if block:
            self._render(block)

    def reset(self) -> None:
        self.buffer = ""

    def _render_complete_blocks(self) -> None:
        while True:
            boundary = _next_markdown_boundary(self.buffer)
            if boundary is None:
                return
            block = self.buffer[:boundary].strip("\n")
            self.buffer = self.buffer[boundary:]
            if block:
                self._render(block)

    def _render(self, block: str) -> None:
        self.console.print(_TerminalMarkdown(block))


def _next_markdown_boundary(text: str) -> int | None:
    """Return the end of the next complete paragraph or fenced code block."""

    position = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = _fence_marker(stripped)
        if marker is not None:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
                position += len(line)
                if line.endswith(("\n", "\r")):
                    return position
                continue

        position += len(line)
        if fence is None and not line.strip() and line.endswith(("\n", "\r")):
            return position
    return None


def _fence_marker(line: str) -> str | None:
    if line.startswith("```"):
        return "```"
    if line.startswith("~~~"):
        return "~~~"
    return None
