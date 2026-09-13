"""Incremental Markdown styling without block buffering or history replay."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text


_HEADING_RE = re.compile(r"^(#{1,6}) $")
_ORDERED_RE = re.compile(r"^(\d{1,4}[.)]) $")
_ORDERED_WAIT_RE = re.compile(r"^\d{1,4}[.)]?$")
_THEMATIC_LINE_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})\n$")


@dataclass(frozen=True)
class _PrefixDecision:
    kind: str
    indent: str = ""
    marker: str = ""


class MarkdownDeltaRenderer:
    """Style visible Markdown as provider deltas arrive.

    Only incomplete Markdown control markers are retained. Visible prose and code
    are emitted in the same ``feed`` call, so the renderer never waits for a
    paragraph, newline, or closing fence and never redraws prior output.
    """

    def __init__(self, console: Console, *, enabled: bool | None = None) -> None:
        self.console = console
        self.enabled = console.is_terminal if enabled is None else enabled
        self.reset()

    def reset(self) -> None:
        self._at_line_start = True
        self._prefix = ""
        self._line_style: str | None = None
        self._bold = False
        self._inline_code = False
        self._pending_star = False
        self._code_block = False
        self._code_line_start = False
        self._fence_header = False
        self._fence_closing = False
        self._fence_indent = ""
        self._fence_text = ""

    def feed(self, text: str) -> None:
        if not text:
            return
        if not self.enabled:
            self._write_plain(text)
            return

        rendered = Text()
        for char in text:
            self._consume(char, rendered)
        self._emit(rendered)

    def finalize(self) -> None:
        """Flush incomplete syntax markers without replaying prior content."""

        if not self.enabled:
            return
        rendered = Text()
        if self._fence_header:
            literal = self._fence_indent + "```" + self._fence_text
            self._append(rendered, literal, self._active_style())
            self._fence_header = False
            self._fence_text = ""
        if self._prefix:
            self._append(rendered, self._prefix, self._active_style())
            self._prefix = ""
        if self._pending_star:
            self._append(rendered, "*", self._active_style())
            self._pending_star = False
        self._emit(rendered)

    def _consume(self, char: str, rendered: Text) -> None:
        if self._fence_header:
            if char == "\n":
                self._emit_fence_line(rendered)
                self._reset_line()
            else:
                self._fence_text += char
            return

        if self._at_line_start:
            self._prefix += char
            decision = self._classify_prefix(self._prefix)
            if decision.kind == "wait":
                return
            if decision.kind == "fence":
                self._fence_header = True
                self._fence_closing = self._code_block
                self._fence_indent = decision.indent
                self._fence_text = ""
                self._prefix = ""
                self._at_line_start = False
                return
            if decision.kind == "heading":
                level = len(decision.marker)
                self._line_style = _heading_style(level)
                self._prefix = ""
                self._at_line_start = False
                return
            if decision.kind == "bullet":
                self._append(rendered, decision.indent + "• ", "dim")
                self._prefix = ""
                self._at_line_start = False
                return
            if decision.kind == "ordered":
                self._append(rendered, decision.indent + decision.marker + " ", "dim")
                self._prefix = ""
                self._at_line_start = False
                return
            if decision.kind == "quote":
                self._append(rendered, decision.indent + "│ ", "dim")
                self._line_style = "italic"
                self._prefix = ""
                self._at_line_start = False
                return
            if decision.kind == "rule":
                self._append(
                    rendered,
                    "─" * min(24, max(8, self.console.width // 3)) + "\n",
                    "dim",
                )
                self._prefix = ""
                self._reset_line()
                return

            literal = self._prefix
            self._prefix = ""
            self._at_line_start = False
            self._consume_inline(literal, rendered)
            return

        self._consume_inline(char, rendered)

    def _consume_inline(self, text: str, rendered: Text) -> None:
        for char in text:
            if char == "\n":
                if self._pending_star:
                    self._append(rendered, "*", self._active_style())
                    self._pending_star = False
                self._append(rendered, "\n", self._active_style())
                self._reset_line()
                continue

            if self._code_block:
                if self._code_line_start:
                    self._append(rendered, "  ", "dim")
                    self._code_line_start = False
                self._append(rendered, char, self._active_style())
                continue

            if self._pending_star:
                if char == "*":
                    self._bold = not self._bold
                    self._pending_star = False
                    continue
                self._append(rendered, "*", self._active_style())
                self._pending_star = False

            if char == "*":
                self._pending_star = True
                continue
            if char == "`":
                self._inline_code = not self._inline_code
                continue
            self._append(rendered, char, self._active_style())

    def _classify_prefix(self, prefix: str) -> _PrefixDecision:
        indent_length = len(prefix) - len(prefix.lstrip(" "))
        indent = prefix[:indent_length]
        rest = prefix[indent_length:]
        if not rest:
            return _PrefixDecision("wait" if indent_length <= 8 else "plain")

        if self._code_block:
            if rest in {"`", "``"}:
                return _PrefixDecision("wait")
            if rest == "```":
                return _PrefixDecision("fence", indent=indent)
            return _PrefixDecision("plain")

        if _THEMATIC_LINE_RE.match(rest):
            return _PrefixDecision("rule", indent=indent)
        if prefix.endswith("\n"):
            return _PrefixDecision("plain")

        if rest in {"`", "``"}:
            return _PrefixDecision("wait")
        if rest == "```":
            return _PrefixDecision("fence", indent=indent)

        if 1 <= len(rest) <= 6 and set(rest) == {"#"}:
            return _PrefixDecision("wait")
        heading = _HEADING_RE.match(rest)
        if heading:
            return _PrefixDecision("heading", indent=indent, marker=heading.group(1))

        if rest in {"-", "*", "+", "--", "---", "**", "***", "_", "__", "___"}:
            return _PrefixDecision("wait")
        if rest in {"- ", "* ", "+ "}:
            return _PrefixDecision("bullet", indent=indent, marker=rest[0])

        if rest == ">":
            return _PrefixDecision("wait")
        if rest == "> ":
            return _PrefixDecision("quote", indent=indent)

        if _ORDERED_WAIT_RE.match(rest):
            return _PrefixDecision("wait")
        ordered = _ORDERED_RE.match(rest)
        if ordered:
            return _PrefixDecision("ordered", indent=indent, marker=ordered.group(1))

        return _PrefixDecision("plain")

    def _emit_fence_line(self, rendered: Text) -> None:
        label = self._fence_text.strip()
        if self._fence_closing:
            self._code_block = False
            self._code_line_start = False
        else:
            if label:
                self._append(rendered, self._fence_indent + "  " + label + "\n", "dim")
            self._code_block = True
            self._code_line_start = True
        self._fence_header = False
        self._fence_closing = False
        self._fence_indent = ""
        self._fence_text = ""

    def _reset_line(self) -> None:
        self._at_line_start = True
        self._prefix = ""
        self._line_style = None
        self._bold = False
        self._inline_code = False
        self._code_line_start = self._code_block

    def _active_style(self) -> str | None:
        styles: list[str] = []
        if self._line_style:
            styles.extend(self._line_style.split())
        if self._code_block:
            styles.append("dim")
        elif self._inline_code:
            styles.append("bold")
        if self._bold:
            styles.append("bold")
        unique = list(dict.fromkeys(styles))
        return " ".join(unique) or None

    def _emit(self, rendered: Text) -> None:
        if not rendered:
            return
        self.console.print(rendered, end="", soft_wrap=True)
        self.console.file.flush()

    def _write_plain(self, text: str) -> None:
        stream = self.console.file
        encoding = getattr(stream, "encoding", None)
        if encoding:
            text = text.encode(encoding, errors="replace").decode(encoding)
        stream.write(text)
        stream.flush()

    @staticmethod
    def _append(rendered: Text, text: str, style: str | None) -> None:
        if text:
            rendered.append(text, style=style)


def _heading_style(level: int) -> str:
    if level == 1:
        return "bold underline"
    return "bold"
