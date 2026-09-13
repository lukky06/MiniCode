from __future__ import annotations

from io import StringIO
import re

from rich.console import Console

from minicode_harness.terminal.markdown_delta import MarkdownDeltaRenderer


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_markdown_delta_renderer_preserves_raw_text_for_non_tty() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    renderer = MarkdownDeltaRenderer(console)
    text = "## Title\n- item\n**bold** and `code`"

    for char in text:
        renderer.feed(char)
    renderer.finalize()

    assert stream.getvalue() == text


def test_markdown_delta_renderer_emits_visible_content_in_same_delta() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="standard",
        width=120,
    )
    renderer = MarkdownDeltaRenderer(console)

    renderer.feed("## M")

    assert _plain(stream.getvalue()) == "M"
    assert "\x1b[" in stream.getvalue()

    renderer.feed("iniCode")

    assert _plain(stream.getvalue()) == "MiniCode"


def test_markdown_delta_renderer_streams_hierarchical_terminal_output() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="standard",
        width=120,
    )
    renderer = MarkdownDeltaRenderer(console)
    markdown = (
        "## MiniCode\n"
        "- first\n"
        "1. second\n"
        "> note\n"
        "**bold** and `code`\n"
        "```python\n"
        "print('ok')\n"
        "```\n"
    )

    for split in ("## ", "Mini", "Code\n- fi", "rst\n1. second\n> note\n", "**bo", "ld** and `co", "de`\n```python\nprint('ok')\n```\n"):
        renderer.feed(split)
    renderer.finalize()

    visible = _plain(stream.getvalue())
    assert visible == (
        "MiniCode\n"
        "• first\n"
        "1. second\n"
        "│ note\n"
        "bold and code\n"
        "  python\n"
        "  print('ok')\n"
    )
    assert "##" not in visible
    assert "**" not in visible
    assert "```" not in visible
    assert "\x1b[" in stream.getvalue()
    assert "\x1b[36m" not in stream.getvalue()
    assert "".join(("## ", "Mini", "Code\n- fi", "rst\n1. second\n> note\n", "**bo", "ld** and `co", "de`\n```python\nprint('ok')\n```\n")) == markdown


def test_markdown_delta_renderer_resets_incomplete_inline_styles_at_newline() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="standard",
        width=120,
    )
    renderer = MarkdownDeltaRenderer(console)

    renderer.feed("`unclosed code\nplain line\n**unclosed bold\nnext line")
    renderer.finalize()

    assert _plain(stream.getvalue()) == (
        "unclosed code\nplain line\nunclosed bold\nnext line"
    )
    assert "\x1b[36m" not in stream.getvalue()


def test_markdown_delta_renderer_renders_thematic_break_without_raw_markers() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="standard",
        width=60,
    )
    renderer = MarkdownDeltaRenderer(console)

    for char in "before\n---\nafter":
        renderer.feed(char)
    renderer.finalize()

    assert _plain(stream.getvalue()) == "before\n" + ("─" * 20) + "\nafter"


def test_markdown_delta_renderer_flushes_incomplete_markers_once() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="standard",
        width=120,
    )
    renderer = MarkdownDeltaRenderer(console)

    renderer.feed("plain *")
    renderer.finalize()
    renderer.finalize()

    assert _plain(stream.getvalue()) == "plain *"
