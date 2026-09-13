from io import StringIO

from rich.console import Console

from minicode_harness.terminal.markdown_stream import MarkdownStreamRenderer


def _renderer() -> tuple[MarkdownStreamRenderer, StringIO]:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=100,
    )
    return MarkdownStreamRenderer(console), stream


def test_markdown_stream_renders_headings_left_aligned() -> None:
    renderer, stream = _renderer()

    renderer.feed("### Architecture\n\n")

    assert stream.getvalue().startswith("Architecture")
    assert "###" not in stream.getvalue()


def test_markdown_stream_waits_for_complete_paragraph() -> None:
    renderer, stream = _renderer()

    renderer.feed("partial paragraph")

    assert stream.getvalue() == ""

    renderer.feed(" completed.\n\n")

    assert "partial paragraph completed." in stream.getvalue()


def test_markdown_stream_waits_for_closing_code_fence() -> None:
    renderer, stream = _renderer()

    renderer.feed("```python\nprint('x')\n")

    assert stream.getvalue() == ""

    renderer.feed("```\n")

    assert "print('x')" in stream.getvalue()
    assert "```" not in stream.getvalue()


def test_markdown_stream_finalize_renders_tail_once() -> None:
    renderer, stream = _renderer()

    renderer.feed("**tail**")
    renderer.finalize()
    renderer.finalize()

    assert stream.getvalue().count("tail") == 1
    assert "**" not in stream.getvalue()
