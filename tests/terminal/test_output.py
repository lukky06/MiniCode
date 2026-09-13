from io import StringIO

from rich.console import Console

from minicode_harness.output import ContextUsage
from minicode_harness.terminal.output import TerminalOutputSink
from minicode_harness.terminal.transient_status import TransientStatusLine
from minicode_harness.terminal.types import TerminalRunState


class _FlushCountingStream(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _sink(*, debug_trace: bool = False) -> tuple[TerminalOutputSink, StringIO]:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=120,
    )
    return TerminalOutputSink(console=console, debug_trace=debug_trace), stream


def test_terminal_output_hides_context_by_default() -> None:
    sink, stream = _sink()
    sink.context_built(
        usage=ContextUsage(
            build_duration_ms=77,
            token_estimate=2890,
            context_window=128000,
            prompt_budget=122000,
            reserved_output=6000,
        )
    )
    assert stream.getvalue() == ""


def test_terminal_output_shows_ascii_context_in_debug_mode() -> None:
    sink, stream = _sink(debug_trace=True)
    sink.context_built(
        usage=ContextUsage(
            build_duration_ms=77,
            token_estimate=2890,
            context_window=128000,
            prompt_budget=122000,
            reserved_output=6000,
        )
    )
    assert stream.getvalue() == "context 2.9k | 77ms | 119k free\n"


def test_terminal_output_updates_complete_context_state() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    state = TerminalRunState()
    sink = TerminalOutputSink(console=console, run_state=state)

    sink.context_built(
        usage=ContextUsage(
            build_duration_ms=12,
            token_estimate=33000,
            context_window=32000,
            prompt_budget=26000,
            reserved_output=6000,
        )
    )

    assert state.phase == "thinking"
    assert state.activity == "thinking"
    assert state.context_token_estimate == 33000
    assert state.context_window == 32000
    assert state.prompt_budget == 26000
    assert state.reserved_output == 6000
    assert state.context_remaining == 0


def test_terminal_output_maps_tool_activity_without_model_text() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    state = TerminalRunState()
    sink = TerminalOutputSink(console=console, run_state=state)

    sink.tool_call_started(
        step=1,
        tool_call_id="search_1",
        tool_name="search",
        arguments={"source": "workspace", "kind": "text", "query": "OrderConsumer"},
    )

    assert state.phase == "tool_running"
    assert state.activity == "searching"
    assert state.activity_target == "OrderConsumer"

    sink.tool_call_finished(
        step=1,
        tool_call_id="search_1",
        tool_name="search",
        status="ok",
    )

    assert state.phase == "thinking"
    assert state.activity == "thinking"
    assert state.activity_target is None


def test_terminal_output_starts_and_clears_transient_status() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    state = TerminalRunState(
        model_name="deepseek-reasoner",
        activity="preparing",
        started_at_monotonic=10.0,
    )
    transient = TransientStatusLine(stream, enabled=True)
    sink = TerminalOutputSink(
        console=console,
        run_state=state,
        transient_status=transient,
    )

    sink.begin_run()
    sink.finalize_answer()

    output = stream.getvalue()
    assert "· preparing" in output
    assert "\x1b" not in output
    assert output.endswith("\r")


def test_terminal_output_never_writes_tool_history() -> None:
    sink, stream = _sink()

    sink.tool_call_started(
        step=1,
        tool_call_id="call_1",
        tool_name="read",
        arguments={"source": "workspace", "target": "src/example.py", "unused_secret": "must-not-print"},
    )
    sink.tool_call_finished(
        step=1,
        tool_call_id="call_1",
        tool_name="read",
        status="command_failed",
        summary="Primary failure: assertion failed.",
        metadata={"path": "src/example.py", "returncode": 1},
    )
    sink.flush()

    assert stream.getvalue() == ""


def test_terminal_output_aggregates_parallel_tool_status_without_rotating_targets() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    state = TerminalRunState()
    sink = TerminalOutputSink(console=console, run_state=state)

    for index, path in enumerate(("a.py", "b.py", "c.py", "d.py"), start=1):
        sink.tool_call_started(
            step=1,
            tool_call_id=f"read_{index}",
            tool_name="read",
            arguments={"source": "workspace", "target": path},
        )

    assert state.activity == "reading"
    assert state.activity_target == "4 files"
    assert stream.getvalue() == ""

    for index in range(1, 5):
        sink.tool_call_finished(
            step=1,
            tool_call_id=f"read_{index}",
            tool_name="read",
            status="ok",
        )

    assert state.activity == "thinking"
    assert state.activity_target is None
    assert stream.getvalue() == ""


def test_terminal_output_uses_generic_status_for_mixed_parallel_tools() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    state = TerminalRunState()
    sink = TerminalOutputSink(console=console, run_state=state)

    sink.tool_call_started(
        step=1,
        tool_call_id="read_1",
        tool_name="read",
        arguments={"source": "workspace", "target": "a.py"},
    )
    sink.tool_call_started(
        step=1,
        tool_call_id="command_1",
        tool_name="run_command",
        arguments={"argv": ["python", "-m", "pytest", "-q"]},
    )

    assert state.activity == "running"
    assert state.activity_target == "2 tools"
    assert stream.getvalue() == ""


def test_terminal_output_flushes_each_model_delta() -> None:
    stream = _FlushCountingStream()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    sink = TerminalOutputSink(console=console)
    baseline = stream.flush_count

    sink.model_text_delta("delta")

    assert stream.getvalue() == "delta"
    assert stream.flush_count == baseline + 1


def test_terminal_output_writes_each_markdown_delta_immediately_and_verbatim() -> None:
    sink, stream = _sink()
    text = (
        "中文 English 😀\n"
        "**Bold** and `code`.\n"
        "- item\n"
        "```python\nprint('ok')\n```\n"
        "```unclosed"
    )

    sink.model_stream_started()
    for index, char in enumerate(text, start=1):
        sink.model_text_delta(char)
        assert stream.getvalue() == text[:index]

    sink.finalize_answer()

    assert stream.getvalue() == text
    assert "Answer\n" not in stream.getvalue()


def test_terminal_output_ensures_exactly_one_trailing_newline() -> None:
    sink, stream = _sink()

    sink.model_text_delta("first")
    sink.ensure_trailing_newline()
    sink.ensure_trailing_newline()

    assert stream.getvalue() == "first\n"

    sink.reset()
    sink.model_text_delta("second\n")
    sink.ensure_trailing_newline()

    assert stream.getvalue() == "first\nsecond\n"


def test_terminal_output_reset_does_not_replay_prior_text() -> None:
    sink, stream = _sink()

    sink.model_text_delta("first")
    sink.finalize_answer()
    sink.reset()
    sink.model_text_delta("second")
    sink.finalize_answer()

    assert stream.getvalue() == "firstsecond"
