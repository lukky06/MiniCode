from io import StringIO

from pathlib import Path

from rich.console import Console

from minicode_harness.runtime.run_executor import RunExecutionResult
from minicode_harness.terminal.rendering import TerminalRenderer
from minicode_harness.terminal.status import TerminalStatus


class _ReconfigurableTTY(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.reconfigured: tuple[str, str] | None = None

    def isatty(self) -> bool:
        return True

    def reconfigure(self, *, encoding: str, errors: str) -> None:
        self.reconfigured = (encoding, errors)


def test_terminal_renderer_configures_interactive_stream_as_utf8() -> None:
    stream = _ReconfigurableTTY()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=80,
    )

    TerminalRenderer(console)

    assert stream.reconfigured == ("utf-8", "replace")


def test_welcome_uses_two_compact_lines_without_panel() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=64,
    )
    renderer = TerminalRenderer(console)
    status = TerminalStatus(
        workspace="D:/very/long/path/to/a/project/that/should/be/compacted/rag_agent",
        project="Python / pyproject",
        state="ready",
        model="qwen / qwen-plus",
        session="session_abcdef123456",
        latest_run="none",
    )

    renderer.show_welcome(status=status, write_enabled=True)

    assert stream.getvalue() == (
        "MiniCode · rag_agent · qwen-plus · write\n"
        "Session abcdef · Python / pyproject\n\n"
    )


def test_run_summary_renders_deterministic_budget_stop_summary() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=100,
    )
    renderer = TerminalRenderer(console)
    result = RunExecutionResult(
        run_id="run_1",
        run_path=Path("runs/run_1"),
        status="stopped",
        stop_reason="max_steps",
        stop_summary=(
            "Run stopped because the model-call budget was exhausted.\n"
            "Changes retained: src/example.py."
        ),
        modified_files=["src/example.py"],
        verification_status="not_run",
    )

    renderer.show_run_summary(result)

    output = stream.getvalue()
    assert "Stopped" in output
    assert "1 changed" in output
    assert "max_steps" in output
    assert "model-call budget was exhausted" in output
    assert "Changes retained: src/example.py" in output


def test_run_summary_shows_memory_publication_and_pending_review() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=100,
    )
    renderer = TerminalRenderer(console)
    result = RunExecutionResult(
        run_id="run_memory",
        run_path=Path("runs/run_memory"),
        status="completed",
        memory_review_status="completed",
        memory_auto_published_count=2,
        memory_pending_candidates=1,
    )

    renderer.show_run_summary(result)

    output = stream.getvalue()
    assert "memory +2" in output
    assert "memory 1 pending" in output
