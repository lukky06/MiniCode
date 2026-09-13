from io import StringIO

from minicode_harness.terminal.transient_status import (
    TransientStatusLine,
    format_transient_status,
)
from minicode_harness.terminal.types import TerminalRunState


def test_transient_status_updates_one_line_and_erases_shorter_tail() -> None:
    stream = StringIO()
    status = TransientStatusLine(stream, enabled=True)

    status.show("reading very-long-file.py")
    status.update("thinking")

    output = stream.getvalue()
    assert "\x1b" not in output
    assert output.startswith("\rreading very-long-file.py")
    assert output.endswith("\rthinking" + (" " * 17))


def test_transient_status_clear_removes_visible_line() -> None:
    stream = StringIO()
    status = TransientStatusLine(stream, enabled=True)

    status.show("⠋ thinking")
    status.clear()

    output = stream.getvalue()
    assert "\x1b" not in output
    assert output.endswith("\r")
    assert " " * 10 in output


def test_transient_status_is_silent_when_disabled() -> None:
    stream = StringIO()
    status = TransientStatusLine(stream, enabled=False)

    status.show("thinking")
    status.update("reading file.py")
    status.clear()

    assert stream.getvalue() == ""


def test_transient_status_suspend_and_resume_preserve_latest_text() -> None:
    stream = StringIO()
    status = TransientStatusLine(stream, enabled=True)

    status.show("thinking")
    status.suspend()
    status.update("reading file.py")
    status.resume()

    output = stream.getvalue()
    assert "\x1b" not in output
    assert output.endswith("\rreading file.py")
    assert stream.getvalue().count("reading file.py") == 1


def test_transient_status_shows_compact_activity_and_context_remaining() -> None:
    state = TerminalRunState(
        activity="reading",
        activity_target="OrderService.java",
        context_remaining=64000,
    )

    rendered = format_transient_status(state, width=120)

    assert rendered == "· reading OrderService.java · ctx 64k"


def test_transient_status_degrades_for_medium_and_narrow_terminals() -> None:
    state = TerminalRunState(
        activity="searching",
        activity_target="OrderConsumer",
        context_remaining=71000,
    )

    medium = format_transient_status(state, width=28)
    narrow = format_transient_status(state, width=12)

    assert medium == "· searching · 71k free"
    assert narrow == "· searching"
