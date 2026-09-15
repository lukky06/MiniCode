import pytest
import typer

from minicode_harness import cli
from minicode_harness.output import TextOutputSink
from minicode_harness.state import NonInteractiveApprovalClient
from minicode_harness.terminal import (
    TerminalApprovalClient,
    TerminalOutputSink,
)


class FakeTTY:
    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


class FakeNonTTY(FakeTTY):
    def isatty(self) -> bool:
        return False


def test_no_argument_main_starts_terminal_when_stdin_and_stdout_are_tty(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINICODE_PROVIDER", raising=False)
    monkeypatch.delenv("MINICODE_MODEL", raising=False)
    monkeypatch.setattr(cli, "launch_tui", lambda **kwargs: calls.update(kwargs=kwargs))
    monkeypatch.setattr(cli.sys, "argv", ["minicode", "--sandbox", "local"])
    monkeypatch.setattr(cli.sys, "stdin", FakeTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeTTY())

    cli.main()

    assert calls["kwargs"]["workspace"] == tmp_path
    assert calls["kwargs"]["provider"] == "qwen"
    assert calls["kwargs"]["session_mode"] == "new"
    assert calls["kwargs"]["session_id"] is None
    assert calls["kwargs"]["permission_mode"] == "read-only"
    assert calls["kwargs"]["approval_policy"] == "on-request"
    assert calls["kwargs"]["collaboration_mode"] == "default"
    assert calls["kwargs"]["initial_task"] is None


def test_continue_main_starts_latest_terminal_session(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "launch_tui", lambda **kwargs: calls.update(kwargs=kwargs))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["minicode", "--continue", "--sandbox", "local"],
    )
    monkeypatch.setattr(cli.sys, "stdin", FakeTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeTTY())

    cli.main()

    assert calls["kwargs"]["session_mode"] == "continue"
    assert calls["kwargs"]["session_id"] is None


def test_resume_command_opens_exact_conversation_session(tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "launch_tui", lambda **kwargs: calls.update(kwargs=kwargs))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "minicode",
            "resume",
            "session_abcdef123456",
            "--no-write",
            "--sandbox",
            "local",
        ],
    )
    monkeypatch.setattr(cli.sys, "stdin", FakeTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeTTY())

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert calls["kwargs"]["session_mode"] == "exact"
    assert calls["kwargs"]["session_id"] == "session_abcdef123456"
    assert calls["kwargs"]["write_enabled"] is False
    assert calls["kwargs"]["initial_task"] is None


def test_bare_task_main_starts_interactive_session_with_options(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {}
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(cli, "launch_tui", lambda **kwargs: calls.update(kwargs=kwargs))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "minicode",
            "--provider",
            "deepseek",
            "explain",
            "this repo",
            "--workspace",
            str(workspace),
            "--no-write",
            "--permission-mode",
            "workspace-write",
            "--approval-policy",
            "never",
            "--mode",
            "plan",
            "--no-subagents",
            "--sandbox",
            "local",
        ],
    )
    monkeypatch.setattr(cli.sys, "stdin", FakeTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeTTY())

    cli.main()

    assert calls["kwargs"]["initial_task"] == "explain this repo"
    assert calls["kwargs"]["workspace"] == workspace
    assert calls["kwargs"]["provider"] == "deepseek"
    assert calls["kwargs"]["write_enabled"] is False
    assert calls["kwargs"]["permission_mode"] == "workspace-write"
    assert calls["kwargs"]["approval_policy"] == "never"
    assert calls["kwargs"]["collaboration_mode"] == "plan"
    assert calls["kwargs"]["subagents_enabled"] is False
    assert calls["kwargs"]["session_mode"] == "new"


def test_bare_task_main_rejects_non_tty(tmp_path, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys, "argv", ["minicode", "explain", "repo"])
    monkeypatch.setattr(cli.sys, "stdin", FakeNonTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeNonTTY())
    monkeypatch.setattr(
        cli.typer,
        "echo",
        lambda message="", **kwargs: messages.append(str(message)),
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.main()

    assert exc_info.value.exit_code == 2
    assert messages == [
        "Interactive MiniCode requires a TTY.",
        'Use `minicode exec "<task>"` for non-interactive execution.',
    ]


def test_execution_io_uses_plain_sink_and_rejecting_approval_for_non_tty(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", FakeNonTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeNonTTY())

    sink, approval, renderer = cli._execution_io(plain=False, no_color=False)

    assert isinstance(sink, TextOutputSink)
    assert isinstance(approval, NonInteractiveApprovalClient)
    assert renderer is None


def test_execution_io_uses_terminal_components_for_tty(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", FakeTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeTTY())

    sink, approval, renderer = cli._execution_io(plain=False, no_color=True)

    assert isinstance(sink, TerminalOutputSink)
    assert isinstance(approval, TerminalApprovalClient)
    assert renderer is not None
