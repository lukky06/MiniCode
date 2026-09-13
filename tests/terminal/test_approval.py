from io import StringIO

import pytest
from rich.console import Console

from minicode_harness.state import ApprovalDecision, ApprovalRequest
from minicode_harness.terminal.approval import TerminalApprovalClient
from minicode_harness.terminal.rendering import DiffRenderer
from minicode_harness.terminal.transient_status import TransientStatusLine
from minicode_harness.terminal.types import TerminalRunState


def _console() -> tuple[Console, StringIO]:
    stream = StringIO()
    return (
        Console(file=stream, force_terminal=False, color_system=None, width=120),
        stream,
    )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("y", ApprovalDecision.APPROVE),
        ("n", ApprovalDecision.REJECT),
        ("s", ApprovalDecision.SKIP),
        ("a", ApprovalDecision.ABORT),
    ],
)
def test_terminal_approval_preserves_runtime_decision_semantics(answer, expected) -> None:
    console, _ = _console()
    client = TerminalApprovalClient(
        console,
        answer_reader=lambda prompt: answer,
        interactive=True,
    )
    request = ApprovalRequest(
        id="approval_1",
        tool_call_id="call_1",
        tool_name="run_command",
        risk_level="high",
        arguments={"command": "python -m pytest tests/unit -q"},
        preview={
            "command": "python -m pytest tests/unit -q",
            "workspace": "C:/repo",
            "timeout_seconds": 120,
        },
    )

    assert client.decide(request).decision == expected


def test_terminal_approval_can_grant_command_for_session() -> None:
    console, _ = _console()
    client = TerminalApprovalClient(
        console,
        answer_reader=lambda prompt: "g",
        interactive=True,
    )
    request = ApprovalRequest(
        id="approval_session",
        tool_call_id="call_session",
        tool_name="run_command",
        risk_level="high",
        can_approve_session=True,
        preview={"command": "python script.py"},
    )

    assert client.decide(request).decision == ApprovalDecision.APPROVE_SESSION


def test_terminal_approval_suspends_and_restores_transient_status() -> None:
    console, stream = _console()
    state = TerminalRunState(
        phase="tool_running",
        activity="running",
        activity_target="pytest -q",
    )
    transient = TransientStatusLine(stream, enabled=True)
    transient.show("· running pytest -q")
    observed_phases: list[str] = []

    def answer_reader(prompt: str) -> str:
        observed_phases.append(state.phase)
        assert state.activity == "waiting approval"
        return "y"

    client = TerminalApprovalClient(
        console,
        answer_reader=answer_reader,
        interactive=True,
        run_state=state,
        transient_status=transient,
    )
    request = ApprovalRequest(
        id="approval_status",
        tool_call_id="call_status",
        tool_name="run_command",
        risk_level="high",
        arguments={"command": "python -m pytest -q"},
        preview={"command": "python -m pytest -q"},
    )

    response = client.decide(request)

    assert response.decision == ApprovalDecision.APPROVE
    assert observed_phases == ["approval"]
    assert state.phase == "tool_running"
    assert state.activity == "running"
    assert state.activity_target == "pytest -q"
    assert "\x1b" not in stream.getvalue()
    assert stream.getvalue().endswith("\r· running pytest -q")


def test_terminal_approval_does_not_restore_cleared_answer_status() -> None:
    console, stream = _console()
    state = TerminalRunState(phase="answering", activity="answering")
    transient = TransientStatusLine(stream, enabled=True)
    transient.show("answering")
    transient.clear()
    client = TerminalApprovalClient(
        console,
        answer_reader=lambda prompt: "y",
        interactive=True,
        run_state=state,
        transient_status=transient,
    )
    request = ApprovalRequest(
        id="approval_after_answer",
        tool_call_id="call_after_answer",
        tool_name="run_command",
        risk_level="high",
        arguments={"command": "python -m pytest -q"},
        preview={"command": "python -m pytest -q"},
    )

    response = client.decide(request)

    assert response.decision == ApprovalDecision.APPROVE
    assert state.phase == "answering"
    assert stream.getvalue().count("answering") == 1


def test_terminal_approval_non_tty_rejects_without_reading_input() -> None:
    console, _ = _console()
    client = TerminalApprovalClient(
        console,
        answer_reader=lambda prompt: (_ for _ in ()).throw(
            AssertionError("must not read")
        ),
        interactive=False,
    )
    request = ApprovalRequest(
        id="approval_2",
        tool_call_id="call_2",
        tool_name="write",
        risk_level="medium",
        arguments={"path": "a.py"},
        preview={"path": "a.py", "bytes": 10},
    )

    response = client.decide(request)

    assert response.decision == ApprovalDecision.REJECT
    assert "interactive terminal" in (response.reason or "")


def test_terminal_approval_view_details_renders_bounded_diff_then_approves() -> None:
    console, stream = _console()
    answers = iter(["v", "y"])
    client = TerminalApprovalClient(
        console,
        answer_reader=lambda prompt: next(answers),
        interactive=True,
        diff_renderer=DiffRenderer(console, max_lines=4),
    )
    request = ApprovalRequest(
        id="approval_3",
        tool_call_id="call_3",
        tool_name="edit",
        risk_level="medium",
        arguments={"path": "a.py"},
        preview={
            "path": "a.py",
            "old_chars": 3,
            "new_chars": 4,
            "diff": "\n".join(f"+line {index}" for index in range(10)),
        },
    )

    response = client.decide(request)

    assert response.decision == ApprovalDecision.APPROVE
    output = stream.getvalue()
    assert "Modify workspace" in output
    assert "diff preview truncated" in output
    assert '"preview"' in output
