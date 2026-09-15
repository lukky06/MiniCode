from __future__ import annotations

from io import StringIO
from threading import Thread
import time

from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.state import ApprovalDecision, ApprovalRequest
from minicode_harness.tui_bridge.approval import JsonlApprovalClient
from minicode_harness.tui_bridge.output import JsonlOutputSink
from minicode_harness.tui_bridge.protocol import parse_server_message
from minicode_harness.tui_bridge.writer import JsonlEventWriter


def _client():
    stream = StringIO()
    sink = JsonlOutputSink(JsonlEventWriter(stream))
    return JsonlApprovalClient(sink), stream


def _request(request_id: str = "approval_1") -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id,
        tool_call_id="call_1",
        tool_name="run_command",
        risk_level="high",
        arguments={
            "argv": ["pytest", "secret-model-argument"],
            "api_key": "must-not-cross-protocol",
        },
        can_approve_session=True,
        preview={
            "summary": "Run focused tests",
            "command": "pytest tests/test_x.py -q",
            "policy_category": "test",
            "allowed": True,
        },
    )


def _wait_for_lines(stream: StringIO, count: int, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while len(stream.getvalue().splitlines()) < count:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for approval event.")
        time.sleep(0.005)


def test_approval_client_emits_safe_preview_and_resolves_matching_id() -> None:
    client, stream = _client()
    result = {}
    thread = Thread(target=lambda: result.setdefault("response", client.decide(_request())))
    thread.start()
    _wait_for_lines(stream, 1)

    event = parse_server_message(stream.getvalue().splitlines()[0])
    assert event.type == "approval_required"
    assert event.id == "approval_1"
    assert event.tool_call_id == "call_1"
    assert event.summary == "Run focused tests"
    assert event.can_approve_session is True
    assert "pytest tests/test_x.py -q" in (event.details or "")
    assert "must-not-cross-protocol" not in stream.getvalue()
    assert "secret-model-argument" not in stream.getvalue()

    resolved, error = client.resolve("approval_1", "approve")
    assert resolved is True
    assert error is None
    thread.join(1.0)
    assert result["response"].decision == ApprovalDecision.APPROVE


def test_approval_client_accepts_session_grant_decision() -> None:
    client, stream = _client()
    result = {}
    thread = Thread(target=lambda: result.setdefault("response", client.decide(_request())))
    thread.start()
    _wait_for_lines(stream, 1)

    assert client.resolve("approval_1", "approve_session") == (True, None)
    thread.join(1.0)
    assert result["response"].decision == ApprovalDecision.APPROVE_SESSION


def test_stale_approval_id_does_not_resolve_current_request() -> None:
    client, stream = _client()
    result = {}
    thread = Thread(target=lambda: result.setdefault("response", client.decide(_request())))
    thread.start()
    _wait_for_lines(stream, 1)

    resolved, error = client.resolve("old_approval", "reject")
    assert resolved is False
    assert error == "Stale approval response: old_approval."
    assert client.has_pending() is True

    assert client.resolve("approval_1", "skip") == (True, None)
    thread.join(1.0)
    assert result["response"].decision == ApprovalDecision.SKIP


def test_cancellation_aborts_pending_approval_without_tui_response() -> None:
    client, stream = _client()
    token = CancellationToken()
    client.bind_cancellation(token)
    result = {}

    thread = Thread(target=lambda: result.setdefault("response", client.decide(_request())))
    thread.start()
    _wait_for_lines(stream, 1)
    token.cancel()

    thread.join(1.0)
    assert not thread.is_alive()
    assert result["response"].decision == ApprovalDecision.ABORT
    assert client.has_pending() is False


def test_parallel_approval_calls_are_serialized_to_one_visible_pending() -> None:
    client, stream = _client()
    results = {}

    first = Thread(
        target=lambda: results.setdefault("first", client.decide(_request("approval_1")))
    )
    first.start()
    _wait_for_lines(stream, 1)

    second = Thread(
        target=lambda: results.setdefault("second", client.decide(_request("approval_2")))
    )
    second.start()
    time.sleep(0.05)
    assert len(stream.getvalue().splitlines()) == 1

    assert client.resolve("approval_1", "reject") == (True, None)
    first.join(1.0)
    _wait_for_lines(stream, 2)

    events = [
        parse_server_message(line) for line in stream.getvalue().splitlines()
    ]
    assert [event.id for event in events] == ["approval_1", "approval_2"]

    assert client.resolve("approval_2", "abort") == (True, None)
    second.join(1.0)
    assert results["first"].decision == ApprovalDecision.REJECT
    assert results["second"].decision == ApprovalDecision.ABORT
