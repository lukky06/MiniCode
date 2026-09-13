from __future__ import annotations

from io import StringIO
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace

import pytest

from minicode_harness.output import NullOutputSink
from minicode_harness.runtime import CollaborationMode
from minicode_harness.runtime.run_executor import RunExecutionRequest
from minicode_harness.state import (
    ApprovalDecision,
    ApprovalRequest,
    ReplSessionStore,
    RunStore,
    StaticApprovalClient,
    UserInputRequest,
)
from minicode_harness.terminal.commands import SlashCommandRouter
from minicode_harness.terminal.types import (
    CommandResult,
    TerminalContext,
    TerminalSessionSettings,
)
import minicode_harness.tui_bridge.backend as backend_module
from minicode_harness.tui_bridge.approval import JsonlApprovalClient
from minicode_harness.tui_bridge.user_input import JsonlUserInputClient
from minicode_harness.tui_bridge.backend import JsonlBackend, _launch_session
from minicode_harness.tui_bridge.output import JsonlOutputSink
from minicode_harness.tui_bridge.protocol import (
    ApprovalResponseMessage,
    CancelMessage,
    CommandMessage,
    SteerMessage,
    TaskMessage,
    UserInputResponseMessage,
    parse_server_message,
)
from minicode_harness.tui_bridge.writer import JsonlEventWriter


class BlockingExecutor:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.request = None
        self.cancellation = None
        self.steering = None

    def execute(
        self,
        request,
        *,
        output_sink,
        approval_client,
        cancellation_token,
        steering_queue,
    ):
        del approval_client
        self.request = request
        self.cancellation = cancellation_token
        self.steering = steering_queue
        output_sink.run_started("run_1")
        self.started.set()
        while not self.release.is_set() and not cancellation_token.is_cancelled:
            time.sleep(0.005)
        status = "cancelled" if cancellation_token.is_cancelled else "completed"
        output_sink.run_finished(status=status, run_id="run_1")
        return SimpleNamespace(status=status, run_id="run_1")


def _backend(executor: BlockingExecutor):
    stream = StringIO()
    sink = JsonlOutputSink(JsonlEventWriter(stream))
    backend = JsonlBackend(
        session_id="session_1",
        executor=executor,
        output_sink=sink,
        approval_client=StaticApprovalClient(),
        request_factory=lambda task: RunExecutionRequest(
            task=task,
            workspace=".",
            dry_run=True,
        ),
    )
    return backend, sink, stream


def test_backend_launch_session_preserves_new_continue_and_exact(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    first = store.create(workspace)
    second = store.create(workspace)

    continued = _launch_session(
        store,
        workspace,
        session_mode="continue",
        session_id=None,
    )
    exact = _launch_session(
        store,
        workspace,
        session_mode="exact",
        session_id=first.session_id,
    )
    fresh = _launch_session(
        store,
        workspace,
        session_mode="new",
        session_id=None,
    )

    assert continued.session_id == second.session_id
    assert exact.session_id == first.session_id
    assert fresh.session_id not in {first.session_id, second.session_id}


def test_backend_launch_session_rejects_invalid_id_combinations(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")

    with pytest.raises(ValueError, match="cannot be supplied"):
        _launch_session(
            store,
            workspace,
            session_mode="new",
            session_id="session_bad",
        )
    with pytest.raises(ValueError, match="requires a session ID"):
        _launch_session(
            store,
            workspace,
            session_mode="exact",
            session_id=None,
        )


def test_backend_routes_idle_task_then_running_steering_fifo() -> None:
    executor = BlockingExecutor()
    backend, _, stream = _backend(executor)

    backend.handle_message(TaskMessage(text="first task"))
    assert executor.started.wait(1.0)
    backend.handle_message(SteerMessage(text="first correction"))
    backend.handle_message(SteerMessage(text="second correction"))

    assert executor.request.task == "first task"
    assert executor.steering.dequeue() == "first correction"
    assert executor.steering.dequeue() == "second correction"

    executor.release.set()
    backend.close(timeout=1.0)
    assert backend.is_active is False
    assert [parse_server_message(line).type for line in stream.getvalue().splitlines()] == [
        "run_started",
        "run_finished",
    ]


def test_backend_cancel_only_sets_existing_runtime_token() -> None:
    executor = BlockingExecutor()
    backend, _, stream = _backend(executor)

    backend.handle_message(TaskMessage(text="cancel me"))
    assert executor.started.wait(1.0)
    backend.handle_message(CancelMessage())
    backend.close(timeout=1.0)

    assert executor.cancellation.is_cancelled is True
    events = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    assert [event.type for event in events] == ["run_started", "run_finished"]
    assert events[-1].status == "cancelled"


def test_backend_rejects_second_task_and_idle_steer_without_touching_runtime() -> None:
    executor = BlockingExecutor()
    backend, _, stream = _backend(executor)

    backend.handle_message(SteerMessage(text="too early"))
    backend.handle_message(TaskMessage(text="first"))
    assert executor.started.wait(1.0)
    backend.handle_message(TaskMessage(text="second"))

    executor.release.set()
    backend.close(timeout=1.0)

    events = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    errors = [event.message for event in events if event.type == "error"]
    assert errors == [
        "No active Run is available for steering.",
        "A Run is already active; send a steer message instead.",
    ]
    assert executor.request.task == "first"


def test_backend_eof_requests_cooperative_cancellation() -> None:
    executor = BlockingExecutor()
    backend, _, stream = _backend(executor)
    input_stream = StringIO('{"type":"task","text":"long run"}\n')

    backend.run_forever(input_stream)

    assert executor.started.is_set()
    assert executor.cancellation.is_cancelled is True
    events = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    assert events[0].type == "session_started"
    assert events[0].session_id == "session_1"
    assert any(event.type == "run_finished" and event.status == "cancelled" for event in events)


def test_backend_invalid_input_is_protocol_error_not_exception() -> None:
    executor = BlockingExecutor()
    backend, _, stream = _backend(executor)

    backend.run_forever(StringIO('{"type":"unknown"}\n'))

    events = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    assert events[0].type == "session_started"
    assert events[1].type == "error"
    assert "Invalid protocol message" in events[1].message


def test_backend_routes_idle_command_through_shared_router(tmp_path: Path) -> None:
    executor = BlockingExecutor()
    backend, sink, stream = _backend(executor)
    run_store = RunStore(tmp_path / "runs")

    class FakeRouter:
        def execute(self, command, context):
            assert command.name == "help"
            assert context.cancellation_token is not None
            assert context.steering_queue is not None
            return CommandResult(
                status="completed",
                content="Python-owned help facts",
            )

    backend.command_router = FakeRouter()
    backend.command_context = TerminalContext(
        workspace=tmp_path,
        provider="test",
        model=None,
        write_enabled=True,
        prompt_cache_enabled=True,
        repository_memory_enabled=True,
        subagents_enabled=True,
        mcp_config=None,
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
        run_store=run_store,
    )
    backend.handle_message(CommandMessage(text="/help"))
    backend.close(timeout=1.0)

    [event] = [
        parse_server_message(line)
        for line in stream.getvalue().splitlines()
    ]
    assert event.type == "panel"
    assert event.name == "help"
    assert event.content == "Python-owned help facts"
    assert sink.current_run_id is None


def test_plan_command_changes_collaboration_mode_for_next_run(tmp_path: Path) -> None:
    executor = BlockingExecutor()
    backend, _, stream = _backend(executor)
    settings = TerminalSessionSettings()
    backend.command_router = SlashCommandRouter()
    backend.command_context = TerminalContext(
        workspace=tmp_path,
        provider="test",
        model=None,
        write_enabled=True,
        prompt_cache_enabled=True,
        repository_memory_enabled=True,
        subagents_enabled=True,
        mcp_config=None,
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
        run_store=RunStore(tmp_path / "runs"),
        session_settings=settings,
    )
    backend.request_factory = lambda task: RunExecutionRequest(
        task=task,
        workspace=tmp_path,
        dry_run=True,
        collaboration_mode=settings.collaboration_mode,
    )

    backend.handle_message(CommandMessage(text="/plan on"))
    _wait_for_event_type(stream, "panel")
    backend.handle_message(TaskMessage(text="plan this change"))

    assert executor.started.wait(1.0)
    assert executor.request is not None
    assert executor.request.collaboration_mode == CollaborationMode.PLAN

    executor.release.set()
    backend.close(timeout=1.0)


def test_completed_plan_run_can_handoff_directly_to_default_execution(tmp_path) -> None:
    stream = StringIO()
    sink = JsonlOutputSink(JsonlEventWriter(stream))
    user_input = JsonlUserInputClient(sink)
    settings = TerminalSessionSettings(collaboration_mode=CollaborationMode.PLAN)

    class PlanHandoffExecutor:
        def __init__(self) -> None:
            self.requests = []

        def execute(
            self,
            request,
            *,
            output_sink,
            approval_client,
            user_input_client,
            cancellation_token,
            steering_queue,
        ):
            del approval_client, user_input_client, cancellation_token, steering_queue
            self.requests.append(request)
            run_id = f"run_{len(self.requests)}"
            output_sink.run_started(run_id)
            output_sink.run_finished(status="completed", run_id=run_id)
            return SimpleNamespace(
                status="completed",
                run_id=run_id,
                final_text="implementation plan",
            )

    executor = PlanHandoffExecutor()
    backend = JsonlBackend(
        session_id="session_1",
        executor=executor,
        output_sink=sink,
        approval_client=StaticApprovalClient(),
        user_input_client=user_input,
        request_factory=lambda task: RunExecutionRequest(
            task=task,
            workspace=tmp_path,
            dry_run=True,
            collaboration_mode=settings.collaboration_mode,
        ),
        command_context=TerminalContext(
            workspace=tmp_path,
            provider="test",
            model=None,
            write_enabled=True,
            prompt_cache_enabled=True,
            repository_memory_enabled=True,
            subagents_enabled=True,
            mcp_config=None,
            output_sink=NullOutputSink(),
            approval_client=StaticApprovalClient(),
            run_store=RunStore(tmp_path / "runs-handoff"),
            session_settings=settings,
        ),
    )

    backend.handle_message(TaskMessage(text="plan this change"))
    _wait_for_event_type(stream, "user_input_required")
    events = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    handoff = next(event for event in events if event.type == "user_input_required")
    assert [option.label for option in handoff.options] == [
        "Execute plan",
        "Continue planning",
        "Finish planning",
    ]

    backend.handle_message(
        UserInputResponseMessage(id=handoff.id, selected_index=0)
    )
    deadline = time.monotonic() + 1.0
    while len(executor.requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    backend.close(timeout=1.0)

    assert [request.collaboration_mode for request in executor.requests] == [
        CollaborationMode.PLAN,
        CollaborationMode.DEFAULT,
    ]
    assert executor.requests[1].task == "Execute the approved plan above."
    assert settings.collaboration_mode == CollaborationMode.DEFAULT


def test_backend_switches_session_when_command_returns_forked_session(tmp_path) -> None:
    stream = StringIO()
    sink = JsonlOutputSink(JsonlEventWriter(stream))

    class SessionSwitchRouter:
        def execute(self, command, context):
            del command, context
            return CommandResult(
                status="completed",
                content="Forked session",
                session_id="session_forked",
            )

    backend = JsonlBackend(
        session_id="session_original",
        executor=BlockingExecutor(),
        output_sink=sink,
        approval_client=StaticApprovalClient(),
        request_factory=lambda task: RunExecutionRequest(
            task=task,
            workspace=tmp_path,
            dry_run=True,
        ),
        command_router=SessionSwitchRouter(),
        command_context=TerminalContext(
            workspace=tmp_path,
            provider="test",
            model=None,
            write_enabled=True,
            prompt_cache_enabled=True,
            repository_memory_enabled=True,
            subagents_enabled=True,
            mcp_config=None,
            output_sink=NullOutputSink(),
            approval_client=StaticApprovalClient(),
            run_store=RunStore(tmp_path / "runs-switch"),
            session_id="session_original",
        ),
    )

    backend.handle_message(CommandMessage(text="/fork"))
    _wait_for_event_type(stream, "session_started")
    backend.close(timeout=1.0)

    events = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    assert backend.session_id == "session_forked"
    assert backend.command_context is not None
    assert backend.command_context.session_id == "session_forked"
    assert any(
        event.type == "session_started" and event.session_id == "session_forked"
        for event in events
    )


def test_backend_rejects_command_while_run_is_active() -> None:
    executor = BlockingExecutor()
    backend, _, stream = _backend(executor)

    backend.handle_message(TaskMessage(text="task"))
    assert executor.started.wait(1.0)
    backend.handle_message(CommandMessage(text="/diff"))

    events = [
        parse_server_message(line)
        for line in stream.getvalue().splitlines()
    ]
    assert any(
        event.type == "error"
        and "unavailable while a Run is active" in event.message
        for event in events
    )

    executor.release.set()
    backend.close(timeout=1.0)


def test_backend_routes_approval_response_to_pending_python_client() -> None:
    stream = StringIO()
    sink = JsonlOutputSink(JsonlEventWriter(stream))
    approval = JsonlApprovalClient(sink)
    completed = Event()
    captured = {}

    class ApprovalExecutor:
        def execute(
            self,
            request,
            *,
            output_sink,
            approval_client,
            cancellation_token,
            steering_queue,
        ):
            del request, cancellation_token, steering_queue
            output_sink.run_started("run_approval")
            captured["response"] = approval_client.decide(
                ApprovalRequest(
                    id="approval_1",
                    tool_call_id="call_1",
                    tool_name="run_command",
                    risk_level="high",
                    preview={
                        "summary": "Run focused tests",
                        "command": "pytest tests/test_x.py -q",
                    },
                )
            )
            output_sink.run_finished(status="completed", run_id="run_approval")
            completed.set()
            return SimpleNamespace(status="completed", run_id="run_approval")

    backend = JsonlBackend(
        session_id="session_1",
        executor=ApprovalExecutor(),
        output_sink=sink,
        approval_client=approval,
        request_factory=lambda task: RunExecutionRequest(
            task=task,
            workspace=".",
            dry_run=True,
        ),
    )

    backend.handle_message(TaskMessage(text="task"))
    _wait_for_event_type(stream, "approval_required")
    backend.handle_message(
        ApprovalResponseMessage(
            id="approval_1",
            decision="skip",
        )
    )

    assert completed.wait(1.0)
    backend.close(timeout=1.0)
    assert captured["response"].decision == ApprovalDecision.SKIP


def test_backend_routes_user_input_response_to_pending_python_client() -> None:
    stream = StringIO()
    sink = JsonlOutputSink(JsonlEventWriter(stream))
    user_input = JsonlUserInputClient(sink)
    completed = Event()
    captured = {}

    class UserInputExecutor:
        def execute(
            self,
            request,
            *,
            output_sink,
            approval_client,
            user_input_client,
            cancellation_token,
            steering_queue,
        ):
            del request, approval_client, cancellation_token, steering_queue
            output_sink.run_started("run_input")
            captured["response"] = user_input_client.choose(
                UserInputRequest(
                    id="input_1",
                    question="Choose compatibility strategy",
                    options=[
                        {"label": "strict", "description": "Break old callers"},
                        {"label": "compat", "description": "Keep compatibility"},
                    ],
                )
            )
            output_sink.run_finished(status="completed", run_id="run_input")
            completed.set()
            return SimpleNamespace(status="completed", run_id="run_input")

    backend = JsonlBackend(
        session_id="session_1",
        executor=UserInputExecutor(),
        output_sink=sink,
        approval_client=StaticApprovalClient(),
        request_factory=lambda task: RunExecutionRequest(
            task=task,
            workspace=".",
            dry_run=True,
        ),
        user_input_client=user_input,
    )

    backend.handle_message(TaskMessage(text="plan task"))
    _wait_for_event_type(stream, "user_input_required")
    backend.handle_message(
        UserInputResponseMessage(id="input_1", selected_index=1)
    )

    assert completed.wait(1.0)
    backend.close(timeout=1.0)
    assert captured["response"].selected_index == 1


def _wait_for_event_type(
    stream: StringIO,
    event_type: str,
    timeout: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if any(
            parse_server_message(line).type == event_type
            for line in stream.getvalue().splitlines()
        ):
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {event_type}.")
        time.sleep(0.005)


def test_backend_main_redirects_incidental_prints_to_stderr(monkeypatch) -> None:
    protocol_stdout = StringIO()
    debug_stderr = StringIO()

    class FakeBackend:
        def __init__(self, output_sink) -> None:
            self.output_sink = output_sink

        def run_forever(self, input_stream) -> None:
            del input_stream
            print("debug must not enter protocol stdout")
            self.output_sink.session_started("session_1")

    monkeypatch.setattr(
        backend_module,
        "_build_backend",
        lambda args, *, output_sink: FakeBackend(output_sink),
    )
    monkeypatch.setattr(backend_module.sys, "stdin", StringIO(""))
    monkeypatch.setattr(backend_module.sys, "stdout", protocol_stdout)
    monkeypatch.setattr(backend_module.sys, "stderr", debug_stderr)

    assert backend_module.main([]) == 0

    lines = protocol_stdout.getvalue().splitlines()
    assert len(lines) == 1
    event = parse_server_message(lines[0])
    assert event.type == "session_started"
    assert "debug must not enter protocol stdout" in debug_stderr.getvalue()
