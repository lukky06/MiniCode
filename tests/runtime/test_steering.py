from concurrent.futures import ThreadPoolExecutor
import json

from minicode_harness.loop import AgentLoop, AgentLoopConfig, AgentRunResult
from minicode_harness.state import CheckpointStore, ReplSessionMemory, RunStore
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.output import NullOutputSink
from minicode_harness.resume import resume_run
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.run_executor import RunExecutionRequest, RunExecutor
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def call_request(self, request):
        self.requests.append(list(request.as_chat_messages()))
        return self.responses.pop(0)


def _read_call(call_id: str, path: str) -> NormalizedToolCall:
    return NormalizedToolCall(
        id=call_id,
        name="read",
        arguments={"source": "workspace", "target": path},
    )


def _message_roles(messages: list[dict]) -> list[str]:
    return [str(message["role"]) for message in messages]


def test_steering_queue_is_fifo_and_thread_safe() -> None:
    queue = SteeringQueue()
    values = [f"message-{index}" for index in range(32)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(queue.enqueue, values))

    assert len(queue) == len(values)
    assert sorted(queue.dequeue() for _ in values) == sorted(values)
    assert queue.dequeue() is None


def test_steering_waits_for_complete_tool_batch_and_is_visible_to_next_model(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("A = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("B = 2\n", encoding="utf-8")
    queue = SteeringQueue()

    class EnqueueAfterFirstTool(NullOutputSink):
        def __init__(self) -> None:
            self.finished = 0

        def tool_call_finished(self, **kwargs) -> None:
            self.finished += 1
            if self.finished == 1:
                queue.enqueue("use the newly requested constraint")

    client = ScriptedModelClient(
        [
            ModelResponse(tool_calls=[_read_call("a", "a.py"), _read_call("b", "b.py")]),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Read both files",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        output_sink=EnqueueAfterFirstTool(),
        steering_queue=queue,
    )

    result = loop.run()

    assert result.status == "completed"
    second_messages = client.requests[1]
    assert _message_roles(second_messages)[-4:] == ["assistant", "tool", "tool", "user"]
    assert second_messages[-1] == {
        "role": "user",
        "content": "use the newly requested constraint",
    }
    assert [message["tool_call_id"] for message in second_messages[-3:-1]] == [
        "a",
        "b",
    ]
    assert len(queue) == 0


def test_steering_is_one_per_boundary_fifo_and_persisted_to_session_checkpoint(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("A = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("B = 2\n", encoding="utf-8")
    queue = SteeringQueue()
    queue.enqueue("first correction")
    queue.enqueue("second correction")
    session = ReplSessionMemory(workspace=str(workspace.resolve()))
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(tool_calls=[_read_call("a", "a.py")]),
            ModelResponse(tool_calls=[_read_call("b", "b.py")]),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Read both files",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        session_memory=session,
        steering_queue=queue,
    )

    result = loop.run()

    assert result.status == "completed"
    assert client.requests[1][-1]["content"] == "first correction"
    assert [message["content"] for message in client.requests[2] if message["role"] == "user"][-2:] == [
        "first correction",
        "second correction",
    ]
    assert [
        event["step"]
        for event in (
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        )
        if event["type"] == "steering_message_consumed"
    ] == [1, 2]
    assert [message["content"] for message in session.load_message_history() if message["role"] == "user"][-3:] == [
        "Read both files",
        "first correction",
        "second correction",
    ]
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert [message["content"] for message in checkpoint.message_history if message["role"] == "user"][-3:] == [
        "Read both files",
        "first correction",
        "second correction",
    ]


def test_final_boundary_does_not_consume_or_append_steering(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    queue = SteeringQueue()
    queue.enqueue("must remain pending")
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient([ModelResponse(final_text="done")])

    result = AgentLoop(
        task="Answer",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        config=AgentLoopConfig(max_steps=1),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        steering_queue=queue,
    ).run()

    assert result == AgentRunResult(
        status="completed",
        final_text="done",
        steps=1,
        tool_calls=0,
        stop_reason="final_text",
    )
    assert len(queue) == 1
    assert not any(
        event["type"] == "steering_message_consumed"
        for event in (
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        )
    )


def test_cancelled_run_keeps_pending_steering_and_persists_consumed_history(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("A = 1\n", encoding="utf-8")
    token = CancellationToken()
    queue = SteeringQueue()
    queue.enqueue("pending after cancellation")
    client = ScriptedModelClient(
        [ModelResponse(tool_calls=[_read_call("a", "a.py")])]
    )

    class CancelAfterTool(NullOutputSink):
        def tool_call_finished(self, **kwargs) -> None:
            token.cancel()

    trace_path = tmp_path / "run" / "trace.jsonl"
    result = AgentLoop(
        task="Read file",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        output_sink=CancelAfterTool(),
        cancellation_token=token,
        steering_queue=queue,
    ).run()

    assert result.status == "cancelled"
    assert len(queue) == 1
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert "pending after cancellation" not in str(checkpoint.message_history)
    assert [message["role"] for message in checkpoint.message_history[-2:]] == [
        "assistant",
        "tool",
    ]


def test_run_executor_passes_explicit_steering_queue_seam(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    queue = SteeringQueue()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.create_model_client",
        lambda **kwargs: object(),
    )

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            calls.update(kwargs)
            self.modified_files = []
            self.run_state = type("RunState", (), {
                "inspected_files": [],
                "verification": type("Verification", (), {"status": "not_run"})(),
            })()

        def run(self) -> AgentRunResult:
            return AgentRunResult(
                status="completed",
                final_text="done",
                steps=1,
                tool_calls=0,
                stop_reason="final_text",
            )

    monkeypatch.setattr("minicode_harness.runtime.run_executor.AgentLoop", FakeLoop)
    result = RunExecutor(run_store=RunStore(tmp_path / "runs")).execute(
        RunExecutionRequest(
            task="task",
            workspace=workspace,
            repository_memory_enabled=False,
        ),
        output_sink=NullOutputSink(),
        approval_client=type("Approval", (), {})(),
        steering_queue=queue,
    )

    assert result.status == "completed"
    assert calls["steering_queue"] is queue


def test_consumed_steering_survives_resume(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("A = 1\n", encoding="utf-8")
    run_store = RunStore(tmp_path / "runs")
    run_session = run_store.create_run(
        task="Read file",
        workspace=workspace,
        run_id="run_20260901_001",
        max_steps=3,
        repository_memory_enabled=False,
    )
    token = CancellationToken()

    class CancelAfterDequeue(SteeringQueue):
        def dequeue(self):
            message = super().dequeue()
            if message is not None:
                token.cancel()
            return message

    queue = CancelAfterDequeue()
    queue.enqueue("persist this correction")
    first_trace = TraceWriter(run_store.path_for(run_session.run_id) / "trace.jsonl")
    first = AgentLoop(
        task=run_session.task,
        workspace=workspace,
        model_client=ScriptedModelClient(
            [ModelResponse(tool_calls=[_read_call("a", "a.py")])]
        ),
        trace_writer=first_trace,
        config=AgentLoopConfig(max_steps=3),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        cancellation_token=token,
        steering_queue=queue,
        run_id=run_session.run_id,
    )

    assert first.run().status == "cancelled"
    resume_client = ScriptedModelClient([ModelResponse(final_text="resumed")])
    resumed = resume_run(
        run_session.run_id,
        run_store=run_store,
        model_client=resume_client,
        stream_model=False,
        steering_queue=SteeringQueue(),
    )

    assert resumed.status == "completed"
    assert any(
        message.get("role") == "user"
        and message.get("content") == "persist this correction"
        for message in resume_client.requests[0]
    )
