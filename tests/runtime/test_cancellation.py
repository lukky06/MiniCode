import json
import subprocess
from typing import Any

from minicode_harness.loop import AgentLoop
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.output import NullOutputSink
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.state import CheckpointStore
from minicode_harness.tools.write_tools import run_command
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def call_request(self, request) -> ModelResponse:
        self.calls += 1
        return self.responses.pop(0)


class CancelAfterFirstToolSink(NullOutputSink):
    def __init__(self, token: CancellationToken) -> None:
        self.token = token
        self.finished = 0

    def tool_call_finished(self, **kwargs) -> None:
        self.finished += 1
        if self.finished == 1:
            self.token.cancel()


def test_agent_loop_honors_pre_cancel_without_calling_model(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run" / "trace.jsonl"
    trace_path.parent.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="must not run")])
    token = CancellationToken()
    token.cancel()

    result = AgentLoop(
        task="Explain the repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        cancellation_token=token,
    ).run()

    assert result.status == "cancelled"
    assert result.stop_reason == "cancelled"
    assert client.calls == 0
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert any(event["type"] == "run_cancelled" for event in events)


def test_cancellation_completes_pending_tool_result_group(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("A = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("B = 2\n", encoding="utf-8")
    trace_path = tmp_path / "run" / "trace.jsonl"
    trace_path.parent.mkdir()
    token = CancellationToken()
    sink = CancelAfterFirstToolSink(token)
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_a",
                        name="read",
                        arguments={"source": "workspace", "target": "a.py"},
                    ),
                    NormalizedToolCall(
                        id="call_b",
                        name="read",
                        arguments={"source": "workspace", "target": "b.py"},
                    ),
                ]
            )
        ]
    )

    result = AgentLoop(
        task="Read two files",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        output_sink=sink,
        cancellation_token=token,
    ).run()

    assert result.status == "cancelled"
    assert result.tool_calls == 2
    checkpoint_store = CheckpointStore(trace_path.parent / "checkpoints")
    checkpoint = checkpoint_store.load_latest()
    assert checkpoint is not None
    assert checkpoint.status == "cancelled"
    checkpoint_history = checkpoint_store.load_history(checkpoint)
    assert [message["role"] for message in checkpoint_history[-3:]] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert checkpoint_history[-2]["tool_call_id"] == "call_a"
    assert checkpoint_history[-1]["tool_call_id"] == "call_b"
    assert "A = 1" in checkpoint_history[-2]["content"]
    assert "B = 2" in checkpoint_history[-1]["content"]


def test_run_command_terminates_process_when_cancelled(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def communicate(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(cmd="pytest", timeout=timeout)
            return b"", b""

    fake = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake)
    token = CancellationToken()
    token.cancel()

    result = run_command(
        tmp_path,
        ["python", "-m", "pytest", "-q"],
        cancellation_token=token,
    )

    assert result.cancelled is True
    assert result.lifecycle_status == "cancelled"
    assert result.returncode == 130
    assert fake.terminated is True
