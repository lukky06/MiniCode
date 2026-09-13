from io import StringIO
import json
from typing import Any

from minicode_harness.loop import AgentLoop
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelProviderError, ModelResponse
from minicode_harness.output import TextOutputSink
from minicode_harness.runtime import ModelRecoveryConfig, ModelRecoveryPolicy
from minicode_harness.trace import TraceWriter


class StreamingTruncatedModelClient(ModelClient):
    def __init__(self) -> None:
        self.calls = 0

    def call_request(self, request):
        raise AssertionError("stream_request should be used")

    def stream_request(self, request, *, on_text_delta=None):
        self.calls += 1
        if self.calls == 1:
            if on_text_delta is not None:
                on_text_delta("<final_answer>partial visible")
            return ModelResponse(
                final_text="<final_answer>partial visible</final_answer>",
                stop_reason="length",
            )
        if on_text_delta is not None:
            on_text_delta("<final_answer>continued</final_answer>")
        return ModelResponse(
            final_text="<final_answer>continued</final_answer>",
            stop_reason="stop",
        )


class FlakyModelClient(ModelClient):
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def call_request(self, request):
        self.calls += 1
        self.requests.append(request.as_chat_messages())
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _policy() -> ModelRecoveryPolicy:
    return ModelRecoveryPolicy(
        ModelRecoveryConfig(
            max_transient_retries=2,
            base_delay_seconds=0,
            max_delay_seconds=0,
            jitter_ratio=0,
            max_reactive_compactions=1,
            max_output_recoveries=1,
        ),
        sleep=lambda _: None,
        random_fraction=lambda: 0,
    )


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_agent_loop_retries_transient_model_failure(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = FlakyModelClient(
        [
            ModelProviderError("busy", kind="overloaded", status_code=529),
            ModelResponse(final_text="Recovered."),
        ]
    )

    result = AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        recovery_policy=_policy(),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert client.calls == 2
    retries = [event for event in _events(trace_path) if event["type"] == "model_retry_scheduled"]
    assert retries[0]["kind"] == "overloaded"


def test_agent_loop_reactively_compacts_prompt_once(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = FlakyModelClient(
        [
            ModelProviderError("context length exceeded", kind="prompt_too_long", status_code=413),
            ModelResponse(final_text="Recovered after compaction."),
        ]
    )

    initial_history = [
        {"role": "user", "content": "Explain repository"},
        *[
            {"role": "assistant", "content": f"earlier answer {index}"}
            for index in range(6)
        ],
    ]
    result = AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        recovery_policy=_policy(),
        no_skills=True,
        initial_message_history=initial_history,
    ).run()

    assert result.status == "completed"
    assert client.calls == 2
    second_request = client.requests[1]
    canonical_messages = [
        message
        for message in second_request
        if message.get("role") != "system"
    ]
    assert canonical_messages == initial_history

    events = _events(trace_path)
    recoveries = [event for event in events if event["type"] == "model_recovery"]
    assert len(recoveries) == 1
    assert recoveries[0]["action"] == "reactive_compact"
    compactions = [
        event
        for event in events
        if event.get("type") == "context_compressed"
        and event.get("reason") == "reactive_prompt_too_long"
    ]
    assert compactions == []


def test_streaming_truncation_continues_without_replaying_visible_text(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = StreamingTruncatedModelClient()
    stream = StringIO()

    result = AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "stream" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        recovery_policy=_policy(),
        output_sink=TextOutputSink(stream=stream),
        stream_model=True,
        no_skills=True,
    ).run()

    output = stream.getvalue()
    assert result.status == "completed"
    assert client.calls == 2
    assert output.count("partial visible") == 1
    assert output.count("continued") == 1


def test_agent_loop_continues_after_output_truncation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = FlakyModelClient(
        [
            ModelResponse(final_text="Partial answer", stop_reason="length"),
            ModelResponse(final_text="Finished answer"),
        ]
    )

    result = AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        recovery_policy=_policy(),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert result.final_text == "Finished answer"
    assert client.calls == 2
