from __future__ import annotations

from typing import Any

from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.models import ModelClient, ModelRequest, ModelResponse, NormalizedToolCall
from minicode_harness.models import deepseek as deepseek_module
from minicode_harness.models.deepseek import DeepSeekModelClient
from minicode_harness.models.normalization import normalize_openai_chat_completion
from minicode_harness.output import NullOutputSink
from minicode_harness.trace import TraceWriter


class RecordingClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def call_request(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def test_normalization_preserves_reasoning_content_for_tool_call() -> None:
    response = normalize_openai_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "定位到 URL 重建时丢失 auth。",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"source":"workspace","target":"app.py"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    assert response.reasoning_content == "定位到 URL 重建时丢失 auth。"
    assert response.tool_calls[0].name == "read"


def test_deepseek_payload_replays_reasoning_and_sets_effort() -> None:
    client = DeepSeekModelClient(api_key="test-key", model="deepseek-v4-flash")
    payload = client._request_payload(
        ModelRequest(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "先验证 auth 在哪个转换中丢失。",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"source":"workspace","target":"app.py"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
            tools=[],
            metadata={"reasoning_effort": "high"},
        )
    )

    assert payload["reasoning_effort"] == "high"
    assert payload["messages"][0]["reasoning_content"] == "先验证 auth 在哪个转换中丢失。"


def test_deepseek_stream_request_emits_provider_deltas_immediately(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "先分析。",
                        "content": "<final_answer>第一段",
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"content": "第二段</final_answer>"},
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    def fake_stream_post_json(url, payload, *, headers=None, timeout=120.0):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        yield from chunks

    monkeypatch.setattr(deepseek_module, "stream_post_json", fake_stream_post_json)
    client = DeepSeekModelClient(api_key="test-key", model="deepseek-chat")
    deltas: list[str] = []
    reasoning_deltas: list[str] = []

    response = client.stream_request(
        ModelRequest(messages=[{"role": "user", "content": "test"}], tools=[]),
        on_text_delta=deltas.append,
        on_reasoning_delta=reasoning_deltas.append,
    )

    assert captured["payload"]["stream"] is True
    assert captured["headers"] == {"Authorization": "Bearer test-key"}
    assert reasoning_deltas == ["先分析。"]
    assert deltas == ["<final_answer>第一段", "第二段</final_answer>"]
    assert response.final_text == "第一段第二段"
    assert response.reasoning_content == "先分析。"


def test_agent_loop_streams_provider_reasoning_to_output_sink(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sink = NullOutputSink()
    reasoning: list[str] = []
    sink.model_reasoning_delta = reasoning.append  # type: ignore[method-assign]
    client = RecordingClient(
        [
            ModelResponse(
                final_text="<final_answer>done</final_answer>",
                reasoning_content="先确认入口，再回答。",
            )
        ]
    )

    result = AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "stream" / "trace.jsonl"),
        output_sink=sink,
        stream_model=True,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert reasoning == ["先确认入口，再回答。"]


def test_agent_loop_keeps_reasoning_in_canonical_tool_message(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("value = 1\n", encoding="utf-8")
    client = RecordingClient(
        [
            ModelResponse(
                reasoning_content="先读取目标文件。",
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="read",
                        arguments={"source": "workspace", "target": "app.py"},
                    )
                ],
            ),
            ModelResponse(final_text="done"),
        ]
    )

    AgentLoop(
        task="Explain app.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        no_skills=True,
        config=AgentLoopConfig(reasoning_effort="high"),
    ).run()

    assert client.requests[0].metadata["reasoning_effort"] == "high"
    assistant_messages = [
        message
        for message in client.requests[1].messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert assistant_messages[0]["reasoning_content"] == "先读取目标文件。"
