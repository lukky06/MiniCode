from minicode_harness.models import ModelRequest, create_model_client
from minicode_harness.models.kimi import (
    DEFAULT_KIMI_BASE_URL,
    DEFAULT_KIMI_MODEL,
    KimiModelClient,
)
from minicode_harness.models.normalization import normalize_openai_chat_completion_chunks


def test_kimi_client_defaults_to_k3(monkeypatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_MODEL", raising=False)
    monkeypatch.delenv("MOONSHOT_MODEL", raising=False)
    monkeypatch.delenv("KIMI_BASE_URL", raising=False)
    monkeypatch.delenv("MOONSHOT_BASE_URL", raising=False)

    client = KimiModelClient()

    assert client.model == DEFAULT_KIMI_MODEL
    assert client.base_url == DEFAULT_KIMI_BASE_URL
    assert client.api_key == "test-key"
    assert client.capabilities.context_window == 1_000_000
    assert client.capabilities.max_output_tokens == 131_072


def test_create_model_client_supports_kimi_aliases(monkeypatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    assert isinstance(create_model_client("kimi"), KimiModelClient)
    assert isinstance(create_model_client("moonshot"), KimiModelClient)


def test_kimi_payload_uses_k3_fields_and_preserves_reasoning() -> None:
    client = KimiModelClient(api_key="test-key", model="kimi-k3")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    payload = client._request_payload(
        ModelRequest(
            system="system rules",
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "先读取目标文件。",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "content"},
            ],
            tools=tools,
            metadata={
                "max_output_tokens": 8192,
                "reasoning_effort": "high",
                "temperature": 0.2,
            },
        )
    )

    assert payload["model"] == "kimi-k3"
    assert payload["max_completion_tokens"] == 8192
    assert payload["reasoning_effort"] == "high"
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == tools
    assert payload["messages"][1]["reasoning_content"] == "先读取目标文件。"
    assert "temperature" not in payload
    assert "max_tokens" not in payload


def test_stream_normalization_preserves_kimi_reasoning_content() -> None:
    response = normalize_openai_chat_completion_chunks(
        [
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": "先定位"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "问题。",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"source":"workspace","target":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
    )

    assert response.reasoning_content == "先定位问题。"
    assert response.tool_calls[0].name == "read"
    assert response.tool_calls[0].arguments == {
        "source": "workspace",
        "target": "README.md",
    }
