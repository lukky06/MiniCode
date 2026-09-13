import pytest

from minicode_harness.models import (
    ModelRequest,
    ModelResponse,
    UnsupportedProviderError,
    create_model_client,
)
from minicode_harness.models.normalization import (
    normalize_anthropic_message,
    normalize_ollama_chat_response,
    normalize_openai_chat_completion,
)


def test_normalize_openai_chat_completion_tool_call() -> None:
    response = normalize_openai_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"source":"workspace","target":"README.md"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )

    assert response.final_text is None
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "read"
    assert response.tool_calls[0].arguments == {
        "source": "workspace",
        "target": "README.md",
    }
    assert response.usage is not None
    assert response.usage.total_tokens == 15


def test_normalize_openai_tool_call_preserves_current_search_arguments() -> None:
    response = normalize_openai_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"source":"workspace","kind":"text","query":"RedisLease","path":"src","limit":50,"max_depth":12,"use_regex":false,"case_sensitive":true}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {
        "source": "workspace",
        "kind": "text",
        "query": "RedisLease",
        "path": "src",
        "limit": 50,
        "max_depth": 12,
        "use_regex": False,
        "case_sensitive": True,
    }


def test_normalize_openai_chat_completion_preserves_malformed_tool_arguments() -> None:
    raw_arguments = '{"path":"generate_test_files.py","content":"abc\\'
    response = normalize_openai_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "write",
                                    "arguments": raw_arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    tool_call = response.tool_calls[0]
    assert tool_call.arguments == {}
    assert tool_call.raw_arguments == raw_arguments
    assert tool_call.argument_parse_error is not None
    assert tool_call.arguments_likely_truncated is True
    assert "raw_arguments" not in tool_call.model_dump(mode="json")


def test_normalize_openai_chat_completion_extracts_dsml_tool_calls() -> None:
    response = normalize_openai_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "content": """我来看看关注功能相关的代码，然后写一个测试。

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name=\"Read\">
<｜｜DSML｜｜parameter name=\"file_path\" string=\"true\">src/main/java/com/hmdp/service/IFollowService.java</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
<｜｜DSML｜｜invoke name=\"Read\">
<｜｜DSML｜｜parameter name=\"file_path\" string=\"true\">src/main/java/com/hmdp/controller/FollowController.java</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>""",
                    }
                }
            ]
        }
    )

    assert response.kind() == "tool_calls"
    assert response.final_text is None
    assert response.assistant_commentary == "我来看看关注功能相关的代码，然后写一个测试。"
    assert [tool_call.name for tool_call in response.tool_calls] == ["read", "read"]
    assert response.tool_calls[0].arguments == {
        "source": "workspace",
        "target": "src/main/java/com/hmdp/service/IFollowService.java",
    }
    assert response.tool_calls[1].arguments == {
        "source": "workspace",
        "target": "src/main/java/com/hmdp/controller/FollowController.java",
    }


def test_model_response_extracts_final_answer_xml_contract() -> None:
    response = ModelResponse(
        final_text="""
<final_answer>
  <summary>已完成仓库架构分析。</summary>
  <changed_files></changed_files>
  <verification>只读分析，无需运行测试。</verification>
  <blockers></blockers>
</final_answer>
"""
    ).enforce_turn_contract()

    assert response.kind() == "final_text"
    assert response.output_protocol == "final_answer_xml"
    assert response.structured_final["summary"] == "已完成仓库架构分析。"
    assert response.structured_final["changed_files"] == []
    assert response.structured_final["verification"] == "只读分析，无需运行测试。"
    assert response.final_text == "已完成仓库架构分析。\n验证：只读分析，无需运行测试。"


def test_model_response_renders_explanation_final_answer_xml_fields() -> None:
    response = ModelResponse(
        final_text="""
<final_answer>
  <summary>这是一个 Spring Boot 本地生活项目。</summary>
  <details>入口类启动 Spring 容器，controller 层接收 HTTP 请求，service 层承载业务逻辑。</details>
  <key_points>
    <point>FollowServiceImpl 负责关注、取关、判断关注和共同关注。</point>
    <point>Redis Set 用于保存用户关注集合并计算共同关注。</point>
  </key_points>
  <evidence>
    <item>已读取 FollowServiceImpl.java。</item>
  </evidence>
</final_answer>
"""
    ).enforce_turn_contract()

    assert response.kind() == "final_text"
    assert response.output_protocol == "final_answer_xml"
    assert response.structured_final["details"].startswith("入口类启动")
    assert response.structured_final["key_points"] == [
        "FollowServiceImpl 负责关注、取关、判断关注和共同关注。",
        "Redis Set 用于保存用户关注集合并计算共同关注。",
    ]
    assert response.structured_final["evidence"] == ["已读取 FollowServiceImpl.java。"]
    assert response.final_text == (
        "这是一个 Spring Boot 本地生活项目。\n"
        "详细说明：\n"
        "入口类启动 Spring 容器，controller 层接收 HTTP 请求，service 层承载业务逻辑。\n"
        "要点：\n"
        "- FollowServiceImpl 负责关注、取关、判断关注和共同关注。\n"
        "- Redis Set 用于保存用户关注集合并计算共同关注。\n"
        "依据：\n"
        "- 已读取 FollowServiceImpl.java。"
    )



def test_model_response_recovers_malformed_final_answer_xml() -> None:
    response = ModelResponse(
        final_text="<final_answer><summary>第一轮 & 非法 XML。</summary></final_answer>"
    ).enforce_turn_contract()

    assert response.kind() == "final_text"
    assert response.output_protocol == "malformed_final_answer_xml_recovered"
    assert response.invalid_reason is None
    assert response.structured_final["summary"] == "第一轮 & 非法 XML。"
    assert response.final_text == "第一轮 & 非法 XML。"


def test_model_response_recovers_structured_text_from_broken_final_answer_xml() -> None:
    response = ModelResponse(
        final_text="<final_answer><summary>missing close tag</final_answer>"
    ).enforce_turn_contract()

    assert response.kind() == "final_text"
    assert response.output_protocol == "malformed_final_answer_xml_recovered"
    assert response.final_text == "missing close tag"


def test_model_response_normalizes_legacy_plain_final_text() -> None:
    response = ModelResponse(final_text="done").enforce_turn_contract()

    assert response.kind() == "final_text"
    assert response.output_protocol == "plain_text"
    assert response.structured_final["summary"] == "done"
    assert response.final_text == "done"


def test_normalize_openai_compatible_usage_includes_prompt_cache_tokens() -> None:
    response = normalize_openai_chat_completion(
        {
            "choices": [{"message": {"content": "done"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 36,
            },
        }
    )

    assert response.usage is not None
    assert response.usage.cached_input_tokens == 64
    assert response.usage.cache_miss_input_tokens == 36


def test_normalize_openai_cached_tokens_from_prompt_details() -> None:
    response = normalize_openai_chat_completion(
        {
            "choices": [{"message": {"content": "done"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 32},
            },
        }
    )

    assert response.usage is not None
    assert response.usage.cached_input_tokens == 32
    assert response.usage.cache_miss_input_tokens == 68


def test_normalize_anthropic_message_tool_use() -> None:
    response = normalize_anthropic_message(
        {
            "content": [
                {"type": "text", "text": "I will inspect the files."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"source": "workspace", "kind": "files", "query": "**/*", "path": ".", "limit": 200, "max_depth": 12},
                },
            ],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        }
    )

    assert response.kind() == "tool_calls"
    assert response.final_text is None
    assert response.assistant_commentary == "I will inspect the files."
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {
        "source": "workspace",
        "kind": "files",
        "query": "**/*",
        "path": ".",
        "limit": 200,
        "max_depth": 12,
    }
    assert response.usage is not None
    assert response.usage.input_tokens == 12


def test_normalize_ollama_chat_response_tool_call() -> None:
    response = normalize_ollama_chat_response(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"source": "workspace", "kind": "text", "query": "CouponService", "path": ".", "limit": 50, "max_depth": 12, "use_regex": False, "case_sensitive": True},
                        }
                    }
                ],
            },
            "prompt_eval_count": 20,
            "eval_count": 8,
        }
    )

    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {
        "source": "workspace",
        "kind": "text",
        "query": "CouponService",
        "path": ".",
        "limit": 50,
        "max_depth": 12,
        "use_regex": False,
        "case_sensitive": True,
    }
    assert response.usage is not None
    assert response.usage.total_tokens == 28


def test_create_model_client_rejects_unknown_provider() -> None:
    with pytest.raises(UnsupportedProviderError, match="Unsupported provider"):
        create_model_client("unknown")


def test_qwen_client_defaults_to_dashscope(monkeypatch) -> None:
    from minicode_harness.models.qwen import (
        DEFAULT_QWEN_BASE_URL,
        DEFAULT_QWEN_MODEL,
        QwenModelClient,
    )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    client = QwenModelClient()

    assert client.model == DEFAULT_QWEN_MODEL
    assert client.base_url == DEFAULT_QWEN_BASE_URL
    assert client.api_key == "test-key"


def test_qwen_payload_omits_tool_fields_when_no_tools_are_supplied() -> None:
    from minicode_harness.models.qwen import QwenModelClient

    client = QwenModelClient(api_key="test-key", model="qwen-plus")
    payload = client._request_payload(
        ModelRequest(messages=[{"role": "user", "content": "answer now"}], tools=[])
    )

    assert payload == {
        "model": "qwen-plus",
        "messages": [{"role": "user", "content": "answer now"}],
        "max_tokens": 8192,
    }


def test_qwen_client_accepts_alias_env_vars(monkeypatch) -> None:
    from minicode_harness.models.qwen import QwenModelClient

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_MODEL", "qwen-coder-plus")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.test/compatible-mode/v1/")

    client = QwenModelClient()

    assert client.model == "qwen-coder-plus"
    assert client.base_url == "https://example.test/compatible-mode/v1"
    assert client.api_key == "qwen-key"


def test_create_model_client_supports_qwen_aliases(monkeypatch) -> None:
    from minicode_harness.models.qwen import QwenModelClient

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert isinstance(create_model_client("qwen"), QwenModelClient)
    assert isinstance(create_model_client("dashscope"), QwenModelClient)
    assert isinstance(create_model_client("tongyi"), QwenModelClient)


def test_openai_call_request_preserves_exact_request_sections() -> None:
    from minicode_harness.models.openai import OpenAIModelClient

    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    client = OpenAIModelClient.__new__(OpenAIModelClient)
    client.model = "test-model"
    client._client = FakeClient()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = client.call_request(
        ModelRequest(
            system="system rules",
            messages=[{"role": "user", "content": "inspect code"}],
            tools=tools,
        )
    )

    assert response.final_text == "ok"
    assert captured["messages"] == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "inspect code"},
    ]
    assert captured["tools"] == tools


def test_anthropic_call_request_keeps_system_out_of_messages() -> None:
    from minicode_harness.models.anthropic import AnthropicModelClient

    captured: dict[str, object] = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {"content": [{"type": "text", "text": "ok"}]}

    class FakeClient:
        messages = FakeMessages()

    client = AnthropicModelClient.__new__(AnthropicModelClient)
    client.model = "test-model"
    client._client = FakeClient()

    response = client.call_request(
        ModelRequest(
            system="primary system",
            messages=[
                {"role": "system", "content": "legacy embedded system"},
                {"role": "user", "content": "inspect code"},
            ],
            tools=[],
        )
    )

    assert response.final_text == "ok"
    assert captured["system"] == "primary system\n\nlegacy embedded system"
    assert captured["messages"] == [{"role": "user", "content": "inspect code"}]


def test_anthropic_merges_compacted_execution_with_adjacent_assistant_message() -> None:
    from minicode_harness.models.anthropic import _to_anthropic_messages

    _, messages = _to_anthropic_messages(
        [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "[MiniCode compacted execution]\n\n- write_file | a.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "content"},
        ]
    )

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][1]["type"] == "tool_use"


def test_qwen_request_payload_accepts_model_request() -> None:
    from minicode_harness.models.qwen import QwenModelClient

    client = QwenModelClient(api_key="test-key", model="qwen-plus")
    payload = client._request_payload(
        ModelRequest(
            system="system rules",
            messages=[{"role": "user", "content": "answer now"}],
            tools=[],
        )
    )

    assert payload == {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "answer now"},
        ],
        "max_tokens": 8192,
    }


def test_model_request_renders_required_openai_tool_choice() -> None:
    request = ModelRequest(metadata={"required_tool_name": "read_file"})

    assert request.required_tool_name() == "read_file"
    assert request.openai_tool_choice() == {
        "type": "function",
        "function": {"name": "read_file"},
    }
    assert ModelRequest().openai_tool_choice() == "auto"


def test_model_request_reads_optional_temperature() -> None:
    assert ModelRequest(metadata={"temperature": 0.0}).temperature() == 0.0
    assert ModelRequest().temperature() is None
    assert ModelRequest(metadata={"temperature": True}).temperature() is None


def test_deepseek_request_payload_applies_explicit_temperature() -> None:
    from minicode_harness.models.deepseek import DeepSeekModelClient

    client = DeepSeekModelClient(api_key="test-key", model="deepseek-chat")
    payload = client._request_payload(
        ModelRequest(
            messages=[{"role": "user", "content": "review memory"}],
            metadata={
                "purpose": "memory_review_v2",
                "temperature": 0.0,
                "max_output_tokens": 256,
            },
        )
    )

    assert payload["tool_choice"] == "auto"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 256


def test_qwen_request_payload_forces_required_tool() -> None:
    from minicode_harness.models.qwen import QwenModelClient

    client = QwenModelClient(api_key="test-key", model="qwen-plus")
    payload = client._request_payload(
        ModelRequest(
            messages=[{"role": "user", "content": "decide memory"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "read resource",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            metadata={"required_tool_name": "read"},
        )
    )

    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "read"},
    }
