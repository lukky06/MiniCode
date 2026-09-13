from __future__ import annotations

import json

from minicode_harness.context import (
    CURRENT_TASK_HEADING,
    SEMANTIC_HISTORY_HEADING,
    ContextPreparer,
    SemanticHistoryItem,
    SemanticHistorySummary,
    TokenBudget,
    insert_current_task_record,
    strip_task_protocol,
    validate_message_protocol,
)


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def test_strip_task_protocol_removes_pure_task_group() -> None:
    messages = [
        {"role": "user", "content": "修复问题"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "task_1", "task", {"action": "create", "tasks": ["定位问题"]}
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "task_1",
            "content": '{"created":[["1","定位问题"]]}',
        },
    ]

    assert strip_task_protocol(messages) == [
        {"role": "user", "content": "修复问题"}
    ]


def test_strip_task_protocol_preserves_non_task_pair_in_mixed_group() -> None:
    messages = [
        {"role": "user", "content": "修复问题"},
        {
            "role": "assistant",
            "content": "继续处理。",
            "tool_calls": [
                _tool_call(
                    "task_1",
                    "task",
                    {"action": "update", "updates": {"1": "completed"}},
                ),
                _tool_call(
                    "read_1",
                    "read",
                    {"source": "workspace", "target": "app.py"},
                ),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "task_1",
            "content": '{"updated":{"1":"completed"}}',
        },
        {
            "role": "tool",
            "tool_call_id": "read_1",
            "content": "source",
        },
    ]

    filtered = strip_task_protocol(messages)

    assert filtered[1]["content"] == "继续处理。"
    assert [
        call["function"]["name"]
        for call in filtered[1]["tool_calls"]
    ] == ["read"]
    assert filtered[2] == {
        "role": "tool",
        "tool_call_id": "read_1",
        "content": "source",
    }
    assert validate_message_protocol(filtered) == []


def test_insert_current_task_record_replaces_old_record_after_latest_user() -> None:
    messages = [
        {"role": "user", "content": "旧任务"},
        {"role": "assistant", "content": "旧回答"},
        {"role": "user", "content": "修复缓存问题并补充测试"},
        {
            "role": "assistant",
            "content": (
                "[MiniCode current task]\n"
                "Request: 旧投影\n\n"
                "Open steps:\n"
                "- #1 [pending] 旧步骤\n"
                "[/MiniCode current task]"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "read_1",
                    "read",
                    {"source": "workspace", "target": "cache.py"},
                )
            ],
        },
        {"role": "tool", "tool_call_id": "read_1", "content": "source"},
    ]

    rebuilt = insert_current_task_record(
        messages,
        user_task="修复缓存问题并补充测试",
        open_rows=[
            ["2", "in_progress", "实现修复"],
            ["3", "pending", "补充测试"],
        ],
    )

    records = [
        message
        for message in rebuilt
        if str(message.get("content") or "").startswith(CURRENT_TASK_HEADING)
    ]
    assert len(records) == 1
    assert rebuilt[2]["role"] == "user"
    assert rebuilt[3] == records[0]
    assert "Request: 修复缓存问题并补充测试" in records[0]["content"]
    assert "- #2 [in_progress] 实现修复" in records[0]["content"]
    assert "- #3 [pending] 补充测试" in records[0]["content"]
    assert rebuilt[4]["tool_calls"][0]["function"]["name"] == "read"
    assert validate_message_protocol(rebuilt) == []


def test_insert_current_task_record_omits_projection_without_open_steps() -> None:
    messages = [
        {"role": "user", "content": "修复问题"},
        {
            "role": "assistant",
            "content": (
                "[MiniCode current task]\n"
                "Request: 修复问题\n\n"
                "Open steps:\n"
                "- #1 [completed] 已完成\n"
                "[/MiniCode current task]"
            ),
        },
    ]

    assert insert_current_task_record(
        messages,
        user_task="修复问题",
        open_rows=[],
    ) == [{"role": "user", "content": "修复问题"}]


def _task_protocol_messages() -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    "task_1", "task", {"action": "create", "tasks": ["定位问题"]}
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "task_1",
            "content": '{"created":[["1","定位问题"]]}',
        },
    ]


def _read_protocol(index: int, *, chars: int = 900) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call(
                    f"read_{index}",
                    "read",
                    {"source": "workspace", "target": f"src/{index}.py"},
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"read_{index}",
            "content": "x" * chars,
        },
    ]


def _task_tool_names(messages: list[dict[str, object]]) -> list[str]:
    return [
        str(call["function"]["name"])
        for message in messages
        for call in message.get("tool_calls") or []
        if str(call["function"]["name"]) == "task"
    ]


def test_prepare_does_not_read_or_project_tasks_without_compaction() -> None:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "修复问题"},
        *_task_protocol_messages(),
    ]
    provider_calls = 0

    def provider() -> tuple[str, list[list[str]]]:
        nonlocal provider_calls
        provider_calls += 1
        return "修复问题", [["1", "pending", "定位问题"]]

    prepared = ContextPreparer(
        TokenBudget(context_budget=100_000, reserved_output=0)
    ).prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        task_projection_provider=provider,
    )

    assert provider_calls == 0
    assert prepared.request.messages == messages
    assert _task_tool_names(prepared.request.messages) == ["task"]
    assert not any(
        str(message.get("content") or "").startswith(CURRENT_TASK_HEADING)
        for message in prepared.request.messages
    )


def test_soft_compaction_replaces_task_protocol_with_current_task_record() -> None:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "修复缓存问题并补充测试"},
        *_task_protocol_messages(),
    ]
    for index in range(8):
        messages.extend(_read_protocol(index))
    provider_calls = 0

    def provider() -> tuple[str, list[list[str]]]:
        nonlocal provider_calls
        provider_calls += 1
        return (
            "修复缓存问题并补充测试",
            [
                ["1", "completed", "定位问题"],
                ["2", "in_progress", "实现修复"],
                ["3", "pending", "补充测试"],
            ][1:],
        )

    preparer = ContextPreparer(
        TokenBudget(
            context_budget=4_000,
            reserved_output=0,
            soft_limit=0.25,
            hard_limit=0.95,
        )
    )
    prepared = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        task_projection_provider=provider,
    )

    assert provider_calls == 1
    assert _task_tool_names(prepared.request.messages) == []
    assert prepared.request.messages[0]["role"] == "user"
    record = prepared.request.messages[1]
    assert str(record.get("content") or "").startswith(CURRENT_TASK_HEADING)
    assert "- #2 [in_progress] 实现修复" in record["content"]
    assert "- #3 [pending] 补充测试" in record["content"]
    assert "定位问题" not in record["content"]
    assert prepared.compression_events[-1].after_tokens == prepared.token_estimate
    assert validate_message_protocol(prepared.request.messages) == []


def test_reactive_compaction_replaces_task_protocol_and_counts_projection() -> None:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "修复问题"},
        *_task_protocol_messages(),
    ]
    for index in range(8):
        messages.extend(_read_protocol(index, chars=1_200))
    provider_calls = 0

    def provider() -> tuple[str, list[list[str]]]:
        nonlocal provider_calls
        provider_calls += 1
        return "修复问题", [["2", "in_progress", "实现修复"]]

    preparer = ContextPreparer()
    compaction_state, event = preparer.reactive_compact(
        messages,
        task_projection_provider=provider,
    )

    assert provider_calls == 1
    assert event is not None
    assert compaction_state.execution is not None
    assert event.after_tokens > 0

    prepared = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        compaction_state=compaction_state,
        task_projection_provider=provider,
    )
    assert provider_calls == 2
    assert _task_tool_names(prepared.request.messages) == []
    assert any(
        str(message.get("content") or "").startswith(CURRENT_TASK_HEADING)
        for message in prepared.request.messages
    )
    assert validate_message_protocol(prepared.request.messages) == []


class _SemanticCompactor:
    def __init__(self) -> None:
        self.calls: list[list[list[dict[str, object]]]] = []

    def compact(
        self,
        groups: list[list[dict[str, object]]],
        *,
        compressible_group_indexes: set[int],
        context_group_indexes: set[int] | None = None,
        max_output_tokens: int,
        focus: str | None = None,
    ) -> SemanticHistorySummary:
        self.calls.append(groups)
        return SemanticHistorySummary(
            items=[
                SemanticHistoryItem(
                    kind="decision",
                    text="保留当前缓存修复约束。",
                    source_turn_ids=["t0001"],
                )
            ]
        )


def test_semantic_compaction_excludes_task_protocol_and_restores_projection() -> None:
    messages: list[dict[str, object]] = []
    for index in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"历史约束 {index} " + "a" * 2_000},
                {"role": "assistant", "content": f"历史结论 {index} " + "b" * 2_000},
            ]
        )
    messages.append({"role": "user", "content": "修复当前缓存问题"})
    messages.extend(_task_protocol_messages())
    for index in range(4):
        messages.extend(_read_protocol(index, chars=1_000))

    compactor = _SemanticCompactor()
    provider_calls = 0

    def provider() -> tuple[str, list[list[str]]]:
        nonlocal provider_calls
        provider_calls += 1
        return "修复当前缓存问题", [["2", "in_progress", "实现修复"]]

    preparer = ContextPreparer(
        TokenBudget(
            context_budget=16_000,
            reserved_output=0,
            soft_limit=0.20,
            semantic_limit=0.38,
            hard_limit=0.75,
        ),
        semantic_compactor=compactor,
    )
    prepared = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        task_projection_provider=provider,
    )

    assert provider_calls == 1
    assert len(compactor.calls) == 1
    compactor_input = json.dumps(compactor.calls[0], ensure_ascii=False)
    assert '"name": "task"' not in compactor_input
    assert CURRENT_TASK_HEADING not in compactor_input
    assert _task_tool_names(prepared.request.messages) == []
    assert any(
        str(message.get("content") or "").startswith(SEMANTIC_HISTORY_HEADING)
        for message in prepared.request.messages
    )
    user_index = max(
        index
        for index, message in enumerate(prepared.request.messages)
        if message.get("role") == "user"
    )
    assert str(
        prepared.request.messages[user_index + 1].get("content") or ""
    ).startswith(CURRENT_TASK_HEADING)
    assert prepared.compression_events[-1].after_tokens == prepared.token_estimate
    assert validate_message_protocol(prepared.request.messages) == []
