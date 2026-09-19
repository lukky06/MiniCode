import json
from copy import deepcopy
from typing import Any

import pytest

import minicode_harness.context as context_api
from minicode_harness.context import (
    COMPACTED_EXECUTION_HEADING,
    CURRENT_TOOL_FRONTIER_HEADING,
    ContextPreparer,
    LLMSemanticHistoryCompactor,
    SEMANTIC_COMPACTION_PURPOSE,
    SEMANTIC_HISTORY_HEADING,
    SemanticHistoryCompactionError,
    SemanticHistoryItem,
    SemanticHistorySummary,
    TokenBudget,
    compact_current_tool_frontier,
    extract_grounding_file_references,
    group_messages,
    validate_message_protocol,
)
from minicode_harness.context.semantic_compaction import build_semantic_compaction_request
from minicode_harness.context.session_projection import project_canonical_messages
from minicode_harness.context.token import estimate_tokens
from minicode_harness.models import ModelClient, ModelRequest, ModelResponse


def test_context_public_api_hides_semantic_compaction_internals() -> None:
    assert "build_semantic_compaction_request" not in context_api.__all__
    assert "estimate_semantic_compaction_request_tokens" not in context_api.__all__
    assert "serialize_semantic_compaction_input" not in context_api.__all__


def _tool_group(
    index: int,
    *,
    tool_name: str = "read",
    chars: int = 1_200,
) -> list[dict[str, Any]]:
    arguments = (
        {"source": "workspace", "target": f"src/{index}.py"}
        if tool_name != "run_command"
        else {
            "argv": [
                "python",
                "-m",
                "pytest",
                f"tests/test_{index}.py",
                "-q",
            ]
        }
    )
    result = (
        {"path": f"src/{index}.py", "content": "x" * chars}
        if tool_name != "run_command"
        else {
            "status": "passed",
            "returncode": 0,
            "stdout": "1 passed",
            "artifact_path": f"artifacts/pytest_{index}.txt",
        }
    )
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{index}",
            "content": json.dumps(result),
        },
    ]


def _pressure_history() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(7):
        messages.extend(
            [
                {"role": "user", "content": f"old task {index}"},
                {
                    "role": "assistant",
                    "content": f"old answer {index} " + ("a" * 6_000),
                },
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": "run the focused verification"},
            *_tool_group(99, tool_name="run_command", chars=200),
            {"role": "assistant", "content": "verification complete"},
            {"role": "user", "content": "current task must stay exact"},
        ]
    )
    for index in range(5):
        messages.extend(_tool_group(index, chars=1_200))
    return messages


def _semantic_pressure_history(*, tool_chars: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"old constraint {index}"},
                {
                    "role": "assistant",
                    "content": f"old decision {index} " + ("a" * 4_000),
                },
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": "run the focused verification"},
            *_tool_group(99, tool_name="run_command", chars=200),
            {"role": "assistant", "content": "verification complete"},
            {"role": "user", "content": "current task must stay exact"},
        ]
    )
    for index in range(4):
        messages.extend(_tool_group(index, chars=tool_chars))
    return messages


class RecordingSemanticCompactor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def compact(
        self,
        groups: list[list[dict[str, Any]]],
        *,
        compressible_group_indexes: set[int],
        context_group_indexes: set[int] | None = None,
        max_output_tokens: int,
        focus: str | None = None,
    ) -> SemanticHistorySummary:
        self.calls.append(
            {
                "groups": deepcopy(groups),
                "compressible_group_indexes": set(compressible_group_indexes),
                "context_group_indexes": set(context_group_indexes or ()),
                "max_output_tokens": max_output_tokens,
                "focus": focus,
            }
        )
        return SemanticHistorySummary(
            items=[
                SemanticHistoryItem(
                    kind="decision",
                    text="Keep one canonical history and use semantic hard compaction.",
                    source_turn_ids=["t0001"],
                )
            ]
        )


class FailingSemanticCompactor:
    def __init__(self) -> None:
        self.calls = 0

    def compact(self, *args: Any, **kwargs: Any) -> SemanticHistorySummary:
        self.calls += 1
        raise SemanticHistoryCompactionError("invalid summary")


class RecordingModelClient(ModelClient):
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests: list[ModelRequest] = []

    def call_request(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.response


def test_llm_compactor_retains_explicit_focused_verification_policy() -> None:
    client = RecordingModelClient(
        ModelResponse(final_text=json.dumps({"items": []}))
    )
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {
                "role": "user",
                "content": (
                    "后续凡需验证，始终只允许运行 "
                    "python -m pytest -q tests/application/test_assign_ticket.py，"
                    "不要运行全量测试。"
                ),
            }
        ]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=800,
    )

    assert len(summary.items) == 1
    item = summary.items[0]
    assert item.kind == "constraint"
    assert (
        "python -m pytest -q tests/application/test_assign_ticket.py"
        in item.text
    )
    assert "不要运行全量测试" in item.text
    assert item.source_turn_ids == ["t0001"]


def test_llm_compactor_emits_started_and_completed_events() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    client = RecordingModelClient(
        ModelResponse(final_text=json.dumps({"items": []}))
    )
    compactor = LLMSemanticHistoryCompactor(
        client,
        event_handler=lambda event_type, payload: events.append(
            (event_type, dict(payload))
        ),
    )
    groups = group_messages(
        [
            {
                "role": "user",
                "content": (
                    "后续只运行 python -m pytest -q tests/focused.py，"
                    "不要运行全量测试。"
                ),
            }
        ]
    )

    compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=800,
    )

    assert [event_type for event_type, _ in events] == [
        "semantic_compaction_started",
        "semantic_compaction_completed",
    ]
    assert events[0][1]["source_group_count"] == 1
    assert events[1][1]["duration_ms"] >= 0


def test_llm_compactor_uses_latest_explicit_verification_policy() -> None:
    client = RecordingModelClient(
        ModelResponse(
            final_text=json.dumps(
                {
                    "items": [
                        {
                            "kind": "constraint",
                            "text": (
                                "验证时运行 python -m pytest -q tests/old_test.py，"
                                "不要运行全量测试。"
                            ),
                            "source_turn_ids": ["t0001"],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
    )
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {
                "role": "user",
                "content": (
                    "后续只运行 python -m pytest -q tests/old_test.py，"
                    "不要运行全量测试。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "新的验证约束：只运行 "
                    "python -m pytest -q tests/new_test.py，"
                    "不要运行全量测试。"
                ),
            },
        ]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0, 1},
        max_output_tokens=800,
    )

    rendered = "\n".join(item.text for item in summary.items)
    assert "python -m pytest -q tests/new_test.py" in rendered
    assert "python -m pytest -q tests/old_test.py" not in rendered
    assert summary.items[0].source_turn_ids == ["t0002"]


def _hard_pressure_preparer(compactor: Any) -> ContextPreparer:
    return ContextPreparer(
        TokenBudget(
            context_budget=16_000,
            reserved_output=0,
            soft_limit=0.20,
            hard_limit=0.75,
        ),
        semantic_compactor=compactor,
    )


def test_semantic_pressure_sends_current_frontier_as_context_only() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = _hard_pressure_preparer(compactor)
    source = _semantic_pressure_history(tool_chars=1_000)

    prepared = preparer.prepare(
        system_messages=[],
        messages=source,
        tools=[],
    )

    assert len(compactor.calls) == 1
    sent_groups = compactor.calls[0]["groups"]
    sent_text = json.dumps(sent_groups, ensure_ascii=False)
    assert "current task must stay exact" in sent_text
    assert "src/3.py" in sent_text
    assert "old constraint 4" in sent_text
    compressible_indexes = compactor.calls[0]["compressible_group_indexes"]
    context_indexes = compactor.calls[0]["context_group_indexes"]
    assert compressible_indexes
    assert context_indexes
    compressible_text = json.dumps(
        [sent_groups[index] for index in sorted(compressible_indexes)],
        ensure_ascii=False,
    )
    context_text = json.dumps(
        [sent_groups[index] for index in sorted(context_indexes)],
        ensure_ascii=False,
    )
    assert "old constraint 0" in compressible_text
    assert "old constraint 4" in compressible_text
    assert "current task must stay exact" in context_text
    for index in range(4):
        assert f"src/{index}.py" in context_text

    canonical = prepared.request.messages
    assert any(
        str(message.get("content") or "").startswith(SEMANTIC_HISTORY_HEADING)
        for message in canonical
    )
    assert sum(message.get("role") == "tool" for message in canonical) == 4
    assert sum(bool(message.get("tool_calls")) for message in canonical) == 4
    assert canonical[-1]["role"] == "tool"
    assert canonical[-1]["tool_call_id"] == "call_3"
    assert [
        message["content"]
        for message in canonical
        if message.get("role") == "user"
    ][-1] == "current task must stay exact"
    for index in range(4):
        assert f"src/{index}.py" in json.dumps(canonical, ensure_ascii=False)
    assert "src/4.py" not in json.dumps(canonical, ensure_ascii=False)
    assert validate_message_protocol(canonical) == []
    assert prepared.token_estimate <= preparer.budget.hard_token_limit

    semantic_event = next(
        event
        for event in prepared.compression_events
        if event.reason == "semantic_history"
    )
    assert semantic_event.details["attempted"] is True
    assert semantic_event.details["success"] is True
    assert semantic_event.details["removed_groups"] > 0
    assert prepared.compaction_update.semantic is not None


def test_semantic_compaction_runs_once_before_current_turn_completes() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = _hard_pressure_preparer(compactor)
    canonical = _semantic_pressure_history(tool_chars=1_000)
    first = preparer.prepare(
        system_messages=[],
        messages=canonical,
        tools=[],
    )
    messages = [*canonical, *_tool_group(5, chars=4_000)]

    second = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert len(compactor.calls) == 1
    assert not any(
        event.reason == "semantic_history"
        for event in second.compression_events
    )
    assert validate_message_protocol(second.request.messages) == []


def test_current_tool_frontier_keeps_read_edges_and_command_result() -> None:
    read_lines = [f"line {index}" for index in range(30)]
    read_group = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps(
                            {"source": "workspace", "target": "src/service.py"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read_1",
            "content": json.dumps(
                {
                    "path": "src/service.py",
                    "content": "\n".join(read_lines),
                    "start_line": 1,
                    "end_line": 30,
                    "total_lines": 30,
                }
            ),
        },
    ]
    command_group = _tool_group(7, tool_name="run_command", chars=20)

    rendered, omitted = compact_current_tool_frontier(
        [read_group, command_group],
        max_chars=4_000,
    )

    assert omitted == 0
    assert rendered.startswith(CURRENT_TOOL_FRONTIER_HEADING)
    assert "- read | src/service.py" in rendered
    assert "range: 1-30 of 30" in rendered
    assert "line 0" not in rendered
    assert "line 29" not in rendered
    assert "exact_content" not in rendered
    assert "python -m pytest tests/test_7.py -q" in rendered
    assert "returncode: 0" in rendered
    assert "1 passed" in rendered
    assert "tool_calls" not in rendered
    assert "tool_call_id" not in rendered


def test_current_tool_frontier_sanitizes_existing_entries() -> None:
    existing = {
        "role": "assistant",
        "content": (
            f"{CURRENT_TOOL_FRONTIER_HEADING}\n\n"
            "- read | src/service.py\n"
            "  status: success\n"
            "  range: 1-30 of 30\n"
            "  result_head: private source\n\n"
            "- run_command | python -m pytest tests/test_service.py -q\n"
            "  status: passed\n"
            "  returncode: 0"
        ),
    }

    rendered, omitted = compact_current_tool_frontier([[existing]], max_chars=4_000)

    assert omitted == 0
    assert "- read | src/service.py" in rendered
    assert "range: 1-30 of 30" in rendered
    assert "private source" not in rendered
    assert "search_text" not in rendered
    assert "run_command | python -m pytest tests/test_service.py -q" in rendered
    assert "returncode: 0" in rendered


def test_semantic_compaction_keeps_execution_facts_deterministic() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = _hard_pressure_preparer(compactor)

    prepared = preparer.prepare(
        system_messages=[],
        messages=_semantic_pressure_history(tool_chars=1_000),
        tools=[],
    )

    execution_record = next(
        str(message.get("content") or "")
        for message in prepared.request.messages
        if COMPACTED_EXECUTION_HEADING in str(message.get("content") or "")
    )
    assert "python -m pytest tests/test_99.py -q" in execution_record
    assert "returncode: 0" in execution_record
    assert "artifact: artifacts/pytest_99.txt" in execution_record


def test_semantic_failure_is_recorded_without_forcing_hard_compaction() -> None:
    compactor = FailingSemanticCompactor()
    preparer = _hard_pressure_preparer(compactor)

    prepared = preparer.prepare(
        system_messages=[],
        messages=_semantic_pressure_history(tool_chars=1_000),
        tools=[],
    )

    assert compactor.calls == 1
    semantic_event = next(
        event
        for event in prepared.compression_events
        if event.reason == "semantic_history"
    )
    assert semantic_event.details["success"] is False
    assert "invalid summary" in semantic_event.details["failure_reason"]
    assert not any(
        event.details.get("phase") == "hard"
        for event in prepared.compression_events
    )
    assert prepared.compaction_update.semantic_attempt_group_id is not None
    assert not any(
        str(message.get("content") or "").startswith(SEMANTIC_HISTORY_HEADING)
        for message in prepared.request.messages
    )
    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert validate_message_protocol(prepared.request.messages) == []


def test_proactive_semantic_failure_compacts_below_trigger_without_retry() -> None:
    compactor = FailingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=16_000,
            reserved_output=0,
            soft_limit=0.20,
            hard_limit=0.98,
        ),
        semantic_compactor=compactor,
    )

    first = preparer.prepare(
        system_messages=[],
        messages=_semantic_pressure_history(tool_chars=1_000),
        tools=[],
    )

    assert compactor.calls == 1
    semantic_event = next(
        event
        for event in first.compression_events
        if event.reason == "semantic_history"
    )
    assert semantic_event.before_tokens <= preparer.budget.hard_token_limit
    assert not any(
        event.details.get("phase") == "hard"
        for event in first.compression_events
    )
    assert first.compaction_update.semantic_attempt_group_id is not None

    second = preparer.prepare(
        system_messages=[],
        messages=_semantic_pressure_history(tool_chars=1_000),
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert compactor.calls == 1
    assert not any(
        event.reason == "semantic_history"
        for event in second.compression_events
    )


def test_semantic_failure_with_irreducible_active_user_is_throttled() -> None:
    compactor = FailingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=10_000,
            reserved_output=0,
            soft_limit=0.20,
            hard_limit=0.95,
        ),
        semantic_compactor=compactor,
    )
    messages: list[dict[str, Any]] = []
    for index in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"old task {index}"},
                {
                    "role": "assistant",
                    "content": f"old answer {index} " + ("a" * 1_000),
                },
            ]
        )
    active_user = "current task " + ("u" * 28_000)
    messages.append({"role": "user", "content": active_user})

    first = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
    )

    assert compactor.calls == 0
    semantic_event = next(
        event
        for event in first.compression_events
        if event.reason == "semantic_history"
    )
    assert semantic_event.details["attempted"] is False
    assert semantic_event.details["skipped"] is True
    assert semantic_event.details["failure_reason"] == (
        "semantic_compaction_request_exceeds_context_window"
    )
    assert first.token_estimate > preparer.budget.soft_token_limit
    assert first.token_estimate <= preparer.budget.hard_token_limit
    assert [
        message["content"]
        for message in first.request.messages
        if message.get("role") == "user"
    ][-1] == active_user
    assert not any(
        event.details.get("phase") == "hard"
        for event in first.compression_events
    )

    second = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert compactor.calls == 0
    assert second.compression_events == []
    assert second.token_estimate == first.token_estimate


def test_semantic_compactor_is_not_called_below_soft_limit() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(context_budget=100_000, reserved_output=0),
        semantic_compactor=compactor,
    )

    prepared = preparer.prepare(
        system_messages=[],
        messages=[{"role": "user", "content": "small task"}],
        tools=[],
    )

    assert compactor.calls == []
    assert not any(
        event.reason == "semantic_history"
        for event in prepared.compression_events
    )


def test_soft_preclean_does_not_consume_active_turn_semantic_attempt() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=12_000,
            reserved_output=0,
            soft_limit=0.10,
            hard_limit=0.95,
        ),
        semantic_compactor=compactor,
    )
    messages = [
        {"role": "user", "content": "old request"},
        *_tool_group(1, chars=18_000),
        {"role": "assistant", "content": "old answer " + ("a" * 4_000)},
        {"role": "user", "content": "current request"},
    ]

    first = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
    )

    assert compactor.calls == []
    assert any(
        event.details.get("phase") == "soft"
        for event in first.compression_events
    )
    assert first.token_estimate <= preparer.budget.soft_token_limit

    continued_messages = deepcopy(first.request.messages)
    continued_messages.extend(_tool_group(2, chars=9_000))
    second = preparer.prepare(
        system_messages=[],
        messages=continued_messages,
        tools=[],
    )

    assert len(compactor.calls) == 1
    semantic_event = next(
        event
        for event in second.compression_events
        if event.reason == "semantic_history"
    )
    assert semantic_event.details["success"] is True


def test_llm_compactor_marks_all_groups_and_uses_no_tools() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "constraint",
                        "text": "Do not run the full test suite.",
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current request"},
            *_tool_group(1, chars=20),
        ]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0, 1},
        max_output_tokens=800,
    )

    assert len(summary.items) == 1
    request = client.requests[0]
    assert request.tools == []
    assert request.metadata["purpose"] == SEMANTIC_COMPACTION_PURPOSE
    payload = json.loads(request.messages[0]["content"])
    assert len(payload["message_turns"]) == 1
    assert payload["compressible_source_turn_ids"] == ["t0001"]
    assert payload["protected_context_only_turn_ids"] == []
    assert payload["message_turns"][0]["messages"] == [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    assert payload["protected_omitted_group_ids"] == ["g0003", "g0004"]
    assert payload["omitted_execution_group_ids"] == []
    serialized = request.messages[0]["content"]
    assert "old request" in serialized
    assert "current request" not in serialized
    assert "src/1.py" not in serialized


def test_llm_compactor_receives_active_user_and_recent_tool_as_context_only() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "constraint",
                        "text": "old request",
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "newer answer"},
            {"role": "user", "content": "current request"},
            *_tool_group(1, chars=20),
        ]
    )

    compactor.compact(
        groups,
        compressible_group_indexes={0, 1},
        context_group_indexes={2, 3},
        max_output_tokens=800,
    )

    payload = json.loads(client.requests[0].messages[0]["content"])
    assert payload["compressible_source_turn_ids"] == ["t0001"]
    assert payload["protected_context_only_turn_ids"] == ["t0002"]
    assert payload["protected_omitted_group_ids"] == []
    assert payload["omitted_execution_group_ids"] == []
    assert [turn["region"] for turn in payload["message_turns"]] == [
        "compressible",
        "context_only",
    ]
    assert [turn["source_turn_id"] for turn in payload["message_turns"]] == [
        "t0001",
        "t0002",
    ]
    serialized = client.requests[0].messages[0]["content"]
    assert "newer answer" in serialized
    assert "current request" in serialized
    assert "src/1.py" in serialized
    assert "tool_calls" in serialized
    assert "tool_call_id" in serialized


def test_semantic_compaction_runs_when_soft_projection_still_exceeds_soft_limit() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=16_000,
            reserved_output=0,
            soft_limit=0.20,
            hard_limit=0.98,
        ),
        semantic_compactor=compactor,
    )

    prepared = preparer.prepare(
        system_messages=[],
        messages=_semantic_pressure_history(tool_chars=1_000),
        tools=[],
    )

    assert len(compactor.calls) == 1
    event = next(
        item
        for item in prepared.compression_events
        if item.reason == "semantic_history"
    )
    assert event.before_tokens > preparer.budget.soft_token_limit
    assert event.before_tokens <= preparer.budget.hard_token_limit
    assert event.details["phase"] == "semantic"
    assert event.details["success"] is True


def test_active_user_and_recent_tool_are_context_only_and_remain_exact() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=16_000,
            reserved_output=0,
            soft_limit=0.20,
            hard_limit=0.98,
        ),
        semantic_compactor=compactor,
    )
    messages: list[dict[str, Any]] = []
    for index in range(7):
        messages.extend(
            [
                {"role": "user", "content": f"old task {index}"},
                {
                    "role": "assistant",
                    "content": f"old answer {index} " + ("a" * 4_000),
                },
            ]
        )
    active_user = "current request " + ("u" * 8_000)
    messages.append({"role": "user", "content": active_user})
    messages.extend(_tool_group(42, chars=12_000))

    prepared = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
    )

    assert len(compactor.calls) == 1
    assert [
        message["content"]
        for message in prepared.request.messages
        if message.get("role") == "user"
    ][-1] == active_user
    assert "src/42.py" in json.dumps(prepared.request.messages, ensure_ascii=False)
    event = next(
        item
        for item in prepared.compression_events
        if item.reason == "semantic_history"
    )
    assert event.details["success"] is True
    assert event.details["request_required_tokens"] < 16_000
    context_indexes = compactor.calls[0]["context_group_indexes"]
    compressible_indexes = compactor.calls[0]["compressible_group_indexes"]
    assert context_indexes
    assert compressible_indexes
    sent_groups = compactor.calls[0]["groups"]
    context_text = json.dumps(
        [sent_groups[index] for index in sorted(context_indexes)],
        ensure_ascii=False,
    )
    compressible_text = json.dumps(
        [sent_groups[index] for index in sorted(compressible_indexes)],
        ensure_ascii=False,
    )
    assert active_user in context_text
    assert "src/42.py" in context_text
    assert active_user not in compressible_text
    assert "src/42.py" not in compressible_text


def test_semantic_compaction_skips_call_when_auxiliary_request_cannot_fit() -> None:
    compactor = RecordingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=3_000,
            reserved_output=0,
            soft_limit=0.20,
            hard_limit=0.95,
        ),
        semantic_compactor=compactor,
    )
    messages: list[dict[str, Any]] = []
    for index in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"old task {index}"},
                {
                    "role": "assistant",
                    "content": f"old answer {index} " + ("x" * 6_000),
                },
            ]
        )
    messages.append(
        {"role": "user", "content": "current task must stay exact"}
    )

    prepared = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
    )

    assert compactor.calls == []
    event = next(
        item
        for item in prepared.compression_events
        if item.reason == "semantic_history"
    )
    assert event.details["attempted"] is False
    assert event.details["skipped"] is True
    assert event.details["failure_reason"] == (
        "semantic_compaction_request_exceeds_context_window"
    )
    assert event.details["request_required_tokens"] > 3_000
    assert any(
        item.details.get("phase") == "hard"
        for item in prepared.compression_events
    )
    assert prepared.token_estimate <= preparer.budget.hard_token_limit


def test_llm_compactor_drops_execution_claim_but_keeps_valid_items() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "decision",
                        "text": "Keep one canonical history.",
                        "source_turn_ids": ["t0001"],
                    },
                    {
                        "kind": "fact",
                        "text": "Tests passed with returncode=0.",
                        "source_turn_ids": ["t0001"],
                    },
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [{"role": "user", "content": "Keep one canonical history."}]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=800,
    )

    assert [item.text for item in summary.items] == [
        "Keep one canonical history."
    ]


def test_llm_compactor_rejects_fabricated_execution_claims() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "fact",
                        "text": "已修改 tests/test_fabricated.py，测试已通过。",
                        "source_turn_ids": ["t0001"],
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [{"role": "user", "content": "Discuss the cache design only."}]
    )

    with pytest.raises(
        SemanticHistoryCompactionError,
        match="deterministic execution claim",
    ):
        compactor.compact(
            groups,
            compressible_group_indexes={0},
            max_output_tokens=800,
        )


def test_llm_compactor_allows_non_execution_architecture_terms() -> None:
    source = "Use Artifact references and a modified-design review checklist."
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "decision",
                        "text": source,
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages([{"role": "user", "content": source}])

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=800,
    )

    assert summary.items[0].text == source


def test_llm_compactor_allows_non_path_slash_terms() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "fact",
                        "text": "Use EMAIL/WEBHOOK channels and tenant/time boundaries.",
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {
                "role": "user",
                "content": "Compare email and webhook channels with tenant and time boundaries.",
            }
        ]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=800,
    )

    assert summary.items[0].text == (
        "Use EMAIL/WEBHOOK channels and tenant/time boundaries."
    )


def test_llm_compactor_accepts_path_grounded_by_context_only_tool_result() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "frontier",
                        "text": "Continue from supportdesk/domain/satisfaction.py.",
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {"role": "user", "content": "Inspect the current satisfaction module."},
            {"role": "user", "content": "Explain the current file."},
            *_tool_group(7, chars=20),
        ]
    )
    groups[2][0]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"path": "supportdesk/domain/satisfaction.py"}
    )
    groups[2][1]["content"] = json.dumps(
        {
            "path": "supportdesk/domain/satisfaction.py",
            "content": "class SatisfactionRepository: pass",
        }
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        context_group_indexes={1, 2},
        max_output_tokens=800,
    )

    assert summary.items[0].text == (
        "Continue from supportdesk/domain/satisfaction.py."
    )


def test_llm_compactor_accepts_reference_from_another_sent_source_group() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "fact",
                        "text": "The inspected module is satisfaction.py.",
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {"role": "user", "content": "Keep the module explanation concise."},
            {
                "role": "assistant",
                "content": "Read supportdesk/domain/satisfaction.py for the contract.",
            },
        ]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0, 1},
        max_output_tokens=800,
    )

    assert summary.items[0].text == "The inspected module is satisfaction.py."


@pytest.mark.parametrize(
    "expression",
    [
        "repository.get/cache.evict",
        "offset/limit",
        "GET/POST",
        "read/write",
    ],
)
def test_grounding_file_extraction_ignores_slash_separated_terms(
    expression: str,
) -> None:
    assert extract_grounding_file_references(expression) == []


def test_llm_compactor_allows_slash_separated_domain_terms() -> None:
    text = (
        "Use repository.get/cache.evict; keep offset/limit bounded; "
        "compare GET/POST and read/write."
    )
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "fact",
                        "text": text,
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [{"role": "user", "content": "Explain the paired domain terms."}]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=800,
    )

    assert summary.items[0].text == text


@pytest.mark.parametrize(
    "reference",
    [
        "tests/test_fabricated.py",
        "src/nonexistent/service.py",
        r"D:\fake\secret.py",
        "/tmp/fabricated.yaml",
    ],
)
def test_llm_compactor_rejects_ungrounded_file_references(
    reference: str,
) -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "frontier",
                        "text": f"Next inspect {reference}.",
                        "source_turn_ids": ["t0001"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [{"role": "user", "content": "Discuss the cache design only."}]
    )

    with pytest.raises(
        SemanticHistoryCompactionError,
        match="ungrounded reference",
    ):
        compactor.compact(
            groups,
            compressible_group_indexes={0},
            max_output_tokens=800,
        )


def test_llm_compactor_accepts_grounded_repository_file_references() -> None:
    text = (
        "Keep supportdesk/application/assign_ticket.py aligned with "
        "tests/application/test_assign_ticket.py."
    )
    client = RecordingModelClient(
        ModelResponse(
            final_text=json.dumps(
                {
                    "items": [
                        {
                            "kind": "constraint",
                            "text": text,
                            "source_turn_ids": ["t0001"],
                        }
                    ]
                }
            )
        )
    )
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages([{"role": "user", "content": text}])

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=800,
    )

    assert summary.items[0].text == text


def test_llm_compactor_rejects_protected_source_turn_ids() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "constraint",
                        "text": "Invented protected-only constraint.",
                        "source_turn_ids": ["t0002"],
                    }
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current request"},
        ]
    )

    with pytest.raises(
        SemanticHistoryCompactionError,
        match="invalid source_turn_id",
    ):
        compactor.compact(
            groups,
            compressible_group_indexes={0, 1},
            max_output_tokens=800,
        )


def test_llm_compactor_normalizes_bounded_near_schema_output() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "fact",
                        "title": "ignored extra field",
                        "text": "x" * 450,
                        "source_turn_ids": [
                            "t0001",
                            "t0002",
                            "t0003",
                            "t0004",
                            "t0005",
                        ],
                    },
                    {
                        "kind": "fact",
                        "text": "   ",
                        "source_turn_ids": ["t0001"],
                    },
                ],
                "extra": "ignored root field",
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {"role": "user", "content": f"source {index}"}
            for index in range(5)
        ]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0, 1, 2, 3, 4},
        max_output_tokens=800,
    )

    assert len(summary.items) == 1
    assert len(summary.items[0].text) == 280
    assert summary.items[0].source_turn_ids == [
        "t0001",
        "t0002",
        "t0003",
    ]


def test_llm_compactor_bounds_summary_item_count() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "fact",
                        "text": f"Retained fact {index}.",
                        "source_turn_ids": ["t0001"],
                    }
                    for index in range(14)
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [{"role": "user", "content": "Retain the bounded facts."}]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0},
        max_output_tokens=1_600,
    )

    assert len(summary.items) == 10
    assert summary.items[-1].text == "Retained fact 9."


def test_maximum_semantic_summary_remains_within_output_reserve() -> None:
    summary = SemanticHistorySummary(
        items=[
            SemanticHistoryItem(
                kind="fact",
                text="x" * 280,
                source_turn_ids=["t0001", "t0002", "t0003"],
            )
            for _ in range(10)
        ]
    )

    payload = summary.model_dump_json()

    assert json.loads(payload)["items"]
    assert estimate_tokens(payload) < 1_600


def test_manual_focus_is_bounded_and_not_a_source_reference() -> None:
    groups = group_messages(
        [
            {"role": "user", "content": "Keep architecture simple."},
            {"role": "assistant", "content": "Agreed."},
            {"role": "user", "content": "Current request."},
        ]
    )

    request, source_ids = build_semantic_compaction_request(
        groups,
        compressible_group_indexes={0, 1},
        context_group_indexes={2},
        max_output_tokens=800,
        focus="  retain   constraints  " + ("x" * 600),
    )
    payload = json.loads(str(request.messages[0]["content"]))

    assert payload["manual_focus"].startswith("retain constraints")
    assert len(payload["manual_focus"]) == 500
    assert request.metadata["manual_focus"] == payload["manual_focus"]
    assert source_ids == {"t0001"}
    assert payload["manual_focus"] not in payload["compressible_source_turn_ids"]


def test_manual_compaction_preserves_latest_completed_turn_and_focus() -> None:
    messages = [
        {"role": "user", "content": "old constraint one " + ("a" * 2_000)},
        {"role": "assistant", "content": "old decision one " + ("b" * 2_000)},
        {"role": "user", "content": "old constraint two " + ("c" * 2_000)},
        {"role": "assistant", "content": "old decision two " + ("d" * 2_000)},
        {"role": "user", "content": "latest request must stay exact"},
        {"role": "assistant", "content": "latest final answer must stay exact"},
    ]
    compactor = RecordingSemanticCompactor()
    preparer = ContextPreparer(
        TokenBudget(context_budget=16_000, reserved_output=0),
        semantic_compactor=compactor,
    )

    state, event = preparer.manual_compact(
        messages,
        focus="保留重构约束和失败测试",
    )
    compacted = project_canonical_messages(messages, state)

    assert event is not None
    assert event.details["success"] is True
    assert compactor.calls[0]["focus"] == "保留重构约束和失败测试"
    assert any(
        str(message.get("content") or "").startswith(SEMANTIC_HISTORY_HEADING)
        for message in compacted
    )
    assert [
        {"role": message.get("role"), "content": message.get("content")}
        for message in compacted[-2:]
    ] == messages[-2:]
    assert "old constraint one" not in json.dumps(compacted, ensure_ascii=False)
    assert validate_message_protocol(compacted) == []


def test_llm_compactor_rejects_truncated_response_without_retry() -> None:
    client = RecordingModelClient(
        ModelResponse(
            final_text=json.dumps({"items": []}),
            stop_reason="length",
        )
    )
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages([{"role": "user", "content": "old request"}])

    with pytest.raises(
        SemanticHistoryCompactionError,
        match="truncated",
    ):
        compactor.compact(
            groups,
            compressible_group_indexes={0},
            max_output_tokens=1_600,
        )

    assert len(client.requests) == 1


def test_llm_compactor_drops_invalid_source_items_when_valid_items_remain() -> None:
    response = ModelResponse(
        final_text=json.dumps(
            {
                "items": [
                    {
                        "kind": "constraint",
                        "text": "Keep the old request constraint.",
                        "source_turn_ids": ["t0001"],
                    },
                    {
                        "kind": "frontier",
                        "text": "Summarize the protected current request.",
                        "source_turn_ids": ["t0002"],
                    },
                ]
            }
        )
    )
    client = RecordingModelClient(response)
    compactor = LLMSemanticHistoryCompactor(client)
    groups = group_messages(
        [
            {"role": "user", "content": "Keep the old request constraint."},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "Current request is protected."},
        ]
    )

    summary = compactor.compact(
        groups,
        compressible_group_indexes={0, 1},
        context_group_indexes={2},
        max_output_tokens=800,
    )

    assert [item.text for item in summary.items] == [
        "Keep the old request constraint."
    ]


def test_reactive_compaction_remains_model_free() -> None:
    compactor = RecordingSemanticCompactor()
    messages = _pressure_history()
    for index in range(12, 24):
        messages.extend(_tool_group(index, chars=1_000))

    state, event = ContextPreparer(
        semantic_compactor=compactor
    ).reactive_compact(messages)
    compacted = project_canonical_messages(messages, state)

    assert compactor.calls == []
    assert event is not None
    assert event.reason == "reactive_prompt_too_long"
    assert event.details["strategy"] == "stable_tool_group_boundary"
    assert validate_message_protocol(compacted) == []
