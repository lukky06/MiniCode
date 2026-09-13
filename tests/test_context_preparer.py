import json

from minicode_harness.context import (
    CURRENT_TOOL_FRONTIER_HEADING,
    CompactionPolicy,
    ContextPreparer,
    PromptBudgetExceeded,
    TokenBudget,
    validate_message_protocol,
)
from minicode_harness.context.preparer import (
    _preclean_history_for_semantic_compaction,
)
from minicode_harness.context.session_projection import (
    COMPACTED_TOOL_HEADING,
    project_canonical_messages,
)


def _tool_messages(count: int, *, tool_name="read", chars=800):
    messages = [{"role": "user", "content": "original task"}]
    for index in range(count):
        arguments = (
            {"source": "workspace", "target": f"src/{index}.py"}
            if tool_name == "read"
            else {"argv": ["python", "-m", "pytest", f"tests/test_{index}.py"]}
        )
        result = (
            "x" * chars
            if tool_name != "run_command"
            else json.dumps({"status": "passed", "returncode": 0})
        )
        messages.extend(
            [
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
                    "content": result,
                },
            ]
        )
    return messages


def _tool_result_status(message: dict) -> str:
    try:
        payload = json.loads(str(message.get("content") or ""))
    except json.JSONDecodeError:
        return ""
    return str(payload.get("status") or "") if isinstance(payload, dict) else ""


def test_compaction_state_is_separate_from_canonical_messages() -> None:
    messages = _tool_messages(8, chars=1_200)
    original = json.loads(json.dumps(messages))
    preparer = ContextPreparer(
        TokenBudget(context_budget=4_000, reserved_output=0, soft_limit=0.2)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert messages == original
    assert prepared.compaction_update.execution is not None
    assert all(
        not any(str(key).startswith("_minicode_") for key in message)
        for message in messages
    )


def test_same_state_reuses_identical_projection() -> None:
    messages = _tool_messages(8, chars=1_200)
    preparer = ContextPreparer(
        TokenBudget(context_budget=4_000, reserved_output=0, soft_limit=0.2)
    )
    first = preparer.prepare(system_messages=[], messages=messages, tools=[])

    second = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert second.request.messages == first.request.messages
    assert second.compaction_update == first.compaction_update
    assert second.compression_events == []


def test_low_pressure_keeps_messages_unchanged():
    messages = _tool_messages(2, chars=20)
    preparer = ContextPreparer(TokenBudget(context_budget=100_000, reserved_output=0))

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert prepared.request.messages == messages
    assert prepared.history_groups_compacted == 0


def test_soft_compaction_drops_old_reads_and_keeps_recent_protocol_groups():
    messages = _tool_messages(10, chars=1200)
    preparer = ContextPreparer(
        TokenBudget(context_budget=5_000, reserved_output=0, soft_limit=0.25, hard_limit=0.95)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert prepared.history_groups_compacted > 0
    assert prepared.request.messages[0]["role"] == "user"
    assert prepared.request.messages[0]["content"] == "original task"
    tool_messages = [
        message
        for message in prepared.request.messages
        if message.get("role") == "tool"
    ]
    retained_ids = [message.get("tool_call_id") for message in tool_messages]
    assert retained_ids[-1] == "call_9"
    assert 1 <= len(retained_ids) < 10
    soft_event = next(
        event
        for event in prepared.compression_events
        if event.details.get("phase") == "soft"
    )
    assert soft_event.details["protected_current_tokens"] <= 1_250
    assert any(
        str(message.get("content") or "").startswith(COMPACTED_TOOL_HEADING)
        for message in prepared.request.messages
    )
    assert validate_message_protocol(prepared.request.messages) == []


def test_soft_compaction_preserves_failed_command_protocol_group():
    messages = _tool_messages(6, tool_name="run_command", chars=500)
    messages[2]["content"] = json.dumps(
        {"status": "failed", "returncode": 1, "summary": "1 failed"}
    )
    preparer = ContextPreparer(
        TokenBudget(context_budget=2_000, reserved_output=0, soft_limit=0.25, hard_limit=0.95)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert any(
        message.get("tool_call_id") == "call_0"
        and _tool_result_status(message) == "failed"
        for message in prepared.request.messages
        if message.get("role") == "tool"
    )
    assert validate_message_protocol(prepared.request.messages) == []


def test_current_parallel_tool_results_are_retained_together() -> None:
    messages = _tool_messages(6, chars=1_200)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "current_a",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"src/a.py"}',
                        },
                    },
                    {
                        "id": "current_b",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"src/b.py"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "current_a", "content": "a" * 400},
            {"role": "tool", "tool_call_id": "current_b", "content": "b" * 400},
        ]
    )
    preparer = ContextPreparer(
        TokenBudget(context_budget=2_000, reserved_output=0, soft_limit=0.25, hard_limit=0.95)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    retained_ids = {
        message.get("tool_call_id")
        for message in prepared.request.messages
        if message.get("role") == "tool"
    }
    assert {"current_a", "current_b"} <= retained_ids
    assert validate_message_protocol(prepared.request.messages) == []


def test_soft_compaction_keeps_sequential_current_read_results_until_hard_limit() -> None:
    messages = [
        {"role": "user", "content": "implement using both files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_a",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"src/a.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read_a", "content": "a" * 1_400},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_b",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"src/b.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read_b", "content": "b" * 1_400},
    ]
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=4_000,
            reserved_output=0,
            soft_limit=0.2,
            hard_limit=0.95,
        )
    )

    prepared = preparer.prepare(
        system_messages=[{"role": "system", "content": "s" * 2_000}],
        messages=messages,
        tools=[],
    )

    retained_ids = {
        message.get("tool_call_id")
        for message in prepared.request.messages
        if message.get("role") == "tool"
    }
    assert retained_ids == {"read_a", "read_b"}
    assert prepared.token_estimate > preparer.budget.soft_token_limit
    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert validate_message_protocol(prepared.request.messages) == []


def test_soft_compaction_keeps_parallel_and_sequential_multi_file_evidence() -> None:
    messages = [
        {"role": "user", "content": "old completed task 1"},
        {"role": "assistant", "content": "old fact " * 3_000},
        {"role": "user", "content": "old completed task 2"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "old completed task 3"},
        {"role": "assistant", "content": "third answer"},
        {"role": "user", "content": "old completed task 4"},
        {"role": "assistant", "content": "fourth answer"},
        {"role": "user", "content": "implement search using all four files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_service",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"supportdesk/application/search_tickets.py"}',
                    },
                },
                {
                    "id": "read_pagination",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"supportdesk/common/pagination.py"}',
                    },
                },
                {
                    "id": "read_test",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"tests/application/test_search_tickets.py"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "read_service", "content": "s" * 700},
        {"role": "tool", "tool_call_id": "read_pagination", "content": "p" * 700},
        {"role": "tool", "tool_call_id": "read_test", "content": "t" * 700},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_repository",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"supportdesk/infrastructure/ticket_repository.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read_repository", "content": "r" * 900},
    ]
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=5_000,
            reserved_output=0,
            soft_limit=0.05,
            hard_limit=0.95,
        )
    )

    prepared = preparer.prepare(
        system_messages=[{"role": "system", "content": "s" * 2_000}],
        messages=messages,
        tools=[],
    )

    retained_ids = {
        message.get("tool_call_id")
        for message in prepared.request.messages
        if message.get("role") == "tool"
    }
    assert retained_ids == {
        "read_service",
        "read_pagination",
        "read_test",
        "read_repository",
    }
    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert validate_message_protocol(prepared.request.messages) == []


def test_soft_preclean_preserves_consumed_current_reads_in_frontier() -> None:
    def read_group(call_id: str, path: str, marker: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps(
                                {"source": "workspace", "target": path}
                            ),
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {
                        "path": path,
                        "content": f"{marker} " * 1_000,
                    }
                ),
            },
        ]

    first_group = read_group("read_create", "supportdesk/application/create_user.py", "create")
    second_group = read_group("read_domain", "supportdesk/domain/user.py", "domain")
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "analyze the user lifecycle"},
        *first_group,
        *second_group,
    ]

    first = _preclean_history_for_semantic_compaction(
        messages,
        token_limit=2_000,
        protected_current_token_limit=1_000,
        tool_effects={"read": {"read_only": True}},
    )

    assert first is not None
    rebuilt, details = first
    frontier = next(
        str(message.get("content") or "")
        for message in rebuilt
        if str(message.get("content") or "").startswith(
            CURRENT_TOOL_FRONTIER_HEADING
        )
    )
    assert "supportdesk/application/create_user.py" in frontier
    assert "supportdesk/domain/user.py" not in frontier
    assert details["current_frontier_tokens"] > 0
    assert {
        message.get("tool_call_id")
        for message in rebuilt
        if message.get("role") == "tool"
    } == {"read_domain"}

    third_group = read_group(
        "read_handler",
        "supportdesk/application/change_user_role.py",
        "handler",
    )
    second = _preclean_history_for_semantic_compaction(
        [*rebuilt, *third_group],
        token_limit=2_000,
        protected_current_token_limit=1_000,
        tool_effects={"read": {"read_only": True}},
    )

    assert second is not None
    rebuilt_again, _ = second
    merged_frontier = next(
        str(message.get("content") or "")
        for message in rebuilt_again
        if str(message.get("content") or "").startswith(
            CURRENT_TOOL_FRONTIER_HEADING
        )
    )
    assert "supportdesk/application/create_user.py" in merged_frontier
    assert "supportdesk/domain/user.py" in merged_frontier
    assert {
        message.get("tool_call_id")
        for message in rebuilt_again
        if message.get("role") == "tool"
    } == {"read_handler"}
    assert validate_message_protocol(rebuilt_again) == []


def test_hard_compaction_prefers_two_current_evidence_groups() -> None:
    messages = [
        {"role": "user", "content": "map the existing search contract"},
        {"role": "assistant", "content": "contract detail " * 500},
        {"role": "user", "content": "implement search using all evidence"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_service",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
                {
                    "id": "read_test",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "read_service", "content": "service " * 250},
        {"role": "tool", "tool_call_id": "read_test", "content": "test " * 250},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_repository",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read_repository",
            "content": "repository " * 250,
        },
    ]
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=4_000,
            reserved_output=0,
            soft_limit=0.2,
            hard_limit=0.7,
        )
    )

    prepared = preparer.prepare(
        system_messages=[{"role": "system", "content": "system " * 300}],
        messages=messages,
        tools=[],
    )

    retained_ids = {
        message.get("tool_call_id")
        for message in prepared.request.messages
        if message.get("role") == "tool"
    }
    hard_event = next(
        event
        for event in prepared.compression_events
        if event.details.get("phase") == "hard"
    )
    assert retained_ids == {"read_repository"}
    assert hard_event.details["protected_current_groups"] >= 1
    assert prepared.compaction_update.execution is not None
    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert validate_message_protocol(prepared.request.messages) == []


def test_hard_pressure_falls_back_to_latest_complete_current_group() -> None:
    messages = [
        {"role": "user", "content": "active task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "consumed_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "consumed_read", "content": "x" * 2_000},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "latest_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "latest_read", "content": "latest evidence"},
    ]
    preparer = ContextPreparer(
        TokenBudget(context_budget=650, reserved_output=0, soft_limit=0.25, hard_limit=0.9)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    retained_ids = {
        message.get("tool_call_id")
        for message in prepared.request.messages
        if message.get("role") == "tool"
    }
    assert retained_ids == {"latest_read"}
    latest = next(
        message
        for message in prepared.request.messages
        if message.get("tool_call_id") == "latest_read"
    )
    assert latest["content"] == "latest evidence"
    pressure_event = next(
        event
        for event in prepared.compression_events
        if event.reason == "execution_history"
    )
    assert pressure_event.details["protected_current_groups"] == 1
    assert pressure_event.details["removed_groups"] == 1
    assert validate_message_protocol(prepared.request.messages) == []


def test_soft_compaction_moves_old_unique_commands_into_execution_record():
    messages = _tool_messages(10, tool_name="run_command", chars=500)
    preparer = ContextPreparer(
        TokenBudget(context_budget=2_000, reserved_output=0, soft_limit=0.25, hard_limit=0.95),
        policy=CompactionPolicy(current_frontier_token_limit=120),
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    records = [
        message
        for message in prepared.request.messages
        if str(message.get("content") or "").startswith(COMPACTED_TOOL_HEADING)
    ]
    assert records
    assert any("python -m pytest" in str(record["content"]) for record in records)
    assert any('"returncode":0' in str(record["content"]) for record in records)
    retained_tool_count = len(
        [
            message
            for message in prepared.request.messages
            if message.get("role") == "tool"
        ]
    )
    assert 1 <= retained_tool_count < 10
    assert all(
        _tool_result_status(message) == "passed"
        for message in prepared.request.messages
        if message.get("role") == "tool"
    )
    assert validate_message_protocol(prepared.request.messages) == []


def test_hard_compaction_preserves_multi_tool_groups():
    messages = [{"role": "user", "content": "original task"}]
    for index in range(8):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"read_{index}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": f"search_{index}",
                            "type": "function",
                            "function": {"name": "search_text", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": f"read_{index}", "content": "x" * 800},
                {"role": "tool", "tool_call_id": f"search_{index}", "content": "y" * 800},
            ]
        )
    preparer = ContextPreparer(
        TokenBudget(context_budget=2_000, reserved_output=0, soft_limit=0.2, hard_limit=0.3)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert validate_message_protocol(prepared.request.messages) == []
    for message in prepared.request.messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            expected = {call["id"] for call in message["tool_calls"]}
            actual = {
                candidate["tool_call_id"]
                for candidate in prepared.request.messages[
                    prepared.request.messages.index(message) + 1 :
                ]
                if candidate.get("role") == "tool"
            }
            assert expected <= actual


def test_reactive_compaction_advances_deterministic_state_without_summary_llm():
    messages = _tool_messages(8, tool_name="run_command", chars=500)
    preparer = ContextPreparer()

    state, event = preparer.reactive_compact(messages)
    projected = project_canonical_messages(messages, state)

    assert event is not None
    assert event.details["strategy"] == "stable_tool_group_boundary"
    assert state.execution is not None
    assert any(
        str(message.get("content") or "").startswith(COMPACTED_TOOL_HEADING)
        for message in projected
    )
    assert validate_message_protocol(projected) == []


def test_soft_compaction_preserves_latest_user_task():
    messages = _tool_messages(6, chars=900)
    messages.extend(
        [
            {"role": "assistant", "content": "old turn complete"},
            {"role": "user", "content": "current task: explain only"},
            *_tool_messages(6, chars=900)[1:],
        ]
    )
    preparer = ContextPreparer(
        TokenBudget(context_budget=3_000, reserved_output=0, soft_limit=0.3, hard_limit=0.95)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    users = [message["content"] for message in prepared.request.messages if message["role"] == "user"]
    assert users == ["original task", "current task: explain only"]
    assert {"role": "assistant", "content": "old turn complete"} in prepared.request.messages


def test_hard_compaction_preserves_latest_user_constraints():
    messages = [
        {"role": "user", "content": "first completed task"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "current task: do not modify files"},
        *_tool_messages(8, chars=2_000)[1:],
    ]
    preparer = ContextPreparer(
        TokenBudget(context_budget=1_200, reserved_output=0, soft_limit=0.2, hard_limit=0.5)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert any(
        message.get("role") == "user"
        and message.get("content") == "current task: do not modify files"
        for message in prepared.request.messages
    )
    assert {"role": "user", "content": "first completed task"} in prepared.request.messages
    assert {"role": "assistant", "content": "done"} in prepared.request.messages
    assert prepared.token_estimate <= preparer.budget.hard_token_limit


def test_reactive_compaction_preserves_active_turn():
    messages = [
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "old answer" * 200},
        {"role": "user", "content": "active task"},
        *_tool_messages(8, chars=1_000)[1:],
    ]

    state, event = ContextPreparer().reactive_compact(messages)
    projected = project_canonical_messages(messages, state)

    assert event is not None
    assert any(
        message.get("role") == "user"
        and message.get("content") == "active task"
        for message in projected
    )
    assert any(
        message.get("role") == "user"
        and message.get("content") == "old task"
        for message in projected
    )
    assert any(str(message.get("content") or "").startswith("old answer") for message in projected)


def test_prepare_compacts_consumed_results_below_hard_limit():
    messages = [
        {"role": "user", "content": "active task"},
        *_tool_messages(10, chars=3_000)[1:],
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "latest_small",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "latest_small", "content": "small result"},
    ]
    preparer = ContextPreparer(
        TokenBudget(context_budget=1_000, reserved_output=0, soft_limit=0.2, hard_limit=0.5)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert any(
        message.get("tool_call_id") == "latest_small"
        for message in prepared.request.messages
        if message.get("role") == "tool"
    )


def test_latest_large_tool_result_uses_emergency_receipt_when_needed():
    messages = [
        {"role": "user", "content": "active task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "latest_large",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "latest_large", "content": "x" * 3_000},
    ]
    preparer = ContextPreparer(
        TokenBudget(context_budget=1_000, reserved_output=0, soft_limit=0.2, hard_limit=0.5)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert prepared.compaction_update.execution is not None
    assert not any(
        message.get("role") == "tool"
        for message in prepared.request.messages
    )
    assert any(
        str(message.get("content") or "").startswith(COMPACTED_TOOL_HEADING)
        for message in prepared.request.messages
    )


def test_impossible_prompt_fails_before_provider_call():
    preparer = ContextPreparer(
        TokenBudget(context_budget=200, reserved_output=0, soft_limit=0.5, hard_limit=0.75)
    )

    try:
        preparer.prepare(
            system_messages=[{"role": "system", "content": "s" * 2_000}],
            messages=[{"role": "user", "content": "active task"}],
            tools=[],
        )
    except PromptBudgetExceeded as exc:
        assert exc.token_estimate > exc.hard_token_limit
        assert exc.source_tokens["system"] > exc.source_tokens["messages"]
    else:
        raise AssertionError("expected local prompt budget failure")


def test_repeated_compaction_keeps_active_goal():
    preparer = ContextPreparer(
        TokenBudget(context_budget=1_400, reserved_output=0, soft_limit=0.3, hard_limit=0.8)
    )
    messages = [
        {"role": "user", "content": "old goal"},
        {"role": "assistant", "content": "old result" * 300},
        {"role": "user", "content": "active goal"},
        *_tool_messages(6, chars=1_200)[1:],
    ]

    first = preparer.prepare(system_messages=[], messages=messages, tools=[])
    expanded = [*messages, *_tool_messages(5, chars=1_200)[1:]]
    second = preparer.prepare(
        system_messages=[],
        messages=expanded,
        tools=[],
        compaction_state=first.compaction_update,
    )

    users = [message.get("content") for message in second.request.messages if message.get("role") == "user"]
    assert users == ["old goal", "active goal"]
    assert validate_message_protocol(second.request.messages) == []


def test_soft_compaction_does_not_delete_semantic_turns_without_summary() -> None:
    messages = []
    for index in range(4):
        messages.extend(
            [
                {"role": "user", "content": f"question-{index}"},
                {"role": "assistant", "content": f"answer-{index} " + ("x" * 800)},
            ]
        )
    messages.append({"role": "user", "content": "继续"})
    preparer = ContextPreparer(
        TokenBudget(context_budget=4_000, reserved_output=0, soft_limit=0.2, hard_limit=0.95)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    users = [message["content"] for message in prepared.request.messages if message["role"] == "user"]
    assert users == [
        "question-0",
        "question-1",
        "question-2",
        "question-3",
        "继续",
    ]
    assert all(
        any(
            str(message.get("content") or "").startswith(f"answer-{index}")
            for message in prepared.request.messages
        )
        for index in (0, 1, 2, 3)
    )
    assert not any(
        str(message.get("content") or "").startswith("[MiniCode semantic history]")
        for message in prepared.request.messages
    )


def test_hard_compaction_shortens_semantic_turns_before_deleting_them() -> None:
    messages = [
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "older answer " + ("x" * 2_000)},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "latest answer " + ("y" * 2_000)},
        {"role": "user", "content": "按刚才的方案做"},
    ]
    preparer = ContextPreparer(
        TokenBudget(context_budget=300, reserved_output=0, soft_limit=0.2, hard_limit=0.65)
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    users = [message["content"] for message in prepared.request.messages if message["role"] == "user"]
    assert users == ["latest question", "按刚才的方案做"]
    assert any(
        "[semantic message compacted]" in str(message.get("content") or "")
        for message in prepared.request.messages
        if message.get("role") == "assistant"
    )
    assert prepared.token_estimate <= preparer.budget.hard_token_limit


def test_hard_compaction_reports_retained_semantic_turns() -> None:
    messages: list[dict[str, object]] = []
    for index in range(3):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"question-{index} " + ("u" * 300),
                },
                {
                    "role": "assistant",
                    "content": f"answer-{index} " + ("x" * 1_600),
                },
            ]
        )
    messages.append({"role": "user", "content": "current task"})
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=700,
            reserved_output=0,
            soft_limit=0.2,
            hard_limit=0.65,
        )
    )

    prepared = preparer.prepare(system_messages=[], messages=messages, tools=[])

    hard_event = next(
        event
        for event in prepared.compression_events
        if event.details.get("phase") == "hard"
    )
    assert hard_event.details["semantic_turns_retained"] >= 1
    assert [
        message["content"]
        for message in prepared.request.messages
        if message["role"] == "user"
    ][-1] == "current task"
    assert prepared.token_estimate <= preparer.budget.hard_token_limit


def test_compacted_snapshot_is_stable_without_enough_new_history() -> None:
    messages = _tool_messages(10, chars=1_200)
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=5_000,
            reserved_output=0,
            soft_limit=0.25,
            hard_limit=0.95,
        )
    )

    first = preparer.prepare(system_messages=[], messages=messages, tools=[])
    second = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert first.compression_events
    assert second.compression_events == []
    assert second.request.messages == first.request.messages
    assert all(
        not any(str(key).startswith("_minicode_") for key in message)
        for message in second.request.as_chat_messages()
    )
    assert validate_message_protocol(second.request.messages) == []


def test_compaction_runs_again_after_four_completed_turns() -> None:
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=5_000,
            reserved_output=0,
            soft_limit=0.25,
            hard_limit=0.95,
        )
    )
    first = preparer.prepare(
        system_messages=[],
        messages=_tool_messages(10, chars=1_200),
        tools=[],
    )
    expanded = _tool_messages(10, chars=1_200)
    expanded.extend(_tool_messages(6, chars=1_200)[1:])
    expanded.append({"role": "assistant", "content": "turn complete"})
    expanded.append({"role": "user", "content": "next active task"})

    second = preparer.prepare(
        system_messages=[],
        messages=expanded,
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert any(
        event.reason == "execution_history"
        for event in second.compression_events
    )
    assert [
        message.get("content")
        for message in second.request.messages
        if message.get("role") == "user"
    ][-1] == "next active task"
    assert validate_message_protocol(second.request.messages) == []
