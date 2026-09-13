from minicode_harness.context.message_groups import (
    bounded_messages,
    find_latest_user_group_index,
    group_messages,
    is_final_assistant_message,
    is_synthetic_assistant_message,
    select_recent_semantic_turns,
    validate_message_protocol,
)


def _history():
    return [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]


def test_assistant_message_classification_uses_one_synthetic_heading_set() -> None:
    headings = (
        "[MiniCode compacted execution]",
        "[MiniCode semantic history]",
        "[MiniCode current tool frontier]",
        "[MiniCode current task]",
        "[MiniCode runtime notification]",
    )

    for heading in headings:
        message = {"role": "assistant", "content": f"{heading}\nrecord"}
        assert is_synthetic_assistant_message(message) is True
        assert is_final_assistant_message(message) is False

    final_message = {"role": "assistant", "content": "已完成聚焦修复。"}
    assert is_synthetic_assistant_message(final_message) is False
    assert is_final_assistant_message(final_message) is True
    assert is_final_assistant_message(
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]}
    ) is False


def test_group_messages_keeps_multi_tool_exchange_indivisible() -> None:
    groups = group_messages(_history())

    assert [len(group) for group in groups] == [1, 3, 1, 1]
    assert validate_message_protocol(_history()) == []


def test_bounded_messages_never_starts_inside_tool_results() -> None:
    bounded = bounded_messages(_history(), 5)

    assert bounded[0]["role"] == "assistant"
    assert bounded[0]["tool_calls"][0]["id"] == "a"
    assert [message["role"] for message in bounded] == ["assistant", "tool", "tool", "assistant", "user"]
    assert validate_message_protocol(bounded) == []


def test_select_recent_semantic_turn_with_multiple_tool_calls() -> None:
    groups = group_messages(_history())
    active_index = find_latest_user_group_index(groups)

    turns = select_recent_semantic_turns(groups, active_user_index=active_index)

    assert turns == [(0, 2)]
    assert groups[turns[0][0]][0]["content"] == "start"
    assert groups[turns[0][1]][0]["content"] == "done"


def test_select_recent_semantic_turns_skips_incomplete_and_compacted_answers() -> None:
    messages = []
    for index in range(4):
        messages.extend(
            [
                {"role": "user", "content": f"user-{index}"},
                {"role": "assistant", "content": f"answer-{index}"},
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": "incomplete"},
            {
                "role": "assistant",
                "content": "[MiniCode compacted execution]\n\n- write_file | a.py",
            },
            {
                "role": "assistant",
                "content": "[MiniCode current tool frontier]\n\n- read_file | a.py",
            },
            {"role": "user", "content": "continue"},
        ]
    )
    groups = group_messages(messages)

    turns = select_recent_semantic_turns(groups, max_turns=3)

    assert [groups[user][0]["content"] for user, _ in turns] == [
        "user-1",
        "user-2",
        "user-3",
    ]
    assert [groups[assistant][0]["content"] for _, assistant in turns] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
