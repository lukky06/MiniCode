from minicode_harness.state import UserTurnState


def test_user_turn_preserves_current_canonical_history_and_appends_task() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"source":"workspace","kind":"text","query":"RedisLease","path":"src"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "content": "{}",
        },
    ]

    state = UserTurnState.create(
        turn_id="run_1",
        task="continue",
        messages=messages,
    )

    assert state.messages[:-1] == messages
    assert state.messages[-1] == {"role": "user", "content": "continue"}
