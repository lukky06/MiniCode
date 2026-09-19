import json

from minicode_harness.context import CompactionPolicy, ContextPreparer, TokenBudget
from minicode_harness.context.session_projection import COMPACTED_TOOL_HEADING


def _tool_history(count: int, *, chars: int = 1_200) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "inspect the repository"}]
    for index in range(count):
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
                                "name": "read_file",
                                "arguments": json.dumps(
                                    {"path": f"src/service_{index}.py"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": "x" * chars,
                },
            ]
        )
    return messages


def _preparer() -> ContextPreparer:
    return ContextPreparer(
        TokenBudget(
            context_budget=5_000,
            reserved_output=0,
            soft_limit=0.25,
            hard_limit=0.95,
        ),
        policy=CompactionPolicy(current_frontier_token_limit=160),
    )


def test_unchanged_boundary_reuses_byte_stable_projection() -> None:
    messages = _tool_history(10)
    preparer = _preparer()
    first = preparer.prepare(system_messages=[], messages=messages, tools=[])

    second = preparer.prepare(
        system_messages=[],
        messages=messages,
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert first.compaction_update.execution is not None
    assert second.compaction_update == first.compaction_update
    assert second.request.messages == first.request.messages
    assert second.request.model_dump_json() == first.request.model_dump_json()
    assert second.compression_events == []


def test_new_tail_keeps_existing_model_prefix_unchanged() -> None:
    messages = _tool_history(10)
    preparer = _preparer()
    first = preparer.prepare(system_messages=[], messages=messages, tools=[])
    expanded = [
        *messages,
        {"role": "assistant", "content": "inspection complete"},
        {"role": "user", "content": "explain the result briefly"},
    ]

    second = preparer.prepare(
        system_messages=[],
        messages=expanded,
        tools=[],
        compaction_state=first.compaction_update,
    )

    assert second.compaction_update == first.compaction_update
    assert second.request.messages[: len(first.request.messages)] == first.request.messages
    assert second.request.messages[-2:] == expanded[-2:]
    assert second.compression_events == []


def test_new_pressure_advances_boundary_once() -> None:
    messages = _tool_history(10)
    preparer = _preparer()
    first = preparer.prepare(system_messages=[], messages=messages, tools=[])
    expanded = [*messages, *_tool_history(12, chars=1_500)[1:]]

    second = preparer.prepare(
        system_messages=[],
        messages=expanded,
        tools=[],
        compaction_state=first.compaction_update,
    )
    third = preparer.prepare(
        system_messages=[],
        messages=expanded,
        tools=[],
        compaction_state=second.compaction_update,
    )

    assert second.compaction_update.execution is not None
    assert second.compaction_update.execution != first.compaction_update.execution
    assert any(
        event.details.get("strategy") == "stable_tool_group_boundary"
        for event in second.compression_events
    )
    assert third.compaction_update == second.compaction_update
    assert third.request.messages == second.request.messages
    assert third.compression_events == []


def test_projection_state_never_enters_canonical_messages() -> None:
    messages = _tool_history(10)
    original = json.loads(json.dumps(messages))

    prepared = _preparer().prepare(
        system_messages=[],
        messages=messages,
        tools=[],
    )

    assert messages == original
    assert any(
        str(message.get("content") or "").startswith(COMPACTED_TOOL_HEADING)
        for message in prepared.request.messages
    )
    assert all(
        not any(str(key).startswith("_minicode_") for key in message)
        for message in messages
    )
