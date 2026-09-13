import json

from minicode_harness.context import (
    ContextPreparer,
    TokenBudget,
    validate_message_protocol,
)
from minicode_harness.context.session_projection import COMPACTED_TOOL_HEADING


def _tool_group(
    index: int,
    *,
    tool_name: str,
    arguments: dict,
    result: dict | str,
) -> list[dict]:
    tool_call_id = f"call_{index}"
    content = result if isinstance(result, str) else json.dumps(result)
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
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
            "tool_call_id": tool_call_id,
            "content": content,
        },
    ]


def test_extreme_tool_history_remains_bounded_and_protocol_safe() -> None:
    messages: list[dict] = [
        {"role": "user", "content": "old requirement " + ("r" * 1_000)},
        {"role": "assistant", "content": "old decision " + ("d" * 1_000)},
        {"role": "user", "content": "current task must remain"},
    ]
    call_index = 0
    for index in range(50):
        messages.extend(
            _tool_group(
                call_index,
                tool_name="read",
                arguments={"source": "workspace", "target": f"src/read_{index}.py"},
                result="source " * 250,
            )
        )
        call_index += 1
        messages.extend(
            _tool_group(
                call_index,
                tool_name="search",
                arguments={
                    "source": "workspace",
                    "kind": "text",
                    "query": f"symbol_{index}",
                    "path": "src",
                },
                result="match " * 180,
            )
        )
        call_index += 1
    for index in range(50):
        messages.extend(
            _tool_group(
                call_index,
                tool_name="write",
                arguments={
                    "path": f"generated/file_{index}.py",
                    "content": "value = 1\n" * 180,
                },
                result={"status": "ok", "path": f"generated/file_{index}.py"},
            )
        )
        call_index += 1
    for index in range(50):
        messages.extend(
            _tool_group(
                call_index,
                tool_name="run_command",
                arguments={
                    "argv": [
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        f"tests/test_{index}.py",
                    ]
                },
                result={
                    "status": "passed",
                    "returncode": 0,
                    "stdout": "passed " + ("x" * 600),
                },
            )
        )
        call_index += 1

    preparer = ContextPreparer(
        TokenBudget(
            context_budget=9_000,
            reserved_output=0,
            soft_limit=0.20,
            hard_limit=0.75,
        )
    )
    prepared = preparer.prepare(
        system_messages=[{"role": "system", "content": "system " * 100}],
        messages=messages,
        tools=[],
        tool_effects={
            "read": {
                "read_only": True,
                "destructive": False,
                "result_reconstructible": True,
            },
            "search": {
                "read_only": True,
                "destructive": False,
                "result_reconstructible": True,
            },
            "write": {
                "read_only": False,
                "destructive": False,
                "result_reconstructible": True,
            },
            "run_command": {
                "read_only": False,
                "destructive": False,
                "result_reconstructible": False,
            },
        },
    )

    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert any(
        message.get("role") == "user"
        and message.get("content") == "current task must remain"
        for message in prepared.request.messages
    )
    assert validate_message_protocol(prepared.request.messages) == []
    assert any(
        str(message.get("content") or "").startswith(COMPACTED_TOOL_HEADING)
        for message in prepared.request.messages
    )
    assert not any(
        str(message.get("content") or "").startswith("[MiniCode semantic history]")
        for message in prepared.request.messages
    )
    assert any(
        event.reason == "execution_history"
        and event.details.get("strategy") == "stable_tool_group_boundary"
        for event in prepared.compression_events
    )
    assert len(prepared.request.messages) < len(messages)
    assert any(
        message.get("tool_call_id") == "call_199"
        for message in prepared.request.messages
        if message.get("role") == "tool"
    )
