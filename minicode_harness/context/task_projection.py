"""Compaction-time projection for one Run's current task state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from .message_groups import (
    find_latest_user_group_index,
    flatten_groups,
    group_messages,
    validate_message_protocol,
)


TASK_TOOL_NAMES = frozenset({"task"})
CURRENT_TASK_HEADING = "[MiniCode current task]"
CURRENT_TASK_END = "[/MiniCode current task]"
_MAX_REQUEST_CHARS = 240


def strip_task_protocol(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove Task Tool sub-protocols and stale task projections."""

    filtered_groups: list[list[dict[str, Any]]] = []
    for group in group_messages(messages):
        if len(group) == 1 and _is_current_task_record(group[0]):
            continue

        owner = group[0]
        tool_calls = (
            list(owner.get("tool_calls") or [])
            if owner.get("role") == "assistant"
            else []
        )
        task_call_ids = {
            str(call.get("id"))
            for call in tool_calls
            if _tool_call_name(call) in TASK_TOOL_NAMES
            and call.get("id") is not None
        }
        if not task_call_ids:
            filtered_groups.append(deepcopy(group))
            continue

        remaining_calls = [
            deepcopy(call)
            for call in tool_calls
            if str(call.get("id")) not in task_call_ids
        ]
        if not remaining_calls:
            continue

        filtered_owner = deepcopy(owner)
        filtered_owner["tool_calls"] = remaining_calls
        filtered_group = [filtered_owner]
        filtered_group.extend(
            deepcopy(message)
            for message in group[1:]
            if not (
                message.get("role") == "tool"
                and str(message.get("tool_call_id")) in task_call_ids
            )
        )
        filtered_groups.append(filtered_group)

    filtered = flatten_groups(filtered_groups)
    errors = validate_message_protocol(filtered)
    if errors:
        raise ValueError("Invalid message protocol after Task filtering: " + "; ".join(errors))
    return filtered


def render_current_task_record(
    user_task: str,
    open_rows: Sequence[Sequence[str]],
) -> str:
    """Render one bounded current-task record after history compaction."""

    request = _compact_text(user_task, max_chars=_MAX_REQUEST_CHARS)
    lines = [CURRENT_TASK_HEADING, f"Request: {request}", "", "Open steps:"]
    for task_id, status, subject in open_rows:
        display_id = f"#{task_id}" if str(task_id).isdigit() else str(task_id)
        lines.append(
            f"- {display_id} [{status}] {_compact_text(str(subject), max_chars=120)}"
        )
    lines.append(CURRENT_TASK_END)
    return "\n".join(lines)


def insert_current_task_record(
    messages: list[dict[str, Any]],
    *,
    user_task: str,
    open_rows: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    """Replace stale projections and insert one record after the latest User."""

    stripped = strip_task_protocol(messages)
    if not open_rows:
        return stripped

    groups = group_messages(stripped)
    active_index = find_latest_user_group_index(groups)
    if active_index is None:
        return stripped

    groups.insert(
        active_index + 1,
        [
            {
                "role": "assistant",
                "content": render_current_task_record(user_task, open_rows),
            }
        ],
    )
    rebuilt = flatten_groups(groups)
    errors = validate_message_protocol(rebuilt)
    if errors:
        raise ValueError("Invalid message protocol after Task projection: " + "; ".join(errors))
    return rebuilt


def _tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "")


def _is_current_task_record(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant" or message.get("tool_calls"):
        return False
    content = str(message.get("content") or "").strip()
    return content.startswith(CURRENT_TASK_HEADING) and content.endswith(CURRENT_TASK_END)


def _compact_text(value: str, *, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."
