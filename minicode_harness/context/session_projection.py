"""Derive a stable model-visible projection from canonical Session messages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .compaction_state import (
    ExecutionCompactionState,
    SemanticCompactionState,
    SessionCompactionState,
)
from .message_groups import MessageGroup, flatten_groups, group_messages


COMPACTED_TOOL_HEADING = "[MiniCode compacted tool]"
COMPACTION_POLICY_VERSION = 1
_MAX_TARGET_CHARS = 180
_MAX_ERROR_MESSAGE_CHARS = 240
_FAILURE_MARKERS = (
    "error",
    "failed",
    "failure",
    "denied",
    "rejected",
    "blocked",
    "invalid",
    "timeout",
    "timed_out",
)


@dataclass(frozen=True)
class CanonicalGroup:
    index: int
    group_id: str
    group: MessageGroup


def canonical_groups(messages: list[dict[str, Any]]) -> list[CanonicalGroup]:
    """Build append-stable IDs for complete canonical Message Groups."""

    records: list[CanonicalGroup] = []
    for index, group in enumerate(group_messages(messages)):
        digest = hashlib.sha256(_canonical_json(group).encode("utf-8")).hexdigest()[:16]
        records.append(
            CanonicalGroup(
                index=index,
                group_id=f"g{index:08d}-{digest}",
                group=group,
            )
        )
    return records


def canonical_prefix_digest(
    groups: list[CanonicalGroup],
    *,
    end_exclusive: int,
) -> str:
    """Hash one immutable canonical prefix for state validation."""

    payload = [record.group for record in groups[: max(0, end_exclusive)]]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def execution_state_for_boundary(
    groups: list[CanonicalGroup],
    *,
    boundary_index: int,
) -> ExecutionCompactionState | None:
    if boundary_index < 0 or boundary_index >= len(groups):
        return None
    return ExecutionCompactionState(
        boundary_group_id=groups[boundary_index].group_id,
        source_digest=canonical_prefix_digest(
            groups,
            end_exclusive=boundary_index + 1,
        ),
        policy_version=COMPACTION_POLICY_VERSION,
    )


def semantic_state_for_boundary(
    groups: list[CanonicalGroup],
    *,
    first_kept_index: int,
    summary: str,
) -> SemanticCompactionState | None:
    compact_summary = str(summary or "").strip()
    if (
        not compact_summary
        or first_kept_index < 0
        or first_kept_index >= len(groups)
    ):
        return None
    return SemanticCompactionState(
        summary=compact_summary,
        first_kept_group_id=groups[first_kept_index].group_id,
        source_digest=canonical_prefix_digest(
            groups,
            end_exclusive=first_kept_index,
        ),
        policy_version=COMPACTION_POLICY_VERSION,
    )


def valid_execution_boundary_index(
    groups: list[CanonicalGroup],
    state: ExecutionCompactionState | None,
) -> int | None:
    if state is None or state.policy_version != COMPACTION_POLICY_VERSION:
        return None
    index = _group_index(groups, state.boundary_group_id)
    if index is None:
        return None
    expected = canonical_prefix_digest(groups, end_exclusive=index + 1)
    return index if expected == state.source_digest else None


def valid_semantic_boundary_index(
    groups: list[CanonicalGroup],
    state: SemanticCompactionState | None,
) -> int | None:
    if state is None or state.policy_version != COMPACTION_POLICY_VERSION:
        return None
    index = _group_index(groups, state.first_kept_group_id)
    if index is None:
        return None
    expected = canonical_prefix_digest(groups, end_exclusive=index)
    return index if expected == state.source_digest else None


def project_canonical_messages(
    messages: list[dict[str, Any]],
    state: SessionCompactionState | None,
) -> list[dict[str, Any]]:
    """Apply persisted semantic and deterministic execution projection state."""

    groups = canonical_groups(messages)
    if not groups:
        return []
    effective = state or SessionCompactionState()
    semantic_index = valid_semantic_boundary_index(groups, effective.semantic)
    execution_index = valid_execution_boundary_index(groups, effective.execution)

    projected: list[dict[str, Any]] = []
    start_index = 0
    if semantic_index is not None and effective.semantic is not None:
        projected.append(
            {
                "role": "assistant",
                "content": effective.semantic.summary,
            }
        )
        start_index = semantic_index

    for record in groups[start_index:]:
        if execution_index is not None and record.index <= execution_index:
            receipt = render_tool_group_receipt(record.group)
            if receipt is not None:
                projected.append(receipt)
                continue
        projected.extend(flatten_groups([record.group]))
    return projected


def render_tool_group_receipt(group: MessageGroup) -> dict[str, Any] | None:
    """Render one deterministic compact receipt in the original Tool Group slot."""

    if not group:
        return None
    owner = group[0]
    if owner.get("role") != "assistant" or not owner.get("tool_calls"):
        return None
    results = {
        str(message.get("tool_call_id") or ""): message
        for message in group[1:]
        if message.get("role") == "tool"
    }
    calls: list[dict[str, Any]] = []
    for call in owner.get("tool_calls") or []:
        function = call.get("function") or {}
        call_id = str(call.get("id") or "")
        arguments = _parse_arguments(function.get("arguments"))
        receipt: dict[str, Any] = {
            "name": str(function.get("name") or "unknown"),
            "status": "missing",
        }
        target = _argument_target(arguments)
        if target:
            receipt["target"] = target
        result = results.get(call_id)
        if result is not None:
            receipt.update(_compact_result(str(result.get("content") or "")))
        calls.append(receipt)
    payload = {"v": COMPACTION_POLICY_VERSION, "calls": calls}
    return {
        "role": "assistant",
        "content": f"{COMPACTED_TOOL_HEADING}\n{_canonical_json(payload)}",
    }


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _argument_target(arguments: dict[str, Any]) -> str:
    for key in ("path", "target", "query", "source", "task", "subject"):
        value = arguments.get(key)
        if value not in (None, "", []):
            return _single_line(value, _MAX_TARGET_CHARS)
    argv = arguments.get("argv")
    if isinstance(argv, list):
        return _single_line(" ".join(map(str, argv)), _MAX_TARGET_CHARS)
    return ""


def _compact_result(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        compact = " ".join(content.split())
        failed = any(marker in compact.lower() for marker in _FAILURE_MARKERS)
        result: dict[str, Any] = {
            "status": "failed" if failed else "ok" if compact else "empty"
        }
        if failed:
            result["message"] = compact[:_MAX_ERROR_MESSAGE_CHARS]
        return result

    payload = parsed.get("payload")
    nested = payload if isinstance(payload, dict) else {}
    status = str(
        parsed.get("status") or nested.get("status") or "ok"
    ).strip().lower()
    result: dict[str, Any] = {"status": status}
    for key in ("returncode", "timed_out", "artifact_path", "match_count"):
        value = parsed.get(key, nested.get(key))
        if value not in (None, "", []):
            result[key] = value
    modified = parsed.get("modified_files", nested.get("modified_files"))
    if isinstance(modified, list) and modified:
        result["modified"] = [
            _single_line(item, _MAX_TARGET_CHARS) for item in modified[:4]
        ]
    if status not in {"ok", "success", "passed", "empty"}:
        message = (
            parsed.get("message")
            or parsed.get("error")
            or nested.get("message")
            or nested.get("error")
            or parsed.get("reason")
            or nested.get("reason")
        )
        if message:
            result["message"] = _single_line(message, _MAX_ERROR_MESSAGE_CHARS)
    return result


def _single_line(value: Any, limit: int) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _group_index(groups: list[CanonicalGroup], group_id: str) -> int | None:
    for record in groups:
        if record.group_id == group_id:
            return record.index
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        deepcopy(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
