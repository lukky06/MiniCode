"""Trace-derived SWE-bench efficiency metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minicode_harness.report import collect_context_metrics


_WRITE_TOOLS = {"edit", "write", "apply_patch"}
_INVESTIGATION_TOOLS = {"read", "search"}


def read_trace_events(path: Path | str) -> list[dict[str, Any]]:
    """Read valid JSONL trace events and ignore a final partial line."""

    target = Path(path)
    if not target.is_file():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def collect_swebench_metrics(
    trace_path: Path | str,
    *,
    run_path: Path | None = None,
) -> dict[str, Any]:
    """Collect stable runtime metrics without changing AgentLoop state."""

    events = read_trace_events(trace_path)
    context = collect_context_metrics(events, run_path=run_path)
    input_tokens = 0
    output_tokens = 0
    for event in events:
        if event.get("type") != "model_response":
            continue
        usage = event.get("usage") or {}
        input_tokens += int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens += int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
    tool_calls = [event for event in events if event.get("type") == "tool_called"]
    convergence = _collect_convergence_metrics(events)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "avg_context_tokens": context.context_token_estimate_avg,
        "max_context_tokens": context.context_token_estimate_max,
        "read_tool_calls": context.workspace_read_calls + context.artifact_read_calls,
        "unique_read_resources": context.unique_read_resources,
        "repeated_read_calls": context.repeated_read_calls,
        "project_cache_hits": context.project_cache_hit_count,
        "model_recoveries": (
            context.model_retry_count
            + context.reactive_compaction_count
            + context.output_recovery_count
        ),
        "context_compactions": sum(
            1 for event in events if event.get("type") == "context_compressed"
        ),
        "verification_commands": sum(
            1 for event in tool_calls if event.get("tool") == "run_command"
        ),
        "command_policy_rejections": sum(
            1
            for event in events
            if event.get("type") == "hook_decision"
            and event.get("hook") == "command_policy_guard"
        ),
        "checkpoint_count": sum(
            1 for event in events if event.get("type") == "checkpoint_saved"
        ),
        "subagent_calls": context.subagent_call_count,
        "mcp_tool_calls": context.mcp_tool_call_count,
        **convergence,
    }


def _collect_convergence_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive lightweight edit/verification convergence signals from Trace order."""

    calls_by_id: dict[str, dict[str, Any]] = {}
    categories_by_id: dict[str, str] = {}
    first_edit_index: int | None = None
    first_edit_step: int | None = None

    for index, event in enumerate(events):
        event_type = event.get("type")
        call_id = str(event.get("tool_call_id") or "")
        if event_type == "tool_called" and call_id:
            calls_by_id[call_id] = event
        elif event_type == "tool_arguments_validated" and call_id:
            category = event.get("command_category")
            if isinstance(category, str) and category:
                categories_by_id[call_id] = category
        elif (
            first_edit_index is None
            and event_type == "run_state_updated"
            and event.get("tool") in _WRITE_TOOLS
            and int(event.get("modified_files") or 0) > 0
        ):
            first_edit_index = index
            first_edit_step = _optional_int(event.get("step"))

    if first_edit_index is None:
        return {
            "first_edit_step": None,
            "non_write_calls_before_first_edit": sum(
                1
                for event in events
                if event.get("type") == "tool_called"
                and event.get("tool") not in _WRITE_TOOLS
            ),
            "reproducer_to_edit_delay": None,
            "target_test_reuse_rate": None,
            "post_edit_diagnostic_calls": 0,
            "root_cause_switch_after_failed_verification": None,
        }

    pre_edit_calls = [
        event
        for event in events[:first_edit_index]
        if event.get("type") == "tool_called"
    ]
    non_write_before = sum(
        1 for event in pre_edit_calls if event.get("tool") not in _WRITE_TOOLS
    )
    pre_edit_diagnostic_steps = [
        _optional_int(event.get("step"))
        for event in pre_edit_calls
        if event.get("tool") == "run_command"
        and categories_by_id.get(str(event.get("tool_call_id") or ""))
        in {"diagnostic", "verification"}
    ]
    last_reproducer_step = next(
        (step for step in reversed(pre_edit_diagnostic_steps) if step is not None),
        None,
    )
    reproducer_delay = (
        first_edit_step - last_reproducer_step
        if first_edit_step is not None and last_reproducer_step is not None
        else None
    )

    pre_edit_verifications = {
        _command_identity(event)
        for event in pre_edit_calls
        if event.get("tool") == "run_command"
        and categories_by_id.get(str(event.get("tool_call_id") or "")) == "verification"
        and _command_identity(event) is not None
    }
    post_edit_calls = [
        event
        for event in events[first_edit_index + 1 :]
        if event.get("type") == "tool_called"
    ]
    post_edit_verifications = [
        identity
        for event in post_edit_calls
        if event.get("tool") == "run_command"
        and categories_by_id.get(str(event.get("tool_call_id") or "")) == "verification"
        if (identity := _command_identity(event)) is not None
    ]
    target_test_reuse_rate = (
        round(
            sum(identity in pre_edit_verifications for identity in post_edit_verifications)
            / len(post_edit_verifications),
            6,
        )
        if post_edit_verifications
        else None
    )
    post_edit_diagnostics = sum(
        1
        for event in post_edit_calls
        if event.get("tool") in _INVESTIGATION_TOOLS
        or (
            event.get("tool") == "run_command"
            and categories_by_id.get(str(event.get("tool_call_id") or "")) == "diagnostic"
        )
    )

    root_cause_switch = _root_cause_switch_proxy(
        events,
        first_edit_index=first_edit_index,
        calls_by_id=calls_by_id,
        categories_by_id=categories_by_id,
    )
    return {
        "first_edit_step": first_edit_step,
        "non_write_calls_before_first_edit": non_write_before,
        "reproducer_to_edit_delay": reproducer_delay,
        "target_test_reuse_rate": target_test_reuse_rate,
        "post_edit_diagnostic_calls": post_edit_diagnostics,
        "root_cause_switch_after_failed_verification": root_cause_switch,
    }


def _root_cause_switch_proxy(
    events: list[dict[str, Any]],
    *,
    first_edit_index: int,
    calls_by_id: dict[str, dict[str, Any]],
    categories_by_id: dict[str, str],
) -> bool | None:
    """Return whether edit scope changed after the first failed post-edit verification.

    Trace cannot observe a model's private hypothesis directly. A changed successful
    edit scope is therefore used as a deterministic proxy for a root-cause switch.
    """

    failure_index: int | None = None
    for index, event in enumerate(events[first_edit_index + 1 :], start=first_edit_index + 1):
        call_id = str(event.get("tool_call_id") or "")
        if (
            event.get("type") == "run_state_updated"
            and event.get("tool") == "run_command"
            and categories_by_id.get(call_id) == "verification"
            and event.get("verification_status") == "failed"
        ):
            failure_index = index
            break
    if failure_index is None:
        return None

    scopes_before = {
        scope
        for event in events[:failure_index]
        if event.get("type") == "run_state_updated"
        and event.get("tool") in _WRITE_TOOLS
        and int(event.get("modified_files") or 0) > 0
        if (scope := _write_scope(calls_by_id.get(str(event.get("tool_call_id") or ""))))
        is not None
    }
    for event in events[failure_index + 1 :]:
        if (
            event.get("type") != "run_state_updated"
            or event.get("tool") not in _WRITE_TOOLS
            or int(event.get("modified_files") or 0) <= 0
        ):
            continue
        scope = _write_scope(calls_by_id.get(str(event.get("tool_call_id") or "")))
        return scope is not None and scope not in scopes_before
    return False


def _command_identity(event: dict[str, Any]) -> tuple[str, ...] | None:
    args = event.get("args") or {}
    argv = args.get("argv") if isinstance(args, dict) else None
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        return None
    return tuple(argv)


def _write_scope(event: dict[str, Any] | None) -> tuple[str, str, str] | None:
    if not isinstance(event, dict):
        return None
    tool = str(event.get("tool") or "")
    args = event.get("args") or {}
    if not isinstance(args, dict):
        return None
    path = str(args.get("path") or args.get("target") or "")
    anchor = str(args.get("old_text") or args.get("oldText") or args.get("patch") or "")
    if not path and not anchor:
        return None
    return tool, path, anchor[:160]


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
