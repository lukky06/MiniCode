"""Trace display and run report generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from minicode_harness.policy import render_argv
from minicode_harness.state import RunSession, RunStore
from minicode_harness.tools import inspect_git_diff


@dataclass(frozen=True)
class RunReportResult:
    """Generated run report paths."""

    run_id: str
    report_path: Path
    final_diff_path: Path


@dataclass(frozen=True)
class ContextMetrics:
    """Context and tool-use metrics derived from a run trace."""

    total_tool_calls: int
    workspace_read_calls: int
    artifact_read_calls: int
    text_search_calls: int
    unique_read_resources: int
    repeated_read_calls: int
    context_token_estimate_avg: int
    context_token_estimate_max: int
    context_window: int
    prompt_budget: int
    reserved_output: int
    budget_usage_ratio_avg: float
    budget_usage_ratio_max: float
    history_groups_compacted: int
    token_estimator_version: str
    model_retry_count: int
    reactive_compaction_count: int
    provider_overflow_count: int
    calibration_sample_count: int
    calibration_ratio_p50: float
    calibration_ratio_p95: float
    calibration_relative_error_p50: float
    calibration_relative_error_p95: float
    output_recovery_count: int
    subagent_call_count: int
    mcp_tool_call_count: int
    history_compaction_tokens_removed: int
    history_compaction_tokens_retained: int
    modified_files_count: int
    final_status: str
    stop_reason: str


def format_run_trace(run_id: str, run_store: RunStore | None = None) -> str:
    """Return a compact human-readable trace for a run."""

    store = run_store or RunStore()
    run_path = _existing_run_path(store, run_id)
    events = _read_trace_events(run_path / "trace.jsonl")
    if not events:
        return f"Run Trace: {run_id}\nNo trace events found."

    lines = [f"Run Trace: {run_id}", ""]
    for index, event in enumerate(events, start=1):
        lines.append(_format_trace_event(index, event))
    return "\n".join(lines)


def format_context_usage(run_id: str, run_store: RunStore | None = None) -> str:
    """Render the latest prepared-request budget from one run trace."""

    store = run_store or RunStore()
    run_path = _existing_run_path(store, run_id)
    events = _read_trace_events(run_path / "trace.jsonl")
    context_events = [
        event for event in events if event.get("type") == "context_built"
    ]
    if not context_events:
        return f"Context: {run_id}\nNo context build events found."

    latest = context_events[-1]
    token_estimate = int(latest.get("token_estimate") or 0)
    prompt_budget = int(latest.get("prompt_budget") or 0)
    ratio = (
        token_estimate / prompt_budget
        if prompt_budget > 0
        else 0.0
    )
    source_tokens = latest.get("source_tokens")
    if not isinstance(source_tokens, dict):
        source_tokens = {}
    labels = (
        ("system", "System Prompt"),
        ("tools", "Tool Schemas"),
        ("historical_messages", "Historical Messages"),
        ("current_turn", "Current Turn"),
        ("semantic_history", "Semantic History"),
        ("execution_record", "Execution Record"),
        ("current_tool_frontier", "Current Tool Frontier"),
        ("task_projection", "Task Projection"),
        ("runtime_notifications", "Runtime Notifications"),
    )
    compactions = [
        event for event in events if event.get("type") == "context_compressed"
    ]
    last_compaction = "none"
    if compactions:
        details = compactions[-1].get("details") or {}
        last_compaction = str(
            details.get("phase")
            or compactions[-1].get("reason")
            or "unknown"
        )
    overflow_count = sum(
        1
        for event in events
        if event.get("type") == "model_recovery"
        and event.get("action") == "reactive_compact"
    )
    calibrations = [
        event
        for event in events
        if event.get("type") == "token_estimate_calibration"
    ]

    lines = [
        f"Context: {run_id}",
        f"Usage: {_format_token_count(token_estimate)} / "
        f"{_format_token_count(prompt_budget)} ({ratio:.1%})",
        "",
    ]
    lines.extend(
        f"{label:<24} {_format_token_count(int(source_tokens.get(key) or 0)):>8}"
        for key, label in labels
    )
    lines.extend(
        [
            "",
            f"Last compaction: {last_compaction}",
            f"Provider overflow: {overflow_count}",
            f"Estimator: {latest.get('token_estimator_version') or 'unknown'}",
        ]
    )
    if calibrations:
        latest_calibration = calibrations[-1]
        lines.append(
            "Latest calibration: "
            f"{float(latest_calibration.get('ratio') or 0.0):.3f}x "
            f"({int(latest_calibration.get('estimated_prompt_tokens') or 0)} estimated, "
            f"{int(latest_calibration.get('provider_prompt_tokens') or 0)} provider)"
        )
    else:
        lines.append("Latest calibration: unavailable")
    return "\n".join(lines)


def generate_run_report(run_id: str, run_store: RunStore | None = None) -> RunReportResult:
    """Generate ``report.md`` and ``final.diff`` for one run."""

    store = run_store or RunStore()
    session = store.load_session(run_id)
    run_path = _existing_run_path(store, run_id)
    events = _read_trace_events(run_path / "trace.jsonl")
    final_diff = _workspace_diff(session)
    final_diff_path = run_path / "final.diff"
    final_diff_path.write_text(final_diff, encoding="utf-8")

    report_path = run_path / "report.md"
    report_path.write_text(
        _render_report(
            session=session,
            events=events,
            final_diff=final_diff,
            run_path=run_path,
        ),
        encoding="utf-8",
    )
    return RunReportResult(
        run_id=run_id,
        report_path=report_path,
        final_diff_path=final_diff_path,
    )


def collect_context_metrics(
    events: list[dict[str, Any]],
    *,
    run_path: Path | None = None,
) -> ContextMetrics:
    """Collect context-management metrics from trace events."""

    tool_calls = [event for event in events if event.get("type") == "tool_called"]
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    context_events = [event for event in events if event.get("type") == "context_built"]
    token_estimates = [
        int(event.get("token_estimate") or 0)
        for event in context_events
        if event.get("token_estimate") is not None
    ]
    budget_usage_ratios = [
        float(event.get("budget_usage_ratio") or 0.0)
        for event in context_events
        if event.get("budget_usage_ratio") is not None
    ]
    latest_context = context_events[-1] if context_events else {}
    status = _final_status(events)
    history_compressions = [
        event
        for event in events
        if event.get("type") == "context_compressed"
    ]
    changed_files = (
        _changed_files(events, run_path)
        if run_path is not None
        else _changed_files_from_events(events)
    )
    read_fingerprints = [
        fingerprint
        for event in tool_calls
        if (fingerprint := _read_tool_fingerprint(event)) is not None
    ]
    unique_read_resources = len(set(read_fingerprints))
    calibrations = [
        event
        for event in events
        if event.get("type") == "token_estimate_calibration"
    ]
    calibration_ratios = [
        float(event["ratio"])
        for event in calibrations
        if isinstance(event.get("ratio"), (int, float))
        and not isinstance(event.get("ratio"), bool)
    ]
    calibration_relative_errors = [
        float(event["relative_error"])
        for event in calibrations
        if isinstance(event.get("relative_error"), (int, float))
        and not isinstance(event.get("relative_error"), bool)
    ]
    provider_overflow_count = sum(
        1
        for event in events
        if event.get("type") == "model_recovery"
        and event.get("action") == "reactive_compact"
    )
    return ContextMetrics(
        total_tool_calls=len(tool_calls),
        workspace_read_calls=sum(
            1 for event in tool_calls if _event_read_source(event) == "workspace"
        ),
        artifact_read_calls=sum(
            1 for event in tool_calls if _event_read_source(event) == "artifact"
        ),
        text_search_calls=sum(
            1 for event in tool_calls if _event_search_kind(event) == "text"
        ),
        unique_read_resources=unique_read_resources,
        repeated_read_calls=max(0, len(read_fingerprints) - unique_read_resources),
        context_token_estimate_avg=(
            int(sum(token_estimates) / len(token_estimates)) if token_estimates else 0
        ),
        context_token_estimate_max=max(token_estimates) if token_estimates else 0,
        context_window=int(latest_context.get("context_window") or 0),
        prompt_budget=int(latest_context.get("prompt_budget") or 0),
        reserved_output=int(latest_context.get("reserved_output") or 0),
        budget_usage_ratio_avg=(
            round(sum(budget_usage_ratios) / len(budget_usage_ratios), 6)
            if budget_usage_ratios
            else 0.0
        ),
        budget_usage_ratio_max=(
            max(budget_usage_ratios) if budget_usage_ratios else 0.0
        ),
        history_groups_compacted=sum(
            int(event.get("history_groups_compacted") or 0)
            for event in context_events
        ),
        token_estimator_version=str(
            latest_context.get("token_estimator_version") or "unknown"
        ),
        model_retry_count=sum(
            1 for event in events if event.get("type") == "model_retry_scheduled"
        ),
        reactive_compaction_count=provider_overflow_count,
        provider_overflow_count=provider_overflow_count,
        calibration_sample_count=len(calibration_ratios),
        calibration_ratio_p50=_percentile(calibration_ratios, 0.50),
        calibration_ratio_p95=_percentile(calibration_ratios, 0.95),
        calibration_relative_error_p50=_percentile(
            calibration_relative_errors,
            0.50,
        ),
        calibration_relative_error_p95=_percentile(
            calibration_relative_errors,
            0.95,
        ),
        output_recovery_count=sum(
            1
            for event in events
            if event.get("type") == "model_recovery"
            and event.get("action")
            in {
                "increase_max_output_tokens",
                "continue_truncated_output",
                "continue_visible_truncated_output",
            }
        ),
        subagent_call_count=sum(
            1 for event in events if event.get("type") == "subagent_started"
        ),
        mcp_tool_call_count=sum(
            1
            for event in tool_calls
            if str(event.get("tool") or "").startswith("mcp__")
        ),
        history_compaction_tokens_removed=sum(
            int((event.get("details") or {}).get("removed_tokens"))
            if (event.get("details") or {}).get("removed_tokens") is not None
            else max(
                0,
                int(event.get("before_tokens") or 0)
                - int(event.get("after_tokens") or 0),
            )
            for event in history_compressions
        ),
        history_compaction_tokens_retained=sum(
            int((event.get("details") or {}).get("execution_record_tokens"))
            if (event.get("details") or {}).get("execution_record_tokens") is not None
            else (
                int(event.get("after_tokens") or 0)
                if int(event.get("before_tokens") or 0)
                > int(event.get("after_tokens") or 0)
                else 0
            )
            for event in history_compressions
        ),
        modified_files_count=len(changed_files),
        final_status=status["status"],
        stop_reason=status["reason"],
    )


def _percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic nearest-rank percentile for trace metrics."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int((len(ordered) * fraction) + 0.999999))
    return round(ordered[min(len(ordered), rank) - 1], 6)


def _read_tool_fingerprint(event: dict[str, Any]) -> str | None:
    """Return an exact read-side invocation fingerprint for repeat analysis."""

    read_tools = {"read", "search"}
    tool = str(event.get("tool") or "")
    if tool not in read_tools:
        return None
    args = event.get("args")
    serialized = json.dumps(
        args if isinstance(args, dict) else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{tool}:{serialized}"


def _event_read_source(event: dict[str, Any]) -> str | None:
    tool = str(event.get("tool") or "")
    args = event.get("args")
    if tool == "read" and isinstance(args, dict):
        source = str(args.get("source") or "")
        return source or None
    return None


def _event_search_kind(event: dict[str, Any]) -> str | None:
    tool = str(event.get("tool") or "")
    args = event.get("args")
    if tool == "search" and isinstance(args, dict):
        kind = str(args.get("kind") or "")
        return kind or None
    return None


def _format_token_count(value: int) -> str:
    tokens = max(0, int(value))
    if tokens < 1_000:
        return str(tokens)
    return f"{tokens / 1_000:.1f}K"


def _existing_run_path(store: RunStore, run_id: str) -> Path:
    run_path = store.path_for(run_id)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run not found: {run_path}")
    return run_path


def _read_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _format_trace_event(index: int, event: dict[str, Any]) -> str:
    step = event.get("step")
    prefix = f"{index:03d}"
    if step is not None:
        prefix += f" | step {step}"
    event_type = event.get("type", "unknown")
    if event_type == "tool_called":
        return f"{prefix} | tool_called | {event.get('tool')} | {event.get('tool_call_id', '')}"
    if event_type == "tool_result":
        return (
            f"{prefix} | tool_result | {event.get('tool')} | "
            f"{event.get('status', 'unknown')}"
        )
    if event_type == "approval_resolved":
        return (
            f"{prefix} | approval_resolved | {event.get('tool')} | "
            f"{event.get('decision')}"
        )
    if event_type == "checkpoint_saved":
        return f"{prefix} | checkpoint_saved | {event.get('status')} | {event.get('reason')}"
    if event_type == "context_built":
        return f"{prefix} | context_built | tokens={event.get('token_estimate', 0)}"
    return f"{prefix} | {event_type}"


def _render_report(
    *,
    session: RunSession,
    events: list[dict[str, Any]],
    final_diff: str,
    run_path: Path,
) -> str:
    status = _final_status(events)
    changed_files = _changed_files(events, run_path)
    commands = _commands_run(events)
    tool_calls = _tool_calls(events)
    approvals = _approvals(events)
    checkpoints = [event for event in events if event.get("type") == "checkpoint_saved"]
    prompt_cache_lines = _prompt_cache_lines(events)
    context_metrics = collect_context_metrics(events, run_path=run_path)
    artifacts = sorted(
        path.relative_to(run_path).as_posix()
        for path in (run_path / "artifacts").rglob("*")
        if path.is_file()
    )

    lines = [
        "# Run Report",
        "",
        "## Task",
        "",
        session.task,
        "",
        "## Final Status",
        "",
        f"- Status: {status['status']}",
        f"- Stop Reason: {status['reason']}",
        f"- Run ID: `{session.run_id}`",
        f"- Workspace: `{session.workspace}`",
        "",
        "## Summary",
        "",
        f"- Trace Events: {len(events)}",
        f"- Tool Calls: {len(tool_calls)}",
        f"- Commands Run: {len(commands)}",
        f"- Approvals: {len(approvals)}",
        f"- Checkpoints: {len(checkpoints)}",
        f"- Final Diff Lines: {len(final_diff.splitlines())}",
        *prompt_cache_lines,
        "",
        "## Context Metrics",
        "",
        *_context_metrics_lines(context_metrics),
        "",
        "## Files Changed",
        "",
        *_bullet_list(changed_files),
        "",
        "## Commands Run",
        "",
        *_bullet_list(commands),
        "",
        "## Tool Calls",
        "",
        *_tool_call_table(tool_calls),
        "",
        "## Approvals",
        "",
        *_approval_table(approvals),
        "",
        "## Checkpoints",
        "",
        *_checkpoint_table(checkpoints),
        "",
        "## Artifacts",
        "",
        *_bullet_list(artifacts),
    ]
    return "\n".join(lines) + "\n"


def _final_status(events: list[dict[str, Any]]) -> dict[str, str]:
    for event in reversed(events):
        if event.get("type") == "run_finished":
            return {
                "status": str(event.get("status") or "unknown"),
                "reason": str(event.get("stop_reason") or "unknown"),
            }
        if event.get("type") == "checkpoint_saved":
            return {
                "status": str(event.get("status") or "unknown"),
                "reason": str(event.get("reason") or "unknown"),
            }
    return {"status": "unknown", "reason": "no terminal event"}


def _changed_files(events: list[dict[str, Any]], run_path: Path) -> list[str]:
    checkpoint = run_path / "checkpoints" / "latest.json"
    if checkpoint.is_file():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        files = payload.get("modified_files") or []
        if files:
            return [str(path) for path in files]

    for event in reversed(events):
        if event.get("type") == "checkpoint_saved":
            files = event.get("modified_files") or []
            if files:
                return [str(path) for path in files]
    return []


def _changed_files_from_events(events: list[dict[str, Any]]) -> list[str]:
    for event in reversed(events):
        if event.get("type") != "checkpoint_saved":
            continue
        files = event.get("modified_files") or []
        if files:
            return [str(path) for path in files]
    return []


def _commands_run(events: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for event in events:
        if event.get("type") == "tool_called" and event.get("tool") == "run_command":
            args = event.get("args") or {}
            argv = args.get("argv")
            command = (
                render_argv(argv)
                if isinstance(argv, list) and all(isinstance(item, str) for item in argv)
                else str(args.get("command") or "")
            )
            if command:
                commands.append(command)
    return commands


def _prompt_cache_lines(events: list[dict[str, Any]]) -> list[str]:
    model_events = [event for event in events if event.get("type") == "model_response"]
    if not model_events:
        return []

    provider_cached_tokens = sum(
        int((event.get("usage") or {}).get("cached_input_tokens") or 0)
        for event in model_events
    )
    provider_miss_tokens = sum(
        int((event.get("usage") or {}).get("cache_miss_input_tokens") or 0)
        for event in model_events
    )

    lines: list[str] = []
    if provider_cached_tokens or provider_miss_tokens:
        lines.extend(
            [
                f"- Provider Cached Input Tokens: {provider_cached_tokens}",
                f"- Provider Cache Miss Input Tokens: {provider_miss_tokens}",
            ]
        )
    return lines


def _context_metrics_lines(metrics: ContextMetrics) -> list[str]:
    return [
        f"- total tool calls: {metrics.total_tool_calls}",
        f"- workspace reads: {metrics.workspace_read_calls}",
        f"- artifact reads: {metrics.artifact_read_calls}",
        f"- text searches: {metrics.text_search_calls}",
        f"- unique read resources: {metrics.unique_read_resources}",
        f"- repeated read calls: {metrics.repeated_read_calls}",
        f"- avg prompt tokens: {metrics.context_token_estimate_avg}",
        f"- max prompt tokens: {metrics.context_token_estimate_max}",
        f"- context window: {metrics.context_window}",
        f"- prompt budget: {metrics.prompt_budget}",
        f"- reserved output: {metrics.reserved_output}",
        f"- avg prompt budget usage: {metrics.budget_usage_ratio_avg:.3f}",
        f"- max prompt budget usage: {metrics.budget_usage_ratio_max:.3f}",
        f"- history groups compacted: {metrics.history_groups_compacted}",
        f"- token estimator: {metrics.token_estimator_version}",
        f"- model retries: {metrics.model_retry_count}",
        f"- reactive compactions: {metrics.reactive_compaction_count}",
        f"- provider overflows: {metrics.provider_overflow_count}",
        f"- token calibration samples: {metrics.calibration_sample_count}",
        f"- token calibration ratio P50: {metrics.calibration_ratio_p50:.3f}",
        f"- token calibration ratio P95: {metrics.calibration_ratio_p95:.3f}",
        f"- token calibration relative error P50: "
        f"{metrics.calibration_relative_error_p50:.3f}",
        f"- token calibration relative error P95: "
        f"{metrics.calibration_relative_error_p95:.3f}",
        f"- output recoveries: {metrics.output_recovery_count}",
        f"- subagent calls: {metrics.subagent_call_count}",
        f"- MCP tool calls: {metrics.mcp_tool_call_count}",
        f"- history compaction tokens removed: {metrics.history_compaction_tokens_removed}",
        f"- history compaction tokens retained: {metrics.history_compaction_tokens_retained}",
        f"- modified files: {metrics.modified_files_count}",
        f"- final status: {metrics.final_status}",
        f"- stop reason: {metrics.stop_reason}",
    ]


def _tool_calls(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    result_status_by_id = {
        str(event.get("tool_call_id")): str(event.get("status") or "")
        for event in events
        if event.get("type") == "tool_result"
    }
    calls: list[dict[str, str]] = []
    for event in events:
        if event.get("type") != "tool_called":
            continue
        tool_call_id = str(event.get("tool_call_id") or "")
        calls.append(
            {
                "step": str(event.get("step") or ""),
                "tool": str(event.get("tool") or ""),
                "status": result_status_by_id.get(tool_call_id, "unknown"),
            }
        )
    return calls


def _approvals(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    approvals: list[dict[str, str]] = []
    for event in events:
        if event.get("type") != "approval_resolved":
            continue
        approvals.append(
            {
                "step": str(event.get("step") or ""),
                "tool": str(event.get("tool") or ""),
                "decision": str(event.get("decision") or ""),
                "reason": str(event.get("reason") or ""),
            }
        )
    return approvals


def _workspace_diff(session: RunSession) -> str:
    try:
        result = inspect_git_diff(session.workspace)
    except Exception as exc:
        return f"# Unable to inspect git diff: {type(exc).__name__}: {exc}\n"
    if result.returncode != 0:
        return f"# Unable to inspect git diff:\n{result.stderr}\n"
    return result.diff


def _bullet_list(values: list[str]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- `{value}`" for value in values]


def _tool_call_table(tool_calls: list[dict[str, str]]) -> list[str]:
    if not tool_calls:
        return ["No tool calls recorded."]
    lines = ["| Step | Tool | Result |", "|---:|---|---|"]
    for call in tool_calls:
        lines.append(f"| {call['step']} | `{call['tool']}` | {call['status']} |")
    return lines


def _approval_table(approvals: list[dict[str, str]]) -> list[str]:
    if not approvals:
        return ["No approvals recorded."]
    lines = ["| Step | Tool | Decision | Reason |", "|---:|---|---|---|"]
    for approval in approvals:
        lines.append(
            f"| {approval['step']} | `{approval['tool']}` | "
            f"{approval['decision']} | {approval['reason']} |"
        )
    return lines


def _checkpoint_table(checkpoints: list[dict[str, Any]]) -> list[str]:
    if not checkpoints:
        return ["No checkpoints recorded."]
    lines = ["| Step | Status | Reason | Path |", "|---:|---|---|---|"]
    for checkpoint in checkpoints:
        lines.append(
            f"| {checkpoint.get('step', '')} | {checkpoint.get('status', '')} | "
            f"{checkpoint.get('reason', '')} | `{checkpoint.get('path', '')}` |"
        )
    return lines
