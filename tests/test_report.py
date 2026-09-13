from __future__ import annotations

import json
from pathlib import Path

from minicode_harness.report import (
    collect_context_metrics,
    format_context_usage,
    format_run_trace,
    generate_run_report,
)
from minicode_harness.state import RunStore
from minicode_harness.trace import TraceWriter


def test_format_trace_and_generate_report(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="Fix README",
        workspace=workspace,
        run_id="run_20260701_001",
    )
    run_path = run_store.path_for(session.run_id)
    writer = TraceWriter(run_path / "trace.jsonl")
    writer.write_event("run_started", run_id=session.run_id, task=session.task)
    writer.write_event(
        "context_built",
        step=1,
        token_estimate=120,
        context_window=32_000,
        prompt_budget=28_000,
        reserved_output=4_000,
        request_tokens_before_compaction=180,
        request_tokens_after_argument_compaction=170,
        budget_usage_ratio=0.004286,
        history_groups_compacted=6,
        source_tokens={
            "system": 20,
            "tools": 30,
            "historical_messages": 15,
            "current_turn": 40,
            "semantic_history": 5,
            "execution_record": 4,
            "current_tool_frontier": 3,
            "task_projection": 2,
            "runtime_notifications": 1,
        },
        reactive_compaction_count=0,
        token_estimator_version="mixed-language-v1",
        prompt_cache_enabled=True,
        prompt_cache_hit=True,
        prompt_cache_key="abc",
        prompt_prefix_hash="def",
        prompt_prefix_tokens=80,
    )
    writer.write_event(
        "context_compressed",
        step=1,
        reason="execution_history",
        before_tokens=1000,
        after_tokens=320,
        details={
            "strategy": "deterministic_execution",
            "removed_tokens": 800,
            "execution_record_tokens": 120,
        },
    )
    writer.write_event(
        "model_response",
        step=1,
        usage={
            "input_tokens": 120,
            "output_tokens": 10,
            "total_tokens": 130,
            "cached_input_tokens": 64,
            "cache_miss_input_tokens": 56,
        },
    )
    writer.write_event(
        "token_estimate_calibration",
        step=1,
        estimated_prompt_tokens=100,
        provider_prompt_tokens=110,
        ratio=1.1,
        relative_error=0.090909,
    )
    writer.write_event(
        "token_estimate_calibration",
        step=2,
        estimated_prompt_tokens=100,
        provider_prompt_tokens=120,
        ratio=1.2,
        relative_error=0.166667,
    )
    writer.write_event(
        "tool_called",
        step=1,
        tool_call_id="call_1",
        tool="write",
        args={"path": "README.md", "content": "hello\n"},
    )
    writer.write_event(
        "tool_result",
        step=1,
        tool_call_id="call_1",
        tool="write",
        status="ok",
    )
    writer.write_event(
        "tool_called",
        step=2,
        tool_call_id="call_2",
        tool="read",
        args={"source": "workspace", "target": "README.md"},
    )
    writer.write_event(
        "tool_result",
        step=2,
        tool_call_id="call_2",
        tool="read",
        status="project_cache_hit",
    )
    writer.write_event(
        "approval_resolved",
        step=1,
        tool_call_id="call_1",
        tool="write",
        decision="approve",
        reason="test",
    )
    writer.write_event(
        "checkpoint_saved",
        step=1,
        path=str(run_path / "checkpoints" / "step_0001.json"),
        status="running",
        reason="tool:write",
        modified_files=["README.md"],
    )
    writer.write_event(
        "run_finished",
        run_id=session.run_id,
        status="completed",
        stop_reason="final_text",
    )

    trace = format_run_trace(session.run_id, run_store)
    context_usage = format_context_usage(session.run_id, run_store)
    report = generate_run_report(session.run_id, run_store)
    report_text = report.report_path.read_text(encoding="utf-8")
    trace_events = [
        json.loads(line)
        for line in (run_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metrics = collect_context_metrics(trace_events, run_path=run_path)

    assert "Run Trace: run_20260701_001" in trace
    assert "tool_called | write" in trace
    assert "Usage: 120 / 28.0K" in context_usage
    assert "System Prompt" in context_usage
    assert "Current Tool Frontier" in context_usage
    assert "Last compaction: execution_history" in context_usage
    assert "Provider overflow: 0" in context_usage
    assert "Latest calibration: 1.200x" in context_usage
    assert report.report_path == run_path / "report.md"
    assert report.final_diff_path == run_path / "final.diff"
    assert metrics.total_tool_calls == 2
    assert metrics.workspace_read_calls == 1
    assert metrics.artifact_read_calls == 0
    assert metrics.unique_read_resources == 1
    assert metrics.repeated_read_calls == 0
    assert metrics.project_cache_hit_count == 1
    assert metrics.context_window == 32_000
    assert metrics.prompt_budget == 28_000
    assert metrics.reserved_output == 4_000
    assert metrics.budget_usage_ratio_avg == 0.004286
    assert metrics.budget_usage_ratio_max == 0.004286
    assert metrics.history_groups_compacted == 6
    assert metrics.token_estimator_version == "mixed-language-v1"
    assert metrics.model_retry_count == 0
    assert metrics.reactive_compaction_count == 0
    assert metrics.provider_overflow_count == 0
    assert metrics.calibration_sample_count == 2
    assert metrics.calibration_ratio_p50 == 1.1
    assert metrics.calibration_ratio_p95 == 1.2
    assert metrics.calibration_relative_error_p50 == 0.090909
    assert metrics.calibration_relative_error_p95 == 0.166667
    assert metrics.output_recovery_count == 0
    assert metrics.subagent_call_count == 0
    assert metrics.mcp_tool_call_count == 0
    assert metrics.history_compaction_tokens_removed == 800
    assert metrics.history_compaction_tokens_retained == 120
    assert "## Files Changed" in report_text
    assert "## Context Metrics" in report_text
    assert "- total tool calls: 2" in report_text
    assert "- workspace reads: 1" in report_text
    assert "- artifact reads: 0" in report_text
    assert "- unique read resources: 1" in report_text
    assert "- repeated read calls: 0" in report_text
    assert "- project cache hits: 1" in report_text
    assert "- context window: 32000" in report_text
    assert "- prompt budget: 28000" in report_text
    assert "- reserved output: 4000" in report_text
    assert "- full tool results retained:" not in report_text
    assert "- tool results compacted:" not in report_text
    assert "- history groups compacted: 6" in report_text
    assert "- token estimator: mixed-language-v1" in report_text
    assert "- model retries: 0" in report_text
    assert "- provider overflows: 0" in report_text
    assert "- token calibration samples: 2" in report_text
    assert "- token calibration ratio P50: 1.100" in report_text
    assert "- token calibration ratio P95: 1.200" in report_text
    assert "- subagent calls: 0" in report_text
    assert "- MCP tool calls: 0" in report_text
    assert "LLM history summary" not in report_text
    assert "- history compaction tokens removed: 800" in report_text
    assert "- history compaction tokens retained: 120" in report_text
    assert "- Prompt Cache Enabled Contexts: 1" in report_text
    assert "- Harness Prompt Cache Hits: 1/1" in report_text
    assert "- Provider Cached Input Tokens: 64" in report_text
    assert "- Provider Cache Miss Input Tokens: 56" in report_text
    assert "- `README.md`" in report_text
    assert "| 1 | `write` | ok |" in report_text
    assert "| 1 | `write` | approve | test |" in report_text


def test_context_metrics_count_exact_repeated_read_invocations() -> None:
    events = [
        {
            "type": "tool_called",
            "tool": "read",
            "args": {"source": "workspace", "target": "README.md", "start_line": 1, "end_line": 20},
        },
        {
            "type": "tool_called",
            "tool": "read",
            "args": {"source": "workspace", "target": "README.md", "start_line": 1, "end_line": 20},
        },
        {
            "type": "tool_called",
            "tool": "read",
            "args": {"source": "workspace", "target": "README.md", "start_line": 21, "end_line": 40},
        },
    ]

    metrics = collect_context_metrics(events)

    assert metrics.workspace_read_calls == 3
    assert metrics.unique_read_resources == 2
    assert metrics.repeated_read_calls == 1
