import json
from io import StringIO
import os
import sys
from threading import Barrier, Lock
from typing import Any

from minicode_harness.context import (
    ContextPreparer,
    RunState,
    TokenBudget,
    VerificationState,
)
from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.state import (
    ApprovalDecision,
    CheckpointStore,
    ReplSessionMemory,
    StaticApprovalClient,
)
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.mcp import InProcessMCPServer, MCPManager, MCPToolSpec
from minicode_harness.models import (
    ModelCapabilities,
    ModelClient,
    ModelResponse,
    ModelUsage,
    NormalizedToolCall,
)
from minicode_harness.output import TextOutputSink
from minicode_harness.trace import TraceWriter


class SmallWindowScriptedModelClient(ModelClient):
    capabilities = ModelCapabilities(
        context_window=6_000,
        max_output_tokens=1_024,
    )

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def call_request(self, request) -> ModelResponse:
        self.calls.append((request.as_chat_messages(), request.tools))
        return self.responses.pop(0)


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def call_request(self, request) -> ModelResponse:
        self.calls.append((request.as_chat_messages(), request.tools))
        return self.responses.pop(0)


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {tool["function"]["name"] for tool in tools}


def _trace_events(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _prepend_current_python_to_path(monkeypatch) -> None:
    current = os.environ.get("PATH", "")
    python_dir = os.path.dirname(sys.executable)
    monkeypatch.setenv("PATH", python_dir + os.pathsep + current)


def test_runtime_resource_closers_keep_shared_registry_as_final_task_boundary(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loop = AgentLoop(
        task="Inspect runtime lifecycle",
        workspace=workspace,
        model_client=ScriptedModelClient([ModelResponse(final_text="done")]),
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    )

    assert [name for name, _ in loop.lifecycle.resource_closers[:3]] == [
        "worktree_workers",
        "background_commands",
        "runtime_tasks",
    ]
    loop.lifecycle.close_runtime_resources()


def test_same_response_mixed_tools_run_in_parallel_and_keep_result_order(
    monkeypatch,
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("A = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("TARGET = 2\n", encoding="utf-8")
    trace_path = tmp_path / "parallel" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read_a",
                        name="read",
                        arguments={"source": "workspace", "target": "a.py"},
                    ),
                    NormalizedToolCall(
                        id="search_b",
                        name="search",
                        arguments={
                            "source": "workspace",
                            "kind": "text",
                            "path": "b.py",
                            "query": "TARGET",
                        },
                    ),
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Inspect two independent targets",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    )
    barrier = Barrier(2)
    lock = Lock()
    started: list[str] = []
    completed: list[str] = []
    original_execute = loop.tools.execute_admitted

    def execute_in_barrier(
        admission,
        *,
        approval_granted=False,
        expected_file_sha256=None,
    ):
        with lock:
            started.append(admission.name)
        barrier.wait(timeout=2)
        result = original_execute(
            admission,
            approval_granted=approval_granted,
            expected_file_sha256=expected_file_sha256,
        )
        with lock:
            completed.append(admission.name)
        return result

    monkeypatch.setattr(loop.tools, "execute_admitted", execute_in_barrier)

    result = loop.run()

    assert result.status == "completed"
    assert set(started) == {"read", "search"}
    assert set(completed) == {"read", "search"}
    tool_results = [
        message for message in client.calls[1][0] if message["role"] == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_results] == [
        "read_a",
        "search_b",
    ]
    events = _trace_events(trace_path)
    started_event = next(event for event in events if event["type"] == "tool_batch_started")
    assert started_event["tools"] == ["read", "search"]
    assert any(event["type"] == "tool_batch_finished" for event in events)


def test_agent_loop_carries_native_tool_history_and_minimal_run_state(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    trace_path = tmp_path / "run" / "trace.jsonl"
    trace_path.parent.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_read",
                        name="read",
                        arguments={"source": "workspace", "target": "README.md"},
                    )
                ],
                usage=ModelUsage(input_tokens=1_234),
            ),
            ModelResponse(final_text="README contains hello."),
        ]
    )

    result = AgentLoop(
        task="Explain README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert result.tool_calls == 1
    first_prompt = "\n".join(
        str(message.get("content") or "") for message in client.calls[0][0]
    )
    second_prompt = "\n".join(
        str(message.get("content") or "") for message in client.calls[1][0]
    )
    assert "工作目录：" not in first_prompt
    assert "工具调用预算" not in first_prompt
    assert "工具调用预算" not in second_prompt
    assert "不得反向解释为要求" in first_prompt
    assert "现有测试不足时才新增" in first_prompt
    second_messages = client.calls[1][0]
    assert [message["role"] for message in second_messages[-2:]] == ["assistant", "tool"]
    assert second_messages[-2]["tool_calls"][0]["id"] == "call_read"
    tool_payload = json.loads(second_messages[-1]["content"])
    assert tool_payload["path"] == "README.md"
    assert "hello" in tool_payload["content"]
    assert "resource:" not in second_messages[-1]["content"]
    assert "result:" not in second_messages[-1]["content"]

    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert checkpoint.run_state.inspected_files[0].path == "README.md"
    payload = json.loads((trace_path.parent / "checkpoints" / "latest.json").read_text())
    assert "run_state" in payload
    assert "working_context" not in payload
    assert set(payload["run_state"]) == {"inspected_files", "verification"}
    assert "task_memory" not in payload

    trace_events = _trace_events(trace_path)
    event_types = [event["type"] for event in trace_events]
    context_event = next(
        event for event in trace_events if event["type"] == "context_built"
    )
    assert context_event["context_window"] == 32_000
    assert context_event["prompt_budget"] == 27_904
    assert context_event["reserved_output"] == 4_096
    assert context_event["request_tokens_before_compaction"] > 0
    assert context_event["request_tokens_after_argument_compaction"] > 0
    assert "request_tokens_after_result_compaction" not in context_event
    assert context_event["budget_usage_ratio"] > 0
    assert "full_tool_results_retained" not in context_event
    assert "tool_results_compacted" not in context_event
    assert context_event["history_groups_compacted"] == 0
    assert context_event["reactive_compaction_count"] == 0
    assert context_event["token_estimator_version"] == "mixed-language-v1"
    assert context_event["tool_schema_chars"] > 0
    assert context_event["tool_schema_tokens"] > 0
    assert context_event["largest_tool_schema"]["name"] in context_event["available_tools"]
    assert context_event["largest_tool_schema"]["tokens"] > 0
    calibration = next(
        event
        for event in trace_events
        if event["type"] == "token_estimate_calibration"
    )
    assert calibration["provider_prompt_tokens"] == 1_234
    assert calibration["estimated_prompt_tokens"] > 0
    assert calibration["ratio"] > 0
    assert calibration["relative_error"] >= 0
    assert "run_state_updated" in event_types
    assert "run_outcome" in event_types
    finalize_pairs = (
        ("session_history_persist_started", "session_history_persist_finished"),
        ("final_text_emit_started", "final_text_emit_finished"),
        ("final_checkpoint_started", "final_checkpoint_finished"),
    )
    for started, finished in finalize_pairs:
        assert event_types.index(started) < event_types.index(finished)
        finished_event = next(event for event in trace_events if event["type"] == finished)
        assert finished_event["status"] == "ok"
        assert finished_event["duration_ms"] >= 0
    assert event_types.index("session_history_persist_finished") < event_types.index(
        "final_text_emit_started"
    )
    assert event_types.index("final_text_emit_finished") < event_types.index(
        "final_checkpoint_started"
    )
    assert "evidence_indexed" not in event_types
    assert "task_memory_updated" not in event_types


def test_agent_loop_reuses_canonical_session_messages_across_runs(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    session = ReplSessionMemory(workspace=str(workspace.resolve()))
    first_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read_first",
                        name="read",
                        arguments={"source": "workspace", "target": "README.md"},
                    )
                ]
            ),
            ModelResponse(final_text="README says hello."),
        ]
    )

    AgentLoop(
        task="Explain README",
        workspace=workspace,
        model_client=first_client,
        trace_writer=TraceWriter(tmp_path / "first" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        session_memory=session,
        no_skills=True,
    ).run()

    second_client = ScriptedModelClient([ModelResponse(final_text="The previous read is retained.")])
    AgentLoop(
        task="What did it say?",
        workspace=workspace,
        model_client=second_client,
        trace_writer=TraceWriter(tmp_path / "second" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        session_memory=session,
        no_skills=True,
    ).run()

    messages = second_client.calls[0][0]
    roles = [message["role"] for message in messages]
    assert roles[-5:] == ["user", "assistant", "tool", "assistant", "user"]
    assert messages[-5]["content"] == "Explain README"
    assert messages[-4]["tool_calls"][0]["id"] == "read_first"
    assert "hello" in messages[-3]["content"]
    assert messages[-2]["content"] == "README says hello."
    assert messages[-1]["content"] == "What did it say?"
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    assert "Earlier Session Tool Transcript" not in rendered
    assert "Call-time Context" not in rendered


def test_agent_loop_context_contains_no_removed_controller_layers(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="done")])

    AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    ).run()

    prompt = "\n".join(str(message.get("content") or "") for message in client.calls[0][0])
    assert "Task Memory" not in prompt
    assert "Evidence Sources" not in prompt
    assert "Evidence Sufficiency" not in prompt
    assert "Edit Target" not in prompt
    assert "active_edit_target" not in prompt


def test_agent_loop_does_not_reuse_workspace_search_for_artifact_source(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "same.txt").write_text("needle in workspace\n", encoding="utf-8")
    trace_path = tmp_path / "run" / "trace.jsonl"
    artifacts = trace_path.parent / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "same.txt").write_text("needle in artifact\n", encoding="utf-8")
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="search_workspace",
                        name="search",
                        arguments={
                            "source": "workspace",
                            "kind": "text",
                            "query": "needle",
                            "path": "same.txt",
                        },
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="search_artifact",
                        name="search",
                        arguments={
                            "source": "artifact",
                            "kind": "text",
                            "query": "needle",
                            "path": "same.txt",
                        },
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )

    AgentLoop(
        task="Compare search sources",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    ).run()

    tool_results = {
        event["tool_call_id"]: event
        for event in _trace_events(trace_path)
        if event.get("type") == "tool_result"
    }
    assert tool_results["search_workspace"]["status"] == "ok"
    assert tool_results["search_artifact"]["status"] == "ok"


def test_agent_loop_exposes_tools_only_from_write_mode(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    read_client = ScriptedModelClient([ModelResponse(final_text="read")])
    write_client = ScriptedModelClient([ModelResponse(final_text="blocked")])

    AgentLoop(
        task="Explain",
        workspace=workspace,
        model_client=read_client,
        trace_writer=TraceWriter(tmp_path / "read-trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory-read"),
        enable_write=False,
    ).run()
    AgentLoop(
        task="Explain",
        workspace=workspace,
        model_client=write_client,
        trace_writer=TraceWriter(tmp_path / "write-trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory-write"),
        enable_write=True,
    ).run()

    read_names = _tool_names(read_client.calls[0][1])
    write_names = _tool_names(write_client.calls[0][1])
    assert {"read", "search", "task"}.issubset(read_names)
    assert {"delegate_task", "delegate_worktree", "runtime_task_status", "runtime_task_stop"}.issubset(read_names)
    assert not {"edit", "write", "apply_patch", "run_command"}.intersection(read_names)
    read_schema = next(
        schema
        for schema in read_client.calls[0][1]
        if schema["function"]["name"] == "read"
    )
    search_schema = next(
        schema
        for schema in read_client.calls[0][1]
        if schema["function"]["name"] == "search"
    )
    delegate_schema = next(
        schema
        for schema in read_client.calls[0][1]
        if schema["function"]["name"] == "delegate_task"
    )
    read_description = str(read_schema["function"]["description"])
    search_description = str(search_schema["function"]["description"])
    delegate_description = str(delegate_schema["function"]["description"])
    search_path_description = str(
        search_schema["function"]["parameters"]["properties"]["path"]["description"]
    )
    assert "explicitly typed resource" in read_description
    assert "Workspace and Artifact targets must identify a file" in read_description
    assert "optional line ranges select an inclusive local range" in read_description
    assert "source=diff returns the current Git diff without a target" in read_description
    assert "kind=files locates candidate paths by glob" in search_description
    assert "narrow path/glob before widening depth/limit" in search_description
    assert "whole-repository inventory" in search_description
    assert "kind=text searches matching lines" in search_description
    assert "Results may truncate by bounds" in search_description
    assert "Source-relative file or directory to search" in search_path_description
    assert "'.' means the selected source root" in search_path_description
    assert "parallel implementations" not in search_description
    assert 'path="."' not in search_description
    assert "Do not bundle modules/workflows" in delegate_description
    assert "whole-repository coverage" in delegate_description
    delegate_task_description = str(
        delegate_schema["function"]["parameters"]["properties"]["task"]["description"]
    )
    assert "one scope/result" in delegate_task_description
    assert "no independent objectives" in delegate_task_description
    assert {"edit", "write", "apply_patch", "run_command"}.issubset(write_names)


def test_agent_loop_writes_non_code_file_and_completes(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run" / "trace.jsonl"
    trace_path.parent.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write",
                        name="write",
                        arguments={"path": "notes.txt", "content": "done\n"},
                    )
                ]
            ),
            ModelResponse(final_text="Created notes.txt."),
        ]
    )

    result = AgentLoop(
        task="Create notes.txt containing done",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    ).run()

    assert result.status == "completed"
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "done\n"
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert checkpoint.modified_files == ["notes.txt"]
    assert checkpoint.run_state.verification.status == "not_run"
    write_result = next(
        event
        for event in _trace_events(trace_path)
        if event["type"] == "tool_result" and event.get("tool") == "write"
    )
    assert write_result["write_strategy"] == "full_write_create"


def test_agent_loop_resume_keeps_runtime_budgets_out_of_system_prompt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="resumed")])

    result = AgentLoop(
        task="Resume the task",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "resume-step" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        config=AgentLoopConfig(start_step=14, max_steps=20, max_tool_calls=30),
    ).run()

    prompt = "\n".join(str(message.get("content") or "") for message in client.calls[0][0])
    assert result.steps == 15
    assert "工作目录：" not in prompt
    assert "模型调用预算" not in prompt
    assert "工具调用预算" not in prompt


def test_agent_loop_unknown_tool_is_unavailable_before_argument_validation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "unknown" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="unknown",
                        name="missing_tool",
                        arguments={"malformed": 123},
                    )
                ]
            ),
            ModelResponse(final_text="The tool is unavailable."),
        ]
    )

    result = AgentLoop(
        task="Use an unavailable tool",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    events = _trace_events(trace_path)
    unavailable = [
        event
        for event in events
        if event["type"] == "tool_result" and event.get("tool") == "missing_tool"
    ]
    assert unavailable[0]["status"] == "tool_unavailable"
    assert not any(event["type"] == "tool_arguments_invalid" for event in events)


def test_agent_loop_stops_at_max_steps(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="one",
                        name="search",
                        arguments={"source": "workspace", "kind": "files", "query": "**/*", "path": "."},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="two",
                        name="search",
                        arguments={"source": "workspace", "kind": "files", "query": "**/*", "path": "."},
                    )
                ]
            ),
        ]
    )

    trace_path = tmp_path / "trace.jsonl"
    loop = AgentLoop(
        task="Keep exploring",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        config=AgentLoopConfig(max_steps=2),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    result = loop.run()

    assert result.status == "stopped"
    assert result.stop_reason == "max_steps"
    assert result.final_text is None
    assert result.stop_summary is not None
    assert "model-call budget was exhausted" in result.stop_summary
    assert "Changes: none" in result.stop_summary
    assert "Last tool result: search [ok]" in result.stop_summary
    assert result.stop_summary not in json.dumps(
        loop.user_turn.snapshot_messages(),
        ensure_ascii=False,
    )
    events = _trace_events(trace_path)
    assert any(event["type"] == "budget_stop_summary" for event in events)
    run_outcome = next(event for event in events if event["type"] == "run_outcome")
    assert run_outcome["final_text"] is None
    assert run_outcome["stop_summary"] == result.stop_summary


def test_agent_loop_pairs_partial_tool_group_at_budget_boundary(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "first.txt").write_text("first\n", encoding="utf-8")
    (workspace / "second.txt").write_text("second\n", encoding="utf-8")
    trace_path = tmp_path / "budget-boundary" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="first",
                        name="read",
                        arguments={"source": "workspace", "target": "first.txt"},
                    ),
                    NormalizedToolCall(
                        id="second",
                        name="read",
                        arguments={"source": "workspace", "target": "second.txt"},
                    ),
                ]
            ),
            ModelResponse(
                final_text="Read first.txt; second.txt was not read because the budget ended."
            ),
        ]
    )

    result = AgentLoop(
        task="Read both files and summarize what is available.",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        config=AgentLoopConfig(max_steps=3, max_tool_calls=1),
        memory_store=ProjectMemoryStore(tmp_path / "budget-boundary-memory"),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert result.tool_calls == 1
    assert len(client.calls) == 2
    second_request = client.calls[1][0]
    assert [message["role"] for message in second_request[-3:]] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert len(second_request[-3]["tool_calls"]) == 2
    skipped_payload = json.loads(second_request[-1]["content"])
    assert skipped_payload["status"] == "tool_budget_exhausted"
    events = _trace_events(trace_path)
    assert any(event["type"] == "tool_call_budget_exhausted" for event in events)
    assert any(
        event["type"] == "tool_result"
        and event.get("tool_call_id") == "second"
        and event.get("status") == "tool_budget_exhausted"
        for event in events
    )


def test_agent_loop_records_tool_error_and_continues(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="missing",
                        name="read",
                        arguments={"source": "workspace", "target": "missing.txt"},
                    )
                ]
            ),
            ModelResponse(final_text="The file is missing."),
        ]
    )

    result = AgentLoop(
        task="Read missing.txt",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    ).run()

    assert result.status == "completed"
    error_payload = json.loads(client.calls[1][0][-1]["content"])
    assert error_payload["status"] == "error"
    assert error_payload["error_type"] == "not_found"
    assert error_payload["retryable"] is True
    assert error_payload["side_effect"] == "none"
    assert "exception_type" not in error_payload


def test_agent_loop_successful_edit_carries_bounded_ui_diff_only_in_metadata(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="edit_value",
                        name="edit",
                        arguments={
                            "path": "app.py",
                            "old_text": "VALUE = 1\n",
                            "new_text": "VALUE = 2\n",
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Updated the value."),
        ]
    )
    loop = AgentLoop(
        task="Update app.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "diff-preview" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "diff-preview-memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    )

    result = loop.run()

    assert result.status == "completed"
    observation = next(item for item in loop.observations if item.tool_call_id == "edit_value")
    diff_preview = observation.metadata["diff_preview"]
    assert "-VALUE = 1" in diff_preview
    assert "+VALUE = 2" in diff_preview
    assert len(diff_preview) <= 2000
    assert len(diff_preview.splitlines()) <= 12
    assert observation.metadata["diff_truncated"] is False
    model_payload = client.calls[1][0][-1]["content"]
    assert "diff_preview" not in model_payload


def test_agent_loop_command_task_finalizes_after_command(tmp_path, monkeypatch) -> None:
    _prepend_current_python_to_path(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_dir = workspace / "src"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="status",
                        name="run_command",
                        arguments={"argv": ["python", "-m", "compileall", "-q", "src/app.py"]},
                    )
                ]
            ),
            ModelResponse(final_text="Focused verification completed."),
        ]
    )

    result = AgentLoop(
        task="Run focused verification",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    ).run()

    assert result.status == "completed"
    assert result.stop_reason == "final_text"
    assert len(client.calls) == 2
    checkpoint = CheckpointStore(tmp_path / "checkpoints").load_latest()
    assert checkpoint is not None
    assert checkpoint.run_state.verification.status == "passed"
    assert checkpoint.run_state.verification.command == (
        "python -m compileall -q src/app.py"
    )
    assert checkpoint.run_state.verification.returncode == 0


def test_agent_loop_records_failed_verification(tmp_path, monkeypatch) -> None:
    _prepend_current_python_to_path(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_dir = workspace / "src"
    source_dir.mkdir()
    (source_dir / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    trace_path = tmp_path / "failed-verification" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="verify",
                        name="run_command",
                        arguments={
                            "argv": ["python", "-m", "compileall", "-q", "src/broken.py"]
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Verification failed because the file has invalid syntax."),
        ]
    )

    result = AgentLoop(
        task="Run focused verification",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "failed-verification-memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    ).run()

    assert result.status == "completed"
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert checkpoint.run_state.verification.status == "failed"
    assert checkpoint.run_state.verification.command == (
        "python -m compileall -q src/broken.py"
    )
    assert checkpoint.run_state.verification.returncode != 0
    assert checkpoint.run_state.verification.reason == "failed"
    command_payload = json.loads(client.calls[1][0][-1]["content"])
    assert command_payload["status"] == "command_failed"
    assert command_payload["command_status"] == "failed"
    assert command_payload["error_type"] == "command_failed"
    assert command_payload["retryable"] is True
    assert command_payload["side_effect"] == "possible"


def test_agent_loop_blocks_same_failed_command_without_workspace_change(
    tmp_path,
    monkeypatch,
) -> None:
    _prepend_current_python_to_path(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_dir = workspace / "src"
    source_dir.mkdir()
    (source_dir / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    command = ["python", "-m", "compileall", "-q", "src/broken.py"]
    trace_path = tmp_path / "same-failed-command" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="verify_first",
                        name="run_command",
                        arguments={"argv": command},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="verify_duplicate",
                        name="run_command",
                        arguments={"argv": command},
                    )
                ]
            ),
            ModelResponse(final_text="The repeated failed command was blocked."),
        ]
    )

    result = AgentLoop(
        task="Run focused verification and do not repeat it without progress",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "same-failed-command-memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    ).run()

    assert result.status == "completed"
    blocked_payload = json.loads(client.calls[2][0][-1]["content"])
    assert blocked_payload["status"] == "same_failed_command_without_workspace_change"
    assert blocked_payload["previous_returncode"] != 0
    assert blocked_payload["workspace_generation"] == 0
    events = _trace_events(trace_path)
    command_results = [
        event
        for event in events
        if event.get("type") == "tool_result" and event.get("tool") == "run_command"
    ]
    assert [event["status"] for event in command_results] == [
        "command_failed",
        "same_failed_command_without_workspace_change",
    ]


def test_agent_loop_allows_same_failed_command_after_workspace_change(
    tmp_path,
    monkeypatch,
) -> None:
    _prepend_current_python_to_path(monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_dir = workspace / "src"
    source_dir.mkdir()
    (source_dir / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    command = ["python", "-m", "compileall", "-q", "src/broken.py"]
    trace_path = tmp_path / "failed-command-after-edit" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="verify_first",
                        name="run_command",
                        arguments={"argv": command},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="repair",
                        name="edit",
                        arguments={
                            "path": "src/broken.py",
                            "old_text": "def broken(:\n",
                            "new_text": "def fixed():\n    return 1\n",
                        },
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="verify_after_edit",
                        name="run_command",
                        arguments={"argv": command},
                    )
                ]
            ),
            ModelResponse(final_text="The focused verification passed after the edit."),
        ]
    )

    result = AgentLoop(
        task="Repair the syntax error and rerun focused verification",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "failed-command-after-edit-memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    ).run()

    assert result.status == "completed"
    events = _trace_events(trace_path)
    command_results = [
        event
        for event in events
        if event.get("type") == "tool_result" and event.get("tool") == "run_command"
    ]
    assert [event["status"] for event in command_results] == [
        "command_failed",
        "ok",
    ]
    assert not any(
        event.get("status") == "same_failed_command_without_workspace_change"
        for event in command_results
    )


def test_agent_loop_rolls_back_with_structured_verification_state(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "rollback" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write",
                        name="write",
                        arguments={"path": "temporary.txt", "content": "temp\n"},
                    )
                ]
            )
        ]
    )

    loop = AgentLoop(
        task="Create a temporary file",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        config=AgentLoopConfig(max_steps=1),
        memory_store=ProjectMemoryStore(tmp_path / "rollback-memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    )

    result = loop.run()

    assert result.status == "stopped"
    assert result.stop_reason == "max_steps"
    assert result.stop_summary is not None
    assert "Changes rolled back: temporary.txt" in result.stop_summary
    assert "Verification: prior changes were rolled back" in result.stop_summary
    assert result.stop_summary not in json.dumps(
        loop.user_turn.snapshot_messages(),
        ensure_ascii=False,
    )
    assert not (workspace / "temporary.txt").exists()
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert checkpoint.modified_files == []
    assert checkpoint.run_state.verification.status == "rolled_back"
    assert checkpoint.run_state.verification.reason == "max_steps"


def test_agent_loop_restores_explicit_minimal_run_state(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "trace.jsonl"
    state = RunState(
        verification=VerificationState(
            status="failed",
            command="python -m pytest tests/test_x.py -q",
            returncode=1,
        )
    )
    client = ScriptedModelClient([ModelResponse(final_text="blocked")])

    result = AgentLoop(
        task="Continue",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        initial_run_state=state,
    ).run()

    assert result.status == "completed"
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert checkpoint.run_state.verification.status == "failed"
    assert checkpoint.run_state.verification.returncode == 1


def test_agent_loop_accepts_final_text_without_inferring_required_change(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="No repository change is required.")])

    result = AgentLoop(
        task="分析并修复这个问题",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "fact-gate" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "fact-gate-memory"),
        enable_write=True,
    ).run()

    assert result.status == "completed"
    assert result.stop_reason == "final_text"
    assert result.tool_calls == 0


def test_agent_loop_rewrites_incomplete_markdown_final_without_tools(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "incomplete-final" / "trace.jsonl"
    trace_path.parent.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(final_text="说明如下：\n```text\n未完成"),
            ModelResponse(final_text="完整说明。"),
        ]
    )
    stream = StringIO()

    result = AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "incomplete-final-memory"),
        output_sink=TextOutputSink(stream=stream),
        stream_model=True,
    ).run()

    assert result.status == "completed"
    assert result.final_text == "完整说明。"
    assert len(client.calls) == 2
    assert client.calls[1][1] == []
    second_prompt = "\n".join(
        str(message.get("content") or "") for message in client.calls[1][0]
    )
    assert "上一份最终答案结构不完整" in second_prompt
    assert "说明如下" in second_prompt
    assert "说明如下" not in stream.getvalue()
    assert stream.getvalue().endswith("完整说明。")
    events = _trace_events(trace_path)
    assert any(
        event.get("type") == "model_recovery"
        and event.get("action") == "rewrite_incomplete_final_text"
        and event.get("reason") == "unclosed_markdown_fence"
        for event in events
    )
    assert any(event.get("type") == "final_answer_rewrite_step" for event in events)


def test_agent_loop_accepts_final_text_after_unverified_code_change(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write_python",
                        name="write",
                        arguments={"path": "src/app.py", "content": "VALUE = 1\n"},
                    )
                ]
            ),
            ModelResponse(final_text="Changed app.py."),
        ]
    )

    stream = StringIO()
    result = AgentLoop(
        task="Update app.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "no-verification-gate" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "no-verification-gate-memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
        output_sink=TextOutputSink(stream=stream),
        stream_model=True,
    ).run()

    assert result.status == "completed"
    assert result.stop_reason == "final_text"
    assert result.final_text == "Changed app.py."
    assert len(client.calls) == 2
    assert stream.getvalue().count("Changed app.py.") == 1


def test_agent_loop_exposes_skill_catalog_and_loads_full_skill_on_request(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="load_repo_skill",
                        name="read",
                        arguments={"source": "skill", "target": "repo-explain"},
                    )
                ]
            ),
            ModelResponse(final_text="Skill loaded."),
        ]
    )

    result = AgentLoop(
        task="Explain repository",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "skill-load" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "skill-load-memory"),
        skill_names=["repo-explain"],
    ).run()

    first_messages, first_tools = client.calls[0]
    first_prompt = "\n".join(str(message.get("content") or "") for message in first_messages)
    assert result.status == "completed"
    assert "read" in _tool_names(first_tools)
    assert "load_skill" not in _tool_names(first_tools)
    assert "可用技能目录" in first_prompt
    assert "search shallowly before reading direct source" in first_prompt
    assert "## Evidence checklist" not in first_prompt
    assert "## Evidence checklist" in client.calls[1][0][-1]["content"]


def test_agent_loop_does_not_soft_compact_semantic_only_history_without_summary(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    history = [
        *[
            {
                "role": "user",
                "content": f"history {index} " + ("x" * 900),
            }
            for index in range(15)
        ],
        {"role": "user", "content": "original task"},
    ]
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=12000,
            reserved_output=0,
            soft_limit=0.10,
            hard_limit=0.95,
        ),
    )
    first_trace = tmp_path / "summary-first" / "trace.jsonl"
    client = ScriptedModelClient([ModelResponse(final_text="done")])

    result = AgentLoop(
        task="original task",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(first_trace),
        config=AgentLoopConfig(max_steps=1),
        context_preparer=preparer,
        memory_store=ProjectMemoryStore(tmp_path / "summary-memory"),
        initial_message_history=history,
        no_skills=True,
    ).run()

    checkpoint = CheckpointStore(first_trace.parent / "checkpoints").load_latest()
    assert result.status == "completed"
    assert len(client.calls) == 1
    assert checkpoint is not None
    assert checkpoint.model_call_count == 1
    assert "llm_history_summary_calls" not in checkpoint.model_dump()
    assert not any(
        event.get("type") == "context_compressed"
        for event in _trace_events(first_trace)
    )
    assert checkpoint.compaction_state.execution is None
    assert checkpoint.compaction_state.semantic is None
    assert checkpoint.message_history[:-1] == history


def test_default_agent_loop_skips_oversized_semantic_request_and_falls_back(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    history: list[dict[str, Any]] = []
    for index in range(10):
        history.extend(
            [
                {
                    "role": "user",
                    "content": f"old requirement {index} " + ("u" * 1_000),
                },
                {
                    "role": "assistant",
                    "content": f"old answer {index} " + ("a" * 1_000),
                },
            ]
        )
    history.append({"role": "user", "content": "current task"})
    trace_path = tmp_path / "semantic-default" / "trace.jsonl"
    client = SmallWindowScriptedModelClient(
        [ModelResponse(final_text="done")]
    )

    result = AgentLoop(
        task="current task",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        config=AgentLoopConfig(max_steps=1),
        memory_store=ProjectMemoryStore(tmp_path / "semantic-default-memory"),
        initial_message_history=history,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert len(client.calls) == 1
    assert client.calls[0][1]
    assert not any(
        str(message.get("content") or "").startswith("[MiniCode semantic history]")
        for message in client.calls[0][0]
    )
    events = _trace_events(trace_path)
    semantic_event = next(
        event
        for event in events
        if event.get("type") == "context_compressed"
        and event.get("reason") == "semantic_history"
    )
    assert semantic_event["details"]["attempted"] is False
    assert semantic_event["details"]["skipped"] is True
    assert semantic_event["details"]["success"] is False
    assert semantic_event["details"]["failure_reason"] == (
        "semantic_compaction_request_exceeds_context_window"
    )
    assert any(
        event.get("type") == "context_compressed"
        and event.get("reason") == "execution_history"
        and (event.get("details") or {}).get("phase") == "hard"
        for event in events
    )


def test_agent_loop_provider_request_keeps_previous_semantic_turn(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    history = [
        {"role": "user", "content": "分析 DiscountService，给出两个方案"},
        {
            "role": "assistant",
            "content": "方案一保持现状；方案二调整 DiscountPolicy。" + ("x" * 2_000),
        },
        {"role": "user", "content": "使用第二种方案"},
    ]
    client = ScriptedModelClient([ModelResponse(final_text="done")])

    result = AgentLoop(
        task="使用第二种方案",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "semantic" / "trace.jsonl"),
        config=AgentLoopConfig(max_steps=1),
        context_preparer=ContextPreparer(
            TokenBudget(
                context_budget=6_000,
                reserved_output=0,
                soft_limit=0.2,
                hard_limit=0.95,
            )
        ),
        memory_store=ProjectMemoryStore(tmp_path / "semantic-memory"),
        initial_message_history=history,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    provider_messages = client.calls[0][0]
    assert {"role": "user", "content": "分析 DiscountService，给出两个方案"} in provider_messages
    assert any(
        message.get("role") == "assistant"
        and str(message.get("content") or "").startswith("方案一保持现状")
        for message in provider_messages
    )
    assert {"role": "user", "content": "使用第二种方案"} in provider_messages


def test_agent_loop_compaction_keeps_successful_mcp_side_effect_record(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = InProcessMCPServer(
        "messages",
        [MCPToolSpec(server_name="messages", name="send")],
        {
            "send": lambda arguments: {
                "status": "success",
                "message_id": "M-1024",
                "body": "x" * 8_000,
            }
        },
    )
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_send",
                        name="mcp__messages__send",
                        arguments={"name": "incident"},
                    )
                ]
            ),
            ModelResponse(final_text="sent"),
        ]
    )

    result = AgentLoop(
        task="发送通知",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "mcp-history" / "trace.jsonl"),
        config=AgentLoopConfig(max_steps=2),
        context_preparer=ContextPreparer(
            TokenBudget(
                context_budget=6_000,
                reserved_output=0,
                soft_limit=0.2,
                hard_limit=0.95,
            )
        ),
        memory_store=ProjectMemoryStore(tmp_path / "mcp-history-memory"),
        mcp_manager=MCPManager([server]),
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    second_messages = client.calls[1][0]
    assert not any(
        str(message.get("content") or "").startswith("[MiniCode compacted execution]")
        for message in second_messages
    )
    side_effect_result = next(
        message
        for message in second_messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") == "call_send"
    )
    assert "M-1024" in side_effect_result["content"]
    assert '"status": "success"' in side_effect_result["content"]


def test_agent_loop_stops_impossible_prompt_before_provider_call(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "budget" / "trace.jsonl"
    client = ScriptedModelClient([ModelResponse(final_text="must not be called")])
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=20,
            reserved_output=0,
            soft_limit=0.5,
            hard_limit=0.75,
        )
    )

    result = AgentLoop(
        task="active task",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        config=AgentLoopConfig(max_steps=1),
        context_preparer=preparer,
        memory_store=ProjectMemoryStore(tmp_path / "budget-memory"),
        no_skills=True,
    ).run()

    assert result.status == "stopped"
    assert result.stop_reason == "prompt_budget_exceeded"
    assert client.calls == []
    budget_events = [
        event
        for event in _trace_events(trace_path)
        if event.get("type") == "context_budget_exceeded"
    ]
    assert len(budget_events) == 1
    assert budget_events[0]["token_estimate"] > budget_events[0]["hard_token_limit"]
    assert set(budget_events[0]["source_tokens"]) == {"system", "messages", "tools"}


def test_record_modified_files_skips_and_reuses_workspace_profile(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from minicode_harness.workspace import WorkspaceProfile

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cached_profile = WorkspaceProfile(
        languages=["Python"],
        build_systems=["pyproject"],
        preferred_verification_commands=["python -m pytest -q"],
    )
    refreshed_profile = WorkspaceProfile(
        languages=["Python"],
        build_systems=["pyproject"],
        preferred_verification_commands=["python -m pytest -q"],
    )
    scans: list[str] = []

    def fake_scan(path):
        scans.append(str(path))
        return refreshed_profile

    monkeypatch.setattr("minicode_harness.loop.scan_workspace_profile", fake_scan)

    loop = AgentLoop.__new__(AgentLoop)
    loop.workspace = workspace
    loop.workspace_generation = 0
    loop.run_state = RunState()
    loop.modified_files = []
    loop.preferred_verification_command = None
    loop._workspace_profile_cache = cached_profile
    loop._repository_structure_card_cache = "cached-card"
    loop.trace_writer = SimpleNamespace(write_event=lambda *args, **kwargs: None)

    loop._record_modified_files([])
    assert scans == []
    assert loop.workspace_generation == 0

    loop._record_modified_files(["src/app.py"])
    assert scans == []
    assert loop._workspace_profile_cache is cached_profile
    assert loop._repository_structure_card_cache == "cached-card"

    loop._record_modified_files(["pyproject.toml"])
    assert scans == [str(workspace)]
    assert loop._workspace_profile_cache is refreshed_profile
    assert loop._repository_structure_card_cache is None
