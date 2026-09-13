import json
import subprocess
from threading import Barrier, Lock
from time import sleep
from typing import Any

from minicode_harness.hooks import HookDecision, HookManager
from minicode_harness.loop import AgentLoop
from minicode_harness.mcp import (
    InProcessMCPServer,
    MCPManager,
    MCPToolAnnotations,
    MCPToolSpec,
)
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import (
    ModelCapabilities,
    ModelClient,
    ModelResponse,
    ModelUsage,
    NormalizedToolCall,
)
import minicode_harness.subagent as subagent_module
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.subagent import ReadonlySubagentRunner
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def call_request(self, request):
        self.calls.append((request.as_chat_messages(), request.tools))
        return self.responses.pop(0)


class ParallelSubagentClient(ModelClient):
    def __init__(
        self,
        barrier: Barrier,
        started: list[str],
        completed: list[str],
        lock: Lock,
    ) -> None:
        self.barrier = barrier
        self.started = started
        self.completed = completed
        self.lock = lock

    def call_request(self, request):
        messages = request.as_chat_messages()
        task = str(messages[-1]["content"])
        with self.lock:
            self.started.append(task)
        self.barrier.wait(timeout=2)
        if task.endswith("A"):
            sleep(0.05)
        with self.lock:
            self.completed.append(task)
        return ModelResponse(final_text=f"summary:{task}")


class RecordingHook:
    name = "recording"
    events = (
        "session_start",
        "before_context_build",
        "after_context_build",
        "before_model_call",
        "after_model_response",
        "pre_tool_use",
        "post_tool_use",
        "before_checkpoint",
        "stop",
        "error",
    )

    def __init__(self) -> None:
        self.seen: list[str] = []

    def handle(self, event):
        self.seen.append(event.name)
        return HookDecision.allow(hook_name=self.name)


def _tool_names(tools):
    return {tool["function"]["name"] for tool in tools}


def test_parent_agent_can_delegate_bounded_readonly_analysis(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello from repository\n", encoding="utf-8")
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="delegate",
                        name="delegate_task",
                        arguments={"task": "Read README.md and summarize it."},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="sub_read",
                        name="read",
                        arguments={"source": "workspace", "target": "README.md"},
                    )
                ]
            ),
            ModelResponse(final_text="README says hello from repository."),
            ModelResponse(final_text="The delegated analysis confirmed the README content."),
        ]
    )

    result = AgentLoop(
        task="Use a subagent to inspect README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert "delegate_task" in _tool_names(client.calls[0][1])
    assert "delegate_task" not in _tool_names(client.calls[1][1])
    subagent_messages = client.calls[1][0]
    assert [message["role"] for message in subagent_messages] == ["system", "user"]
    assert subagent_messages[-1] == {
        "role": "user",
        "content": "Read README.md and summarize it.",
    }
    parent_second_call = client.calls[3][0]
    assert "README says hello" in parent_second_call[-1]["content"]
    delegated_payload = json.loads(parent_second_call[-1]["content"])
    assert delegated_payload == {
        "status": "completed",
        "summary": "README says hello from repository.",
    }


def test_parent_agent_runs_same_response_subagents_in_parallel(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run" / "trace.jsonl"
    parent_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="delegate_a",
                        name="delegate_task",
                        arguments={"task": "inspect module A"},
                    ),
                    NormalizedToolCall(
                        id="delegate_b",
                        name="delegate_task",
                        arguments={"task": "inspect module B"},
                    ),
                ]
            ),
            ModelResponse(final_text="combined"),
        ]
    )
    barrier = Barrier(2)
    lock = Lock()
    started: list[str] = []
    completed: list[str] = []

    result = AgentLoop(
        task="Explore two modules",
        workspace=workspace,
        model_client=parent_client,
        subagent_model_client_factory=lambda: ParallelSubagentClient(
            barrier,
            started,
            completed,
            lock,
        ),
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert set(started) == {"inspect module A", "inspect module B"}
    assert completed == ["inspect module B", "inspect module A"]
    parent_second_call = parent_client.calls[1][0]
    tool_results = [message for message in parent_second_call if message["role"] == "tool"]
    assert len(tool_results) == 2
    assert "summary:inspect module A" in tool_results[0]["content"]
    assert "summary:inspect module B" in tool_results[1]["content"]

    parent_events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [event["type"] for event in parent_events]
    assert "tool_batch_started" in event_types
    assert "tool_batch_finished" in event_types
    assert {
        event["subagent_id"]
        for event in parent_events
        if event["type"] == "subagent_finished"
    } == {"call_01", "call_02"}
    for call_id in ("call_01", "call_02"):
        child_trace = tmp_path / "run" / "artifacts" / "subagents" / call_id / "trace.jsonl"
        assert child_trace.is_file()
        child_event_types = [
            json.loads(line)["type"]
            for line in child_trace.read_text(encoding="utf-8").splitlines()
        ]
        assert child_event_types[0] == "subagent_started"
        assert child_event_types[-1] == "subagent_finished"


def test_subagent_stops_before_model_call_when_cancelled(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="unused")])
    cancellation = CancellationToken()
    cancellation.cancel()
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "cancelled" / "trace.jsonl"),
        artifact_dir=tmp_path / "cancelled-artifacts",
        cancellation_token=cancellation,
    )

    result = runner.run("Inspect one module")

    assert result.status == "partial"
    assert result.summary == "子任务未完整完成，当前没有足够证据形成结论。"
    assert client.calls == []
    events = [
        json.loads(line)
        for line in (tmp_path / "cancelled" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["stop_reason"] == "cancelled"


def test_subagent_stops_at_total_duration_boundary(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="unused")])
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(subagent_module.time, "monotonic", lambda: next(ticks))
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "timeout" / "trace.jsonl"),
        artifact_dir=tmp_path / "timeout-artifacts",
        max_duration_seconds=1.0,
    )

    result = runner.run("Inspect one module")

    assert result.status == "partial"
    assert result.summary == "子任务未完整完成，当前没有足够证据形成结论。"
    assert client.calls == []
    events = [
        json.loads(line)
        for line in (tmp_path / "timeout" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["stop_reason"] == "timeout"


def test_subagent_budget_uses_model_capabilities_without_expanding_steps(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    qwen_client = ScriptedModelClient([ModelResponse(final_text="done")])
    qwen_client.capabilities = ModelCapabilities(
        context_window=131_072,
        max_output_tokens=8_192,
    )
    qwen_runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=qwen_client,
        trace_writer=TraceWriter(tmp_path / "qwen" / "trace.jsonl"),
        artifact_dir=tmp_path / "qwen-artifacts",
    )

    ollama_client = ScriptedModelClient([ModelResponse(final_text="done")])
    ollama_client.capabilities = ModelCapabilities(
        context_window=32_000,
        max_output_tokens=4_096,
    )
    ollama_runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=ollama_client,
        trace_writer=TraceWriter(tmp_path / "ollama" / "trace.jsonl"),
        artifact_dir=tmp_path / "ollama-artifacts",
    )

    assert qwen_runner.preparer.budget.context_budget == 64_000
    assert qwen_runner.preparer.budget.reserved_output == 8_192
    assert ollama_runner.preparer.budget.context_budget == 32_000
    assert ollama_runner.preparer.budget.reserved_output == 4_096
    assert qwen_runner.max_steps == ollama_runner.max_steps == 8
    assert qwen_runner.max_tool_calls == ollama_runner.max_tool_calls == 12


def test_subagent_prompt_requires_search_before_large_reads(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="done")])
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "search-first" / "trace.jsonl"),
        artifact_dir=tmp_path / "search-first-artifacts",
    )

    result = runner.run("Inspect one implementation")

    assert result.status == "completed"
    prompt = "\n".join(
        str(message.get("content") or "") for message in client.calls[0][0]
    )
    assert "只读代码分析子 Agent" in prompt
    assert "目标是回答这个问题，不是理解整个相关模块或穷举仓库" in prompt
    assert "文件或实现位置未知时先搜索定位" in prompt
    assert "不为完整性扫描整个目录" in prompt
    assert "足够时立即停止探索并返回结论" in prompt
    assert "基于实际工具结果" in prompt
    assert "可直接使用的结论" in prompt


def test_subagent_result_contains_only_status_and_summary(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="模块分析完成。")])
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "compact-result" / "trace.jsonl"),
        artifact_dir=tmp_path / "compact-result-artifacts",
    )

    result = runner.run("分析一个模块")

    assert result.model_dump(mode="json") == {
        "status": "completed",
        "summary": "模块分析完成。",
    }


def test_unknown_model_subagent_uses_default_32k_budget(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="done")])
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "default" / "trace.jsonl"),
        artifact_dir=tmp_path / "default-artifacts",
    )

    assert runner.preparer.budget.context_budget == 32_000
    assert runner.preparer.budget.reserved_output == 4_096


def test_subagent_schema_hides_destructive_mcp_tools(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = MCPManager(
        [
            InProcessMCPServer(
                "external",
                [
                    MCPToolSpec(
                        server_name="external",
                        name="lookup",
                        annotations=MCPToolAnnotations(read_only=True),
                    ),
                    MCPToolSpec(
                        server_name="external",
                        name="metadata",
                        annotations=MCPToolAnnotations(read_only=True),
                    ),
                    MCPToolSpec(
                        server_name="external",
                        name="delete",
                        annotations=MCPToolAnnotations(read_only=False, destructive=True),
                    ),
                ],
                {
                    "lookup": lambda arguments: "ok",
                    "metadata": lambda arguments: "metadata",
                    "delete": lambda arguments: "deleted",
                },
            )
        ],
        tool_filters={
            "external": (["metadata", "delete"], ["lookup"]),
        },
    )
    client = ScriptedModelClient([ModelResponse(final_text="done")])
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "subagent-mcp" / "trace.jsonl"),
        artifact_dir=tmp_path / "artifacts",
        mcp_manager=manager,
    )

    result = runner.run("Inspect external information")
    names = _tool_names(client.calls[0][1])

    assert result.status == "completed"
    assert result.summary == "done"
    assert "mcp__external__metadata" in names
    assert "mcp__external__lookup" not in names
    assert "mcp__external__delete" not in names


def test_subagent_traces_tool_start_before_result(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    trace_path = tmp_path / "trace" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read",
                        name="read",
                        arguments={"source": "workspace", "target": "README.md"},
                    )
                ],
                usage=ModelUsage(input_tokens=100, output_tokens=10, total_tokens=110),
            ),
            ModelResponse(
                final_text="done",
                usage=ModelUsage(input_tokens=120, output_tokens=5, total_tokens=125),
            ),
        ]
    )
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        artifact_dir=tmp_path / "artifacts",
    )

    result = runner.run("Inspect README.md")
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]

    assert result.status == "completed"
    assert event_types.index("subagent_tool_called") < event_types.index("subagent_tool_result")
    called = next(event for event in events if event["type"] == "subagent_tool_called")
    assert called["tool"] == "read"
    assert called["args"] == {"source": "workspace", "target": "README.md"}
    model_started = [event for event in events if event["type"] == "model_call_started"]
    model_finished = [event for event in events if event["type"] == "model_call_finished"]
    assert [event["model_call_index"] for event in model_started] == [1, 2]
    assert [event["phase"] for event in model_started] == ["explore", "explore"]
    assert [event["input_tokens"] for event in model_finished] == [100, 120]
    assert [event["output_tokens"] for event in model_finished] == [10, 5]


def test_subagent_reports_actual_no_action_stop_reason(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(),
            ModelResponse(final_text="当前没有工具证据，子任务未完成。"),
        ]
    )
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "no-action" / "trace.jsonl"),
        artifact_dir=tmp_path / "no-action-artifacts",
        max_steps=8,
    )

    result = runner.run("Inspect one module")

    assert result.status == "partial"
    assert result.summary == "当前没有工具证据，子任务未完成。"
    assert len(client.calls) == 2
    assert client.calls[-1][1] == []


def test_subagent_finalizes_once_after_max_steps(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    trace_path = tmp_path / "max-steps" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read_a",
                        name="read",
                        arguments={"source": "workspace", "target": "a.py"},
                    )
                ]
            ),
            ModelResponse(final_text="已确认 a.py 中 VALUE 为 1；其余部分未检查。"),
        ]
    )
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        artifact_dir=tmp_path / "max-steps-artifacts",
        max_steps=1,
    )

    result = runner.run("分析 VALUE")

    assert result.model_dump(mode="json") == {
        "status": "partial",
        "summary": "已确认 a.py 中 VALUE 为 1；其余部分未检查。",
    }
    assert len(client.calls) == 2
    assert client.calls[-1][1] == []
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    finished = [event for event in events if event["type"] == "subagent_finished"][-1]
    assert finished["stop_reason"] == "max_steps"
    assert finished["summary_source"] == "model"


def test_subagent_summarizes_existing_evidence_when_tool_budget_is_exhausted(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    trace_path = tmp_path / "budget" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read_1",
                        name="read",
                        arguments={"source": "workspace", "target": "a.py"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read_2",
                        name="read",
                        arguments={"source": "workspace", "target": "b.py"},
                    )
                ]
            ),
            ModelResponse(
                final_text="Confirmed from a.py: VALUE is 1. b.py was not inspected.",
                usage=ModelUsage(input_tokens=160, output_tokens=18, total_tokens=178),
            ),
        ]
    )
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        artifact_dir=tmp_path / "budget-artifacts",
        max_tool_calls=1,
    )

    result = runner.run("Inspect two files")

    assert result.status == "partial"
    assert "VALUE is 1" in result.summary
    assert len(client.calls) == 3
    assert client.calls[-1][1] == []
    assert any(message["role"] == "tool" for message in client.calls[-1][0])

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(
        event["type"] == "model_call_started" and event["phase"] == "finalize"
        for event in events
    )
    assert any(event["type"] == "subagent_summary_finalized" for event in events)
    finished = [event for event in events if event["type"] == "subagent_finished"][-1]
    assert finished["summary_finalized"] is True


def test_subagent_uses_deterministic_evidence_when_final_only_call_fails(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pom.xml").write_text("<project/>\n", encoding="utf-8")
    (workspace / "empty").mkdir()
    deep_file = workspace / "deep" / "a" / "b" / "Only.java"
    deep_file.parent.mkdir(parents=True)
    deep_file.write_text("class Only {}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    trace_path = tmp_path / "deterministic" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="root_pom",
                        name="search",
                        arguments={
                            "kind": "files",
                            "query": "**/pom.xml",
                            "path": ".",
                        },
                    ),
                    NormalizedToolCall(
                        id="empty_java",
                        name="search",
                        arguments={
                            "kind": "files",
                            "query": "**/*.java",
                            "path": "empty",
                        },
                    ),
                    NormalizedToolCall(
                        id="missing_java",
                        name="search",
                        arguments={
                            "kind": "files",
                            "query": "**/*.java",
                            "path": "missing",
                        },
                    ),
                    NormalizedToolCall(
                        id="deep_java",
                        name="search",
                        arguments={
                            "kind": "files",
                            "query": "**/*.java",
                            "path": "deep",
                            "max_depth": 1,
                        },
                    ),
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="over_budget",
                        name="search",
                        arguments={"kind": "files", "query": "*", "path": "."},
                    )
                ]
            ),
        ]
    )
    runner = ReadonlySubagentRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        artifact_dir=tmp_path / "deterministic-artifacts",
        max_tool_calls=4,
    )

    result = runner.run("Inspect repository layout")

    assert result.status == "partial"
    assert "Subagent stopped before producing a final answer" not in result.summary
    assert 'path="."' in result.summary
    assert "pom.xml" in result.summary
    assert 'path="empty"' in result.summary
    assert "路径存在，当前范围内无匹配" in result.summary
    assert 'path="missing"' in result.summary
    assert "路径不存在" in result.summary
    deep_line = next(line for line in result.summary.splitlines() if 'path="deep"' in line)
    assert "路径存在，当前范围内无匹配" in deep_line

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    failed = [event for event in events if event["type"] == "subagent_summary_failed"]
    assert failed[-1]["reason"] == "model_error"
    finished = [event for event in events if event["type"] == "subagent_finished"][-1]
    assert finished["summary_finalized"] is False
    assert finished["summary_source"] == "deterministic_evidence"


def test_agent_loop_emits_complete_hook_lifecycle(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    hook = RecordingHook()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read",
                        name="read",
                        arguments={"source": "workspace", "target": "README.md"},
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )

    result = AgentLoop(
        task="Read README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "hooks" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        hook_manager=HookManager([hook]),
        no_skills=True,
    ).run()

    assert result.status == "completed"
    for expected in (
        "session_start",
        "before_context_build",
        "after_context_build",
        "before_model_call",
        "after_model_response",
        "pre_tool_use",
        "post_tool_use",
        "before_checkpoint",
        "stop",
    ):
        assert expected in hook.seen
