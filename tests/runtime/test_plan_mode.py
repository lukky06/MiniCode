from __future__ import annotations

from minicode_harness.loop import AgentLoop
from minicode_harness.mcp import (
    InProcessMCPServer,
    MCPManager,
    MCPToolAnnotations,
    MCPToolSpec,
)
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.output import NullOutputSink
from minicode_harness.resume import resume_run
from minicode_harness.runtime import CollaborationMode
from minicode_harness.runtime.run_executor import RunExecutionRequest, RunExecutor
from minicode_harness.state import (
    ApprovalDecision,
    StaticUserInputClient,
    ExecutionJournal,
    RunStore,
    StaticApprovalClient,
)
from minicode_harness.tools import ToolRegistry
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def call_request(self, request):
        self.calls.append((list(request.as_chat_messages()), list(request.tools)))
        return self.responses.pop(0)


def _tool_names(tools: list[dict]) -> set[str]:
    return {str(tool["function"]["name"]) for tool in tools}


def _system_text(messages: list[dict]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
    )


def test_plan_mode_exposes_only_readonly_tools_and_dynamic_plan_prompt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="implementation plan")])

    user_input = StaticUserInputClient(selected_index=1)
    result = AgentLoop(
        task="Plan a repository change",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "plan" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        collaboration_mode=CollaborationMode.PLAN,
        user_input_client=user_input,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    names = _tool_names(client.calls[0][1])
    assert {
        "read",
        "search",
        "task",
        "delegate_task",
        "runtime_task_status",
        "request_user_input",
    }.issubset(names)
    assert {
        "edit",
        "write",
        "apply_patch",
        "run_command",
        "delegate_worktree",
        "runtime_task_stop",
    }.isdisjoint(names)
    system = _system_text(client.calls[0][0])
    assert "Plan Mode" in system
    assert "不得修改工作区" in system
    assert "request_user_input" in system


def test_default_mode_does_not_expose_request_user_input(tmp_path) -> None:
    workspace = tmp_path / "workspace-default-input"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="done")])

    AgentLoop(
        task="Normal coding run",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "default-input" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory-default-input"),
        user_input_client=StaticUserInputClient(selected_index=0),
        no_skills=True,
    ).run()

    assert "request_user_input" not in _tool_names(client.calls[0][1])


def test_plan_mode_user_input_selection_returns_to_model(tmp_path) -> None:
    workspace = tmp_path / "workspace-choice"
    workspace.mkdir()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_input",
                        name="request_user_input",
                        arguments={
                            "question": "Choose compatibility strategy",
                            "options": [
                                {"label": "strict", "description": "Break old callers"},
                                {"label": "compat", "description": "Keep compatibility"},
                            ],
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Use compatibility strategy."),
        ]
    )
    user_input = StaticUserInputClient(selected_index=1)

    result = AgentLoop(
        task="Plan compatibility work",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "choice" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory-choice"),
        collaboration_mode=CollaborationMode.PLAN,
        user_input_client=user_input,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert len(user_input.requests) == 1
    tool_messages = [
        message
        for message in client.calls[1][0]
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "compat" in str(tool_messages[0]["content"])


def test_plan_mode_rejects_model_attempt_to_call_hidden_write_tool(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval = StaticApprovalClient(ApprovalDecision.APPROVE)
    trace_path = tmp_path / "hidden-write" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_write",
                        name="write",
                        arguments={"path": "README.md", "content": "must not land\n"},
                    )
                ]
            ),
            ModelResponse(final_text="write was unavailable"),
        ]
    )

    result = AgentLoop(
        task="Plan only",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        collaboration_mode=CollaborationMode.PLAN,
        approval_client=approval,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert not (workspace / "README.md").exists()
    assert approval.requests == []
    journal = ExecutionJournal(trace_path.parent / "execution-journal.jsonl")
    assert journal.load_events() == []
    tool_messages = [
        message
        for message in client.calls[1][0]
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "tool_unavailable" in str(tool_messages[0]["content"])


def test_default_mode_keeps_existing_write_and_command_surface(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = ScriptedModelClient([ModelResponse(final_text="done")])

    result = AgentLoop(
        task="Normal coding run",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "default" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    names = _tool_names(client.calls[0][1])
    assert {"edit", "write", "apply_patch", "run_command"}.issubset(names)
    assert "Plan Mode" not in _system_text(client.calls[0][0])


def test_plan_safe_tool_filter_keeps_readonly_mcp_and_hides_side_effect_mcp(
    tmp_path,
) -> None:
    server = InProcessMCPServer(
        "support",
        [
            MCPToolSpec(
                server_name="support",
                name="search_docs",
                annotations=MCPToolAnnotations(read_only=True, destructive=False),
            ),
            MCPToolSpec(
                server_name="support",
                name="create_ticket",
                annotations=MCPToolAnnotations(read_only=False, destructive=False),
            ),
        ],
        {
            "search_docs": lambda arguments: arguments,
            "create_ticket": lambda arguments: arguments,
        },
    )
    registry = ToolRegistry(
        str(tmp_path),
        enable_write=True,
        mcp_manager=MCPManager([server]),
    )

    names = set(registry.readonly_tool_names())

    assert "mcp__support__search_docs" in names
    assert "mcp__support__create_ticket" not in names
    assert "write" not in names
    assert "run_command" not in names


def test_run_executor_persists_plan_collaboration_mode(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")

    result = RunExecutor(run_store=run_store).execute(
        RunExecutionRequest(
            task="plan",
            workspace=workspace,
            dry_run=True,
            collaboration_mode=CollaborationMode.PLAN,
        ),
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
    )

    session = run_store.load_session(result.run_id)
    assert session.collaboration_mode == "plan"


def test_resume_restores_plan_mode_into_model_request(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="continue planning",
        workspace=workspace,
        run_id="run_20260907_950",
        no_write=False,
        collaboration_mode="plan",
        repository_memory_enabled=False,
    )
    client = ScriptedModelClient([ModelResponse(final_text="resumed plan")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
    )

    assert result.status == "completed"
    assert "Plan Mode" in _system_text(client.calls[0][0])
    names = _tool_names(client.calls[0][1])
    assert {"write", "apply_patch", "run_command"}.isdisjoint(names)
