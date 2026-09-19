from __future__ import annotations

import json

import pytest

from minicode_harness.loop import AgentLoop
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.policy import (
    ExecutionMode,
    PermissionDecision,
    check_command_allowed,
    decide_permission,
)
from minicode_harness.state import ApprovalDecision, StaticApprovalClient
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.tools import CommandRunResult
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)

    def call_request(self, request):
        return self.responses.pop(0)


class RecordingCommandExecutor:
    sandboxed = False
    command_rules = ()

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def classify(self, argv):
        return check_command_allowed(argv, sandboxed=False, rules=self.command_rules)

    def execute(
        self,
        workspace,
        argv,
        timeout_seconds,
        cancellation_token=None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        del workspace, cancellation_token
        normalized = list(argv)
        self.calls.append((normalized, approval_granted))
        return CommandRunResult(
            argv=normalized,
            command=" ".join(normalized),
            returncode=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.01,
            timeout_seconds=timeout_seconds,
            allowlist_rule="test",
        )


@pytest.mark.parametrize(
    ("tool_name", "requires_approval", "execution_mode", "expected"),
    [
        ("read", False, ExecutionMode.DEFAULT, PermissionDecision.ALLOW),
        ("write", True, ExecutionMode.DEFAULT, PermissionDecision.ALLOW),
        ("run_command", True, ExecutionMode.DEFAULT, PermissionDecision.ASK),
        ("write", True, ExecutionMode.REVIEW_CHANGES, PermissionDecision.ASK),
        ("run_command", True, ExecutionMode.REVIEW_CHANGES, PermissionDecision.ASK),
        ("write", True, ExecutionMode.FULL_ACCESS, PermissionDecision.ALLOW),
        ("run_command", True, ExecutionMode.FULL_ACCESS, PermissionDecision.ALLOW),
    ],
)
def test_permission_decision_matrix(
    tool_name,
    requires_approval,
    execution_mode,
    expected,
) -> None:
    assert decide_permission(
        tool_name=tool_name,
        requires_approval=requires_approval,
        execution_mode=execution_mode,
    ) == expected


def test_default_mode_allows_workspace_mutation_without_prompt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval = StaticApprovalClient(ApprovalDecision.REJECT)
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_write",
                        name="write",
                        arguments={"path": "README.md", "content": "hello\n"},
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )

    result = AgentLoop(
        task="Create README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "default-write" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        execution_mode=ExecutionMode.DEFAULT,
        approval_client=approval,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert approval.requests == []


def test_review_changes_requires_workspace_mutation_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval = StaticApprovalClient(ApprovalDecision.REJECT)
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_write",
                        name="write",
                        arguments={"path": "README.md", "content": "blocked\n"},
                    )
                ]
            ),
            ModelResponse(final_text="blocked"),
        ]
    )

    result = AgentLoop(
        task="Create README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "review-write" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        execution_mode=ExecutionMode.REVIEW_CHANGES,
        approval_client=approval,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert not (workspace / "README.md").exists()
    assert len(approval.requests) == 1


def test_full_access_runs_admitted_host_command_without_prompt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval = StaticApprovalClient(ApprovalDecision.REJECT)
    executor = RecordingCommandExecutor()
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_cmd",
                        name="run_command",
                        arguments={"argv": ["python", "-c", "print('ok')"]},
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )

    result = AgentLoop(
        task="Run diagnostic",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "full-access" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        execution_mode=ExecutionMode.FULL_ACCESS,
        approval_client=approval,
        command_executor=executor,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert approval.requests == []
    assert executor.calls == [(["python", "-c", "print('ok')"], True)]


def test_full_access_cannot_bypass_deterministic_command_deny(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval = StaticApprovalClient(ApprovalDecision.APPROVE)
    executor = RecordingCommandExecutor()
    trace_path = tmp_path / "full-access-deny" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_cmd",
                        name="run_command",
                        arguments={"argv": ["bash", "-lc", "whoami"]},
                    )
                ]
            ),
            ModelResponse(final_text="blocked"),
        ]
    )

    result = AgentLoop(
        task="Run denied shell wrapper",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        execution_mode=ExecutionMode.FULL_ACCESS,
        approval_client=approval,
        command_executor=executor,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert approval.requests == []
    assert executor.calls == []
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    result_event = next(
        event
        for event in events
        if event["type"] == "tool_result" and event.get("tool") == "run_command"
    )
    assert result_event["status"] == "command_rejected_by_policy"
