from __future__ import annotations

import json

import pytest

from minicode_harness.loop import AgentLoop
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.policy import (
    ApprovalPolicy,
    PermissionDecision,
    PermissionMode,
    check_command_allowed,
    decide_permission,
)
from minicode_harness.state import ApprovalDecision, StaticApprovalClient
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
    (
        "tool_name",
        "requires_approval",
        "is_command",
        "approval_policy",
        "permission_mode",
        "expected",
    ),
    [
        (
            "read",
            False,
            False,
            ApprovalPolicy.ON_REQUEST,
            PermissionMode.READ_ONLY,
            PermissionDecision.ALLOW,
        ),
        (
            "write",
            True,
            False,
            ApprovalPolicy.ON_REQUEST,
            PermissionMode.READ_ONLY,
            PermissionDecision.ASK,
        ),
        (
            "write",
            True,
            False,
            ApprovalPolicy.NEVER,
            PermissionMode.READ_ONLY,
            PermissionDecision.DENY,
        ),
        (
            "write",
            True,
            False,
            ApprovalPolicy.ON_REQUEST,
            PermissionMode.WORKSPACE_WRITE,
            PermissionDecision.ALLOW,
        ),
        (
            "write",
            True,
            False,
            ApprovalPolicy.NEVER,
            PermissionMode.WORKSPACE_WRITE,
            PermissionDecision.ALLOW,
        ),
        (
            "run_command",
            True,
            True,
            ApprovalPolicy.ON_REQUEST,
            PermissionMode.WORKSPACE_WRITE,
            PermissionDecision.ASK,
        ),
        (
            "run_command",
            True,
            True,
            ApprovalPolicy.NEVER,
            PermissionMode.WORKSPACE_WRITE,
            PermissionDecision.DENY,
        ),
        (
            "run_command",
            True,
            True,
            ApprovalPolicy.ON_REQUEST,
            PermissionMode.FULL_ACCESS,
            PermissionDecision.ASK,
        ),
        (
            "run_command",
            True,
            True,
            ApprovalPolicy.NEVER,
            PermissionMode.FULL_ACCESS,
            PermissionDecision.DENY,
        ),
        (
            "write",
            True,
            False,
            ApprovalPolicy.ON_REQUEST,
            PermissionMode.FULL_ACCESS,
            PermissionDecision.ALLOW,
        ),
    ],
)
def test_permission_decision_matrix(
    tool_name,
    requires_approval,
    is_command,
    approval_policy,
    permission_mode,
    expected,
) -> None:
    assert decide_permission(
        tool_name=tool_name,
        requires_approval=requires_approval,
        is_command=is_command,
        approval_policy=approval_policy,
        permission_mode=permission_mode,
    ) == expected


def test_workspace_write_allows_workspace_mutation_without_prompt(tmp_path) -> None:
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
        trace_writer=TraceWriter(tmp_path / "workspace-write" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_policy=ApprovalPolicy.ON_REQUEST,
        permission_mode=PermissionMode.WORKSPACE_WRITE,
        approval_client=approval,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert approval.requests == []


def test_read_only_never_denies_mutation_without_prompt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval = StaticApprovalClient(ApprovalDecision.APPROVE)
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
        trace_writer=TraceWriter(tmp_path / "read-only-never" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_policy=ApprovalPolicy.NEVER,
        permission_mode=PermissionMode.READ_ONLY,
        approval_client=approval,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert not (workspace / "README.md").exists()
    assert approval.requests == []


def test_full_access_never_does_not_bypass_command_approval_requirement(tmp_path) -> None:
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
                        arguments={
                            "argv": [
                                "python",
                                "-c",
                                "open('diagnostic.txt', 'w').write('x')",
                            ]
                        },
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
        trace_writer=TraceWriter(tmp_path / "full-access-never" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_policy=ApprovalPolicy.NEVER,
        permission_mode=PermissionMode.FULL_ACCESS,
        approval_client=approval,
        command_executor=executor,
        no_skills=True,
    ).run()

    assert result.status == "completed"
    assert approval.requests == []
    assert executor.calls == []


def test_full_access_never_cannot_bypass_deterministic_command_deny(tmp_path) -> None:
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
        approval_policy=ApprovalPolicy.NEVER,
        permission_mode=PermissionMode.FULL_ACCESS,
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
