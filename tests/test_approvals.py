import json
import sys

import minicode_harness.tools.registry as registry_module
from minicode_harness.hooks import HookDecision, HookEvent, HookManager
from minicode_harness.loop import AgentLoop
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.state import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStore,
    ReplSessionMemory,
    StaticApprovalClient,
)
from minicode_harness.tools import CommandRunResult, ToolRegistry
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def call_request(self, request):
        return self.responses.pop(0)


class RecordingPreToolHook:
    name = "recording_pre_tool"
    events = ("pre_tool_use",)

    def __init__(self) -> None:
        self.arguments: list[dict[str, object]] = []

    @property
    def called(self) -> bool:
        return bool(self.arguments)

    def handle(self, event: HookEvent) -> HookDecision:
        tool_call = event.payload["tool_call"]
        self.arguments.append(dict(tool_call.arguments))
        return HookDecision.allow(hook_name=self.name)


class RecordingCommandExecutor:
    def __init__(self, *, sandboxed: bool = False) -> None:
        self.sandboxed = sandboxed
        self.command_rules = ()
        self.calls: list[tuple[list[str], bool]] = []

    def execute(
        self,
        workspace,
        argv,
        timeout_seconds,
        cancellation_token=None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        _ = (workspace, cancellation_token)
        normalized_argv = list(argv)
        self.calls.append((normalized_argv, approval_granted))
        return CommandRunResult(
            argv=normalized_argv,
            command=" ".join(normalized_argv),
            returncode=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.01,
            timeout_seconds=timeout_seconds,
            allowlist_rule="test",
        )


def test_approval_store_persists_and_clears_pending_request(tmp_path) -> None:
    store = ApprovalStore(tmp_path / "approvals")
    request = ApprovalRequest(
        id="approval_1",
        tool_call_id="call_1",
        tool_name="write",
        risk_level="medium",
        arguments={"path": "README.md"},
        preview={"summary": {"path": "README.md"}},
    )

    store.save_pending(request)
    loaded = store.load_pending()

    assert loaded is not None
    assert loaded.id == "approval_1"
    assert loaded.preview["summary"]["path"] == "README.md"

    store.clear_pending()
    assert store.load_pending() is None


def test_tool_registry_builds_approval_previews(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(str(workspace), enable_write=True)
    patch = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""

    patch_preview = registry.preview_admitted(
        registry.admit("apply_patch", {"patch": patch})
    )
    command_arguments = {
        "argv": ["python", "-m", "pytest", "-q", "tests/test_service.py"],
        "timeout_seconds": 30,
    }
    command_preview = registry.preview_admitted(
        registry.admit("run_command", command_arguments)
    )
    diagnostic_arguments = {
        "argv": ["python", "-c", "open('diagnostic.txt', 'w').write('x')"],
        "timeout_seconds": 30,
    }
    diagnostic_preview = registry.preview_admitted(
        registry.admit("run_command", diagnostic_arguments)
    )

    assert patch_preview["files"] == ["README.md"]
    assert patch_preview["summary"]["changed_lines"] == 2
    assert command_preview["allowed"] is True
    assert command_preview["policy_action"] == "require_approval"
    assert command_preview["allowlist_rule"] == "local execution default"
    assert command_preview["timeout_seconds"] == 30
    assert registry.requires_approval("run_command", command_arguments) is True
    assert diagnostic_preview["policy_action"] == "require_approval"
    assert diagnostic_preview["policy_category"] == "unknown"
    assert diagnostic_preview["effects"]
    assert registry.requires_approval("run_command", diagnostic_arguments) is True


def test_agent_loop_approves_write_file_and_clears_pending_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "context.txt").write_text("existing context\n", encoding="utf-8")
    trace_path = tmp_path / "run" / "trace.jsonl"
    approval_store = ApprovalStore(tmp_path / "run" / "approvals")
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="read",
                        arguments={"source": "workspace", "target": "context.txt"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_2",
                        name="write",
                        arguments={"path": "README.md", "content": "hello\n"},
                    )
                ]
            ),
            ModelResponse(final_text="Wrote README.md."),
        ]
    )

    result = AgentLoop(
        task="Write README",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
        approval_store=approval_store,
    ).run()

    assert result.status == "completed"
    assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert approval_store.load_pending() is None
    assert approval_client.requests[0].tool_name == "write"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert "approval_required" in [event["type"] for event in events]
    assert "approval_resolved" in [event["type"] for event in events]


def test_session_command_grant_skips_repeated_approval_for_same_command(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "session-grant" / "trace.jsonl"
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE_SESSION)
    command_executor = RecordingCommandExecutor()
    session_memory = ReplSessionMemory(workspace=str(workspace.resolve()))
    argv = [sys.executable, "-c", "open('a.txt','w').write('a')"]
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={"argv": argv, "timeout_seconds": 30},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_2",
                        name="run_command",
                        arguments={"argv": argv, "timeout_seconds": 30},
                    )
                ]
            ),
            ModelResponse(final_text="Commands completed."),
        ]
    )

    result = AgentLoop(
        task="Run the same approved diagnostic twice",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        session_memory=session_memory,
        command_executor=command_executor,
        approval_client=approval_client,
        enable_write=True,
    ).run()

    assert result.status == "completed"
    assert len(approval_client.requests) == 1
    assert len(command_executor.calls) == 2
    assert len(session_memory.command_approval_grants) == 1
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "approval_session_granted" for event in events)
    assert any(event["type"] == "approval_session_grant_used" for event in events)


def test_session_command_grant_does_not_cover_different_interpreter_payload(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE_SESSION)
    command_executor = RecordingCommandExecutor()
    session_memory = ReplSessionMemory(workspace=str(workspace.resolve()))
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={
                            "argv": [sys.executable, "-c", "open('a.txt','w').write('a')"],
                            "timeout_seconds": 30,
                        },
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_2",
                        name="run_command",
                        arguments={
                            "argv": [sys.executable, "-c", "open('b.txt','w').write('b')"],
                            "timeout_seconds": 30,
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Commands completed."),
        ]
    )

    result = AgentLoop(
        task="Run two distinct approved diagnostics",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(tmp_path / "distinct-session-grant" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        session_memory=session_memory,
        command_executor=command_executor,
        approval_client=approval_client,
        enable_write=True,
    ).run()

    assert result.status == "completed"
    assert len(approval_client.requests) == 2
    assert len(command_executor.calls) == 2
    assert len(session_memory.command_approval_grants) == 2


def test_agent_loop_rejects_write_file_without_modifying_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "context.txt").write_text("existing context\n", encoding="utf-8")
    trace_path = tmp_path / "run" / "trace.jsonl"
    approval_store = ApprovalStore(tmp_path / "run" / "approvals")
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="read",
                        arguments={"source": "workspace", "target": "context.txt"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_2",
                        name="write",
                        arguments={"path": "README.md", "content": "hello\n"},
                    )
                ]
            ),
            ModelResponse(final_text="The write was rejected, so no file was changed."),
        ]
    )

    result = AgentLoop(
        task="Write README",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.REJECT),
        approval_store=approval_store,
    ).run()

    assert result.status == "completed"
    assert result.stop_reason == "final_text"
    assert not (workspace / "README.md").exists()
    assert approval_store.load_pending() is None


def test_agent_loop_skips_write_file_without_modifying_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "skip" / "trace.jsonl"
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="write",
                        arguments={"path": "README.md", "content": "hello\n"},
                    )
                ]
            ),
            ModelResponse(final_text="The write was skipped, so no file was changed."),
        ]
    )

    result = AgentLoop(
        task="Write README",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.SKIP),
    ).run()

    assert result.status == "completed"
    assert not (workspace / "README.md").exists()
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    resolved = [event for event in events if event["type"] == "approval_resolved"]
    results = [event for event in events if event["type"] == "tool_result"]
    assert resolved[0]["decision"] == "skip"
    assert results[0]["status"] == "approval_skip"


def test_explicit_approval_abort_stops_the_run(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="write",
                        arguments={"path": "README.md", "content": "hello\n"},
                    )
                ]
            )
        ]
    )

    result = AgentLoop(
        task="Write README",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(tmp_path / "abort" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.ABORT),
    ).run()

    assert result.status == "stopped"
    assert result.stop_reason == "approval_aborted"
    assert not (workspace / "README.md").exists()


def test_repeated_identical_denied_tool_call_is_not_prompted_again(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval_client = StaticApprovalClient(ApprovalDecision.REJECT)
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="write",
                        arguments={"path": "README.md", "content": "hello\n"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_2",
                        name="write",
                        arguments={"path": "README.md", "content": "hello\n"},
                    )
                ]
            ),
            ModelResponse(final_text="The repeated denied write was not executed."),
        ]
    )

    result = AgentLoop(
        task="Write README",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
    ).run()

    assert result.status == "completed"
    assert len(approval_client.requests) == 1
    assert not (workspace / "README.md").exists()


def test_invalid_tool_arguments_do_not_reach_hook_approval_or_executor(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "invalid" / "trace.jsonl"
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)
    executor = RecordingCommandExecutor()
    hook = RecordingPreToolHook()
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={"argv": ["python", 123]},
                    )
                ]
            ),
            ModelResponse(final_text="Correctable invalid tool arguments were reported."),
        ]
    )

    loop = AgentLoop(
        task="Run focused verification",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
        command_executor=executor,
        hook_manager=HookManager([hook]),
    )
    result = loop.run()

    assert result.status == "completed"
    assert hook.called is False
    assert approval_client.requests == []
    assert executor.calls == []
    tool_message = next(
        message
        for message in loop.user_turn.snapshot_messages()
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_1"
    )
    payload = json.loads(tool_message["content"])
    assert payload["status"] == "invalid_tool_arguments"
    assert payload["error_type"] == "invalid_tool_arguments"
    assert payload["retryable"] is True
    assert payload["side_effect"] == "none"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    invalid = [event for event in events if event["type"] == "tool_arguments_invalid"]
    assert invalid[0]["tool"] == "run_command"
    assert invalid[0]["error_locations"] == ["argv.1"]
    assert not any(event["type"] == "approval_required" for event in events)


def test_malformed_tool_arguments_raise_output_budget_and_bypass_execution(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "malformed" / "trace.jsonl"
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)
    hook = RecordingPreToolHook()
    raw_arguments = '{"path":"generate_test_files.py","content":"abc\\'
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_bad",
                        name="write",
                        arguments={},
                        raw_arguments=raw_arguments,
                        argument_parse_error="Unterminated string",
                        arguments_likely_truncated=True,
                    )
                ]
            ),
            ModelResponse(final_text="The malformed tool call was reported."),
        ]
    )
    model_client.configure_capabilities("qwen", "qwen-plus")

    loop = AgentLoop(
        task="Create test data",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
        hook_manager=HookManager([hook]),
    )
    result = loop.run()

    assert result.status == "completed"
    assert loop.output_budget.current == 8192
    assert hook.called is False
    assert approval_client.requests == []
    assert not (workspace / "generate_test_files.py").exists()
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    malformed = [event for event in events if event["type"] == "tool_arguments_malformed"]
    assert malformed[0]["repeat_count"] == 1
    assert malformed[0]["likely_truncated"] is True
    assert malformed[0]["raw_argument_chars"] == len(raw_arguments)
    assert any(
        event["type"] == "model_recovery"
        and event["action"] == "increase_max_output_tokens_for_malformed_tool_call"
        for event in events
    )


def test_repeated_malformed_tool_arguments_stop_after_three_attempts(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "repeated_malformed" / "trace.jsonl"
    raw_prefix = '{"path":"generate_test_files.py","content":"' + ("x" * 1100)

    def malformed_call(call_id: str, suffix: str) -> ModelResponse:
        return ModelResponse(
            tool_calls=[
                NormalizedToolCall(
                    id=call_id,
                    name="write",
                    arguments={},
                    raw_arguments=raw_prefix + suffix,
                    argument_parse_error="Unterminated string",
                    arguments_likely_truncated=True,
                )
            ]
        )

    result = AgentLoop(
        task="Create test data",
        workspace=workspace,
        model_client=ScriptedModelClient(
            [
                malformed_call("call_1", "a"),
                malformed_call("call_2", "bb"),
                malformed_call("call_3", "ccc"),
            ]
        ),
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
    ).run()

    assert result.status == "stopped"
    assert result.stop_reason == "repeated_malformed_tool_call"
    assert result.tool_calls == 3
    assert not (workspace / "generate_test_files.py").exists()
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    malformed = [event for event in events if event["type"] == "tool_arguments_malformed"]
    assert [event["repeat_count"] for event in malformed] == [1, 2, 3]
    assert len({event["raw_argument_sha256"] for event in malformed}) == 3
    assert len({event["fingerprint"] for event in malformed}) == 1


def test_hook_receives_validated_run_command_arguments(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "validated-command" / "trace.jsonl"
    executor = RecordingCommandExecutor(sandboxed=True)
    hook = RecordingPreToolHook()
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={"argv": ["python", "-m", "pytest", "-q"]},
                    )
                ]
            ),
            ModelResponse(final_text="Verification completed."),
        ]
    )

    result = AgentLoop(
        task="Run focused verification",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        command_executor=executor,
        hook_manager=HookManager([hook]),
    ).run()

    assert result.status == "completed"
    assert hook.arguments == [
        {
            "argv": ["python", "-m", "pytest", "-q"],
            "timeout_seconds": 120,
            "background": False,
        }
    ]
    assert executor.calls == [(["python", "-m", "pytest", "-q"], True)]
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "tool_arguments_validated" for event in events)


def test_agent_loop_runs_safe_verification_without_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)
    executor = RecordingCommandExecutor(sandboxed=True)
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={"argv": ["python", "-m", "pytest", "-q", "tests/test_service.py"]},
                    )
                ]
            ),
            ModelResponse(final_text="Verification completed."),
        ]
    )

    result = AgentLoop(
        task="Run focused verification",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(tmp_path / "safe" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
        command_executor=executor,
    ).run()

    assert result.status == "completed"
    assert approval_client.requests == []
    assert executor.calls == [
        (["python", "-m", "pytest", "-q", "tests/test_service.py"], True)
    ]


def test_agent_loop_reuses_one_command_admission_across_governance(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)
    executor = RecordingCommandExecutor()
    policy_calls: list[tuple[str, ...]] = []
    original_check = registry_module.check_command_allowed

    def counting_check(argv, *, sandboxed=False, rules=()):
        policy_calls.append(tuple(argv))
        return original_check(argv, sandboxed=sandboxed, rules=rules)

    monkeypatch.setattr(registry_module, "check_command_allowed", counting_check)
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={
                            "argv": ["python", "-c", "open('diagnostic.txt', 'w').write('x')"]
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Diagnostic completed."),
        ]
    )

    result = AgentLoop(
        task="Run a diagnostic",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(tmp_path / "single-admission" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
        command_executor=executor,
    ).run()

    assert result.status == "completed"
    assert policy_calls == [
        ("python", "-c", "open('diagnostic.txt', 'w').write('x')")
    ]
    assert len(approval_client.requests) == 1
    assert executor.calls == [
        (["python", "-c", "open('diagnostic.txt', 'w').write('x')"], True)
    ]


def test_agent_loop_requests_approval_for_diagnostic_command(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)
    executor = RecordingCommandExecutor()
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={
                            "argv": ["python", "-c", "open('diagnostic.txt', 'w').write('x')"]
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Diagnostic completed."),
        ]
    )

    result = AgentLoop(
        task="Run a diagnostic",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(tmp_path / "diagnostic" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
        command_executor=executor,
    ).run()

    assert result.status == "completed"
    assert len(approval_client.requests) == 1
    assert approval_client.requests[0].preview["policy_category"] == "unknown"
    assert executor.calls == [
        (["python", "-c", "open('diagnostic.txt', 'w').write('x')"], True)
    ]


def test_agent_loop_blocks_dangerous_command_before_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "context.txt").write_text("existing context\n", encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)
    model_client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_1",
                        name="read",
                        arguments={"source": "workspace", "target": "context.txt"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_2",
                        name="run_command",
                        arguments={"argv": ["bash", "-lc", "whoami"]},
                    )
                ]
            ),
            ModelResponse(final_text="Command was blocked."),
        ]
    )

    result = AgentLoop(
        task="Run a shell command",
        workspace=workspace,
        model_client=model_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        approval_client=approval_client,
        approval_store=ApprovalStore(tmp_path / "approvals"),
    ).run()

    assert result.status == "completed"
    assert approval_client.requests == []
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    tool_result = [
        event
        for event in events
        if event["type"] == "tool_result" and event.get("tool") == "run_command"
    ][0]
    assert tool_result["status"] == "command_rejected_by_policy"
    assert tool_result["hook"] == "command_policy_guard"
