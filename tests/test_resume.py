import json
import sys

import pytest

from minicode_harness.context import RunState, VerificationState
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
import minicode_harness.resume as resume_module
from minicode_harness.resume import latest_recoverable_run_id, resume_run
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.policy import check_command_allowed
from minicode_harness.tools import CommandRunResult
from minicode_harness.state import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStore,
    CheckpointStore,
    RunCheckpoint,
    ReplSessionStore,
    RunStore,
    StaticApprovalClient,
    TaskListState,
    TaskRecord,
    digest_workspace_files,
)


def test_latest_recoverable_run_requires_checkpoint_and_skips_completed(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    recoverable = run_store.create_run(
        task="recoverable",
        workspace=workspace,
        run_id="run_20260827_001",
    )
    CheckpointStore(
        run_store.path_for(recoverable.run_id) / "checkpoints"
    ).save(
        RunCheckpoint(
            run_id=recoverable.run_id,
            step=1,
            task=recoverable.task,
            workspace=recoverable.workspace,
            status="stopped",
        )
    )
    run_store.create_run(
        task="newer without checkpoint",
        workspace=workspace,
        run_id="run_20260827_002",
    )
    completed = run_store.create_run(
        task="completed",
        workspace=workspace,
        run_id="run_20260827_003",
    )
    CheckpointStore(
        run_store.path_for(completed.run_id) / "checkpoints"
    ).save(
        RunCheckpoint(
            run_id=completed.run_id,
            step=2,
            task=completed.task,
            workspace=completed.workspace,
            status="running",
        )
    )
    run_store.update_session_state(
        completed.run_id,
        status="completed",
        current_step=2,
    )

    assert latest_recoverable_run_id(
        run_store=run_store,
        workspace=workspace,
    ) == recoverable.run_id


def test_latest_recoverable_run_fails_on_invalid_current_checkpoint(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    older = run_store.create_run(
        task="older",
        workspace=workspace,
        run_id="run_20260918_001",
    )
    CheckpointStore(run_store.path_for(older.run_id) / "checkpoints").save(
        RunCheckpoint(
            run_id=older.run_id,
            step=1,
            task=older.task,
            workspace=older.workspace,
            status="stopped",
        )
    )
    newer = run_store.create_run(
        task="newer",
        workspace=workspace,
        run_id="run_20260918_002",
    )
    checkpoint_store = CheckpointStore(
        run_store.path_for(newer.run_id) / "checkpoints"
    )
    checkpoint_store.save(
        RunCheckpoint(
            run_id=newer.run_id,
            step=1,
            task=newer.task,
            workspace=newer.workspace,
            status="running",
        )
    )
    checkpoint_store.latest_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError):
        latest_recoverable_run_id(run_store=run_store, workspace=workspace)


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[list[dict]] = []

    def call_request(self, request):
        self.requests.append(list(request.as_chat_messages()))
        return self.responses.pop(0)


class RecordingSandboxCommandExecutor:
    sandboxed = True
    command_rules = ()

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def classify(self, argv):
        return check_command_allowed(argv, sandboxed=True, rules=self.command_rules)

    def execute(
        self,
        workspace,
        argv,
        timeout_seconds,
        cancellation_token=None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        del workspace, cancellation_token, approval_granted
        normalized = list(argv)
        self.calls.append(normalized)
        return CommandRunResult(
            argv=normalized,
            command=" ".join(normalized),
            returncode=0,
            stdout="sandboxed\n",
            stderr="",
            duration_seconds=0.01,
            timeout_seconds=timeout_seconds,
            allowlist_rule="sandbox default",
        )


def test_resume_blocks_when_modified_file_digest_changed(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    readme = workspace / "README.md"
    readme.write_text("old\n", encoding="utf-8")
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="task",
        workspace=workspace,
        run_id="run_20260701_001",
        no_write=True,
    )
    run_path = run_store.path_for(session.run_id)
    CheckpointStore(run_path / "checkpoints").save(
        RunCheckpoint(
            run_id=session.run_id,
            step=1,
            task=session.task,
            workspace=session.workspace,
            modified_files=["README.md"],
            workspace_digest=digest_workspace_files(workspace, ["README.md"]),
            tool_calls=1,
            status="running",
        )
    )
    readme.write_text("changed by user\n", encoding="utf-8")

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="unused")]),
    )

    assert result.status == "blocked"
    assert result.reason == "workspace_conflict"
    assert result.conflicts == ["README.md"]


def test_resume_force_rebuild_context_ignores_digest_conflict(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    readme = workspace / "README.md"
    readme.write_text("old\n", encoding="utf-8")
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="task",
        workspace=workspace,
        run_id="run_20260701_001",
        no_write=True,
    )
    run_path = run_store.path_for(session.run_id)
    CheckpointStore(run_path / "checkpoints").save(
        RunCheckpoint(
            run_id=session.run_id,
            step=1,
            task=session.task,
            workspace=session.workspace,
            modified_files=["README.md"],
            workspace_digest=digest_workspace_files(workspace, ["README.md"]),
            tool_calls=1,
            status="running",
        )
    )
    readme.write_text("changed by user\n", encoding="utf-8")

    result = resume_run(
        session.run_id,
        run_store=run_store,
        force_rebuild_context=True,
        model_client=ScriptedModelClient([ModelResponse(final_text="resumed")]),
    )

    assert result.status == "completed"
    assert result.final_text == "resumed"


def test_resume_restores_persisted_command_sandbox(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="continue safely",
        workspace=workspace,
        run_id="run_20260701_099",
        no_write=True,
        sandbox_mode="docker",
        sandbox_image="python:3.11-slim",
        repository_memory_enabled=False,
    )
    CheckpointStore(run_store.path_for(session.run_id) / "checkpoints").save(
        RunCheckpoint(
            run_id=session.run_id,
            step=1,
            task=session.task,
            workspace=session.workspace,
            status="running",
        )
    )
    captured = {}

    class FakeExecutor:
        sandboxed = True

        def execute(self, *args, **kwargs):
            raise AssertionError("No command should execute in this test")

    def fake_create(mode, *, image=None, command_rules=(), workspace_writable=True):
        captured["mode"] = str(mode)
        captured["image"] = image
        captured["command_rules"] = list(command_rules)
        captured["workspace_writable"] = workspace_writable
        return FakeExecutor()

    monkeypatch.setattr(resume_module, "create_command_executor", fake_create)
    monkeypatch.setattr(
        resume_module,
        "RepositoryMemoryStore",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Repository Memory must stay disabled during Resume")
        ),
    )

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="resumed")]),
    )

    assert result.status == "completed"
    assert captured == {
        "mode": "docker",
        "image": "python:3.11-slim",
        "command_rules": [],
        "workspace_writable": False,
    }


def test_resume_restores_task_state_for_task_list(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="continue repair",
        workspace=workspace,
        run_id="run_20260701_002",
        no_write=True,
        repository_memory_enabled=False,
    )
    run_path = run_store.path_for(session.run_id)
    checkpoint_store = CheckpointStore(run_path / "checkpoints")
    checkpoint_store.save(
        RunCheckpoint(
            run_id=session.run_id,
            step=1,
            task=session.task,
            workspace=session.workspace,
            task_state=TaskListState(
                next_id=3,
                tasks=[
                    TaskRecord(id="1", subject="定位问题", status="completed"),
                    TaskRecord(id="2", subject="实现修复", status="in_progress"),
                ],
            ),
            status="running",
        ),
        message_history=[{"role": "user", "content": session.task}],
    )
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="list_tasks",
                        name="task",
                        arguments={"action": "list"},
                    )
                ]
            ),
            ModelResponse(final_text="resumed"),
        ]
    )

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    payload = json.loads(client.requests[1][-1]["content"])
    assert payload["tasks"] == [
        ["1", "completed", "定位问题"],
        ["2", "in_progress", "实现修复"],
    ]


def test_resume_restores_pending_approval_and_continues(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="write readme",
        workspace=workspace,
        run_id="run_20260701_001",
        no_write=False,
        repository_memory_enabled=False,
    )
    run_path = run_store.path_for(session.run_id)
    approval_store = ApprovalStore(run_path / "approvals")
    approval_store.save_pending(
        ApprovalRequest(
            id="step_0001_call_1",
            tool_call_id="call_1",
            tool_name="write",
            risk_level="medium",
            step=1,
            arguments={"path": "README.md", "content": "hello\n"},
            preview={"summary": {"tool": "write_file", "path": "README.md"}},
        )
    )
    approval_client = StaticApprovalClient(ApprovalDecision.APPROVE)

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="resumed")]),
        approval_client=approval_client,
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert approval_store.load_pending() is None
    latest = CheckpointStore(run_path / "checkpoints").load_latest()
    assert latest is not None
    assert latest.modified_files == ["README.md"]
    assert latest.run_state.verification.status == "not_run"
    journal = resume_module.ExecutionJournal(run_path / "execution-journal.jsonl")
    assert [event.event for event in journal.load_events()] == ["PREPARED", "COMPLETED"]
    payload = json.loads(
        (run_path / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    assert "working_context" not in payload
    events = [json.loads(line) for line in (run_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "approval_restored" in [event["type"] for event in events]
    assert "run_state_updated" in [event["type"] for event in events]
    assert "task_memory_updated" not in [event["type"] for event in events]


def test_resume_pending_command_uses_original_sandbox_executor_and_journal(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="run pending sandbox command",
        workspace=workspace,
        run_id="run_20260916_001",
        no_write=False,
        repository_memory_enabled=False,
        sandbox_mode="docker",
        sandbox_image="minicode-test-image",
    )
    run_path = run_store.path_for(session.run_id)
    ApprovalStore(run_path / "approvals").save_pending(
        ApprovalRequest(
            id="step_0001_call_1",
            tool_call_id="call_1",
            tool_name="run_command",
            risk_level="medium",
            step=1,
            arguments={
                "argv": ["definitely-not-real-minicode-command"],
                "timeout_seconds": 30,
            },
            preview={"summary": "Run sandboxed command"},
        )
    )
    executor = RecordingSandboxCommandExecutor()
    monkeypatch.setattr(
        resume_module,
        "create_command_executor",
        lambda *args, **kwargs: executor,
    )

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="resumed")]),
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    assert executor.calls == [["definitely-not-real-minicode-command"]]
    journal = resume_module.ExecutionJournal(run_path / "execution-journal.jsonl")
    assert [event.event for event in journal.load_events()] == ["PREPARED", "COMPLETED"]


def test_resume_can_persist_session_command_grant_for_restored_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "sessions")
    conversation = session_store.create(workspace)
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="run approved command",
        workspace=workspace,
        run_id="run_20260907_001",
        conversation_session_id=conversation.session_id,
        no_write=False,
        repository_memory_enabled=False,
    )
    run_path = run_store.path_for(session.run_id)
    ApprovalStore(run_path / "approvals").save_pending(
        ApprovalRequest(
            id="step_0001_call_1",
            tool_call_id="call_1",
            tool_name="run_command",
            risk_level="high",
            step=1,
            arguments={
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('approved.txt').write_text('ok')",
                ],
                "timeout_seconds": 30,
            },
            preview={"summary": "Run approved command"},
            can_approve_session=True,
        )
    )

    result = resume_run(
        session.run_id,
        run_store=run_store,
        session_store=session_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="resumed")]),
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE_SESSION),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    assert (workspace / "approved.txt").read_text(encoding="utf-8") == "ok"
    persisted = session_store.load(workspace, conversation.session_id)
    assert len(persisted.command_approval_grants) == 1


def test_resume_rejects_pending_approval_and_returns_to_model(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="write readme",
        workspace=workspace,
        run_id="run_20260701_001",
        no_write=False,
        repository_memory_enabled=False,
    )
    run_path = run_store.path_for(session.run_id)
    ApprovalStore(run_path / "approvals").save_pending(
        ApprovalRequest(
            id="step_0001_call_1",
            tool_call_id="call_1",
            tool_name="write",
            risk_level="medium",
            step=1,
            arguments={"path": "README.md", "content": "hello\n"},
            preview={"summary": {"tool": "write_file", "path": "README.md"}},
        )
    )

    result = resume_run(
        session.run_id,
        run_store=run_store,
        approval_client=StaticApprovalClient(ApprovalDecision.REJECT),
        model_client=ScriptedModelClient(
            [ModelResponse(final_text="The write was rejected; no file was changed.")]
        ),
    )

    assert result.status == "completed"
    assert result.reason == "final_text"
    assert not (workspace / "README.md").exists()


def _prepare_pending_approval_resume(tmp_path, decision: ApprovalDecision):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "sessions")
    conversation = session_store.create(workspace)
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="write readme",
        workspace=workspace,
        run_id=f"run_20260901_{'001' if decision == ApprovalDecision.APPROVE else '002'}",
        conversation_session_id=conversation.session_id,
        no_write=False,
        repository_memory_enabled=False,
    )
    run_path = run_store.path_for(session.run_id)
    ApprovalStore(run_path / "approvals").save_pending(
        ApprovalRequest(
            id="step_0001_call_1",
            tool_call_id="call_1",
            tool_name="write",
            risk_level="medium",
            step=1,
            arguments={"path": "README.md", "content": "hello\n"},
            preview={"summary": {"tool": "write_file", "path": "README.md"}},
        )
    )
    queue = SteeringQueue()
    queue.enqueue("first resumed correction")
    queue.enqueue("second pending correction")
    model = ScriptedModelClient([ModelResponse(final_text="resumed")])
    result = resume_run(
        session.run_id,
        run_store=run_store,
        session_store=session_store,
        model_client=model,
        approval_client=StaticApprovalClient(decision),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        steering_queue=queue,
    )
    return result, model, queue, run_store, session, session_store


def test_resume_approve_consumes_one_steering_after_restored_tool_batch(
    tmp_path,
) -> None:
    result, model, queue, run_store, session, session_store = (
        _prepare_pending_approval_resume(tmp_path, ApprovalDecision.APPROVE)
    )

    assert result.status == "completed"
    assert model.requests[0][-1] == {
        "role": "user",
        "content": "first resumed correction",
    }
    assert len(queue) == 1
    checkpoint_store = CheckpointStore(
        run_store.path_for(result.run_id) / "checkpoints"
    )
    latest = checkpoint_store.load_latest()
    assert latest is not None
    assert {
        "role": "user",
        "content": "first resumed correction",
    } in checkpoint_store.load_history(latest)
    persisted = session_store.load(
        session.workspace,
        session.conversation_session_id,
    )
    assert {
        "role": "user",
        "content": "first resumed correction",
    } in persisted.message_history


def test_resume_reject_consumes_one_steering_after_restored_tool_batch(
    tmp_path,
) -> None:
    result, model, queue, run_store, session, session_store = (
        _prepare_pending_approval_resume(tmp_path, ApprovalDecision.REJECT)
    )

    assert result.status == "completed"
    assert model.requests[0][-1] == {
        "role": "user",
        "content": "first resumed correction",
    }
    assert len(queue) == 1
    checkpoint_store = CheckpointStore(
        run_store.path_for(result.run_id) / "checkpoints"
    )
    latest = checkpoint_store.load_latest()
    assert latest is not None
    assert {
        "role": "user",
        "content": "first resumed correction",
    } in checkpoint_store.load_history(latest)
    persisted = session_store.load(
        session.workspace,
        session.conversation_session_id,
    )
    assert {
        "role": "user",
        "content": "first resumed correction",
    } in persisted.message_history


def test_resume_syncs_canonical_history_to_original_conversation_session(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "session-data")
    conversation = session_store.create(workspace)
    run_store = RunStore(tmp_path / "runs")
    run = run_store.create_run(
        task="continue session run",
        workspace=workspace,
        run_id="run_20260701_009",
        no_write=True,
        conversation_session_id=conversation.session_id,
        repository_memory_enabled=False,
    )
    checkpoint_store = CheckpointStore(run_store.path_for(run.run_id) / "checkpoints")
    full_history = [
        {"role": "user", "content": run.task},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "source"},
    ]
    checkpoint_store.save(
        RunCheckpoint(
            run_id=run.run_id,
            step=1,
            task=run.task,
            workspace=run.workspace,
            status="running",
        ),
        message_history=full_history,
    )

    result = resume_run(
        run.run_id,
        run_store=run_store,
        session_store=session_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="resumed answer")]),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    reloaded = session_store.load(workspace, conversation.session_id)
    assert result.status == "completed"
    assert reloaded.load_message_history()[:-1] == full_history
    assert reloaded.message_history[-1]["role"] == "assistant"
    assert reloaded.message_history[-1]["content"] == "resumed answer"


def test_resume_does_not_require_conversation_session_file(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "session-data")
    run_store = RunStore(tmp_path / "runs")
    run = run_store.create_run(
        task="resume without session file",
        workspace=workspace,
        run_id="run_20260701_010",
        no_write=True,
        conversation_session_id="session_abcdef123456",
        repository_memory_enabled=False,
    )

    result = resume_run(
        run.run_id,
        run_store=run_store,
        session_store=session_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="resumed")]),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    trace = (run_store.path_for(run.run_id) / "trace.jsonl").read_text(encoding="utf-8")
    assert "session_sync_skipped" in trace


def test_resume_after_read_fault_keeps_unconsumed_tool_result_visible(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.py").write_text("VALUE = 42\n", encoding="utf-8")
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="Inspect source.py and continue",
        workspace=workspace,
        run_id="run_20260715_101",
        no_write=True,
        no_skills=True,
        repository_memory_enabled=False,
    )
    checkpoint_store = CheckpointStore(
        run_store.path_for(session.run_id) / "checkpoints"
    )
    history = [
        {"role": "user", "content": session.task},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "read_before_fault",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"source":"workspace","target":"source.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read_before_fault",
            "content": "critical evidence: VALUE = 42",
        },
    ]
    checkpoint_store.save(
        RunCheckpoint(
            run_id=session.run_id,
            step=1,
            task=session.task,
            workspace=session.workspace,
            tool_calls=1,
            status="running",
        ),
        message_history=history,
    )
    client = ScriptedModelClient([ModelResponse(final_text="resumed from evidence")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    first_request = json.dumps(client.requests[0], ensure_ascii=False)
    assert "read_before_fault" in first_request
    assert "critical evidence: VALUE = 42" in first_request


def test_resume_after_write_fault_preserves_modified_file_state(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.py"
    source.write_text("VALUE = 43\n", encoding="utf-8")
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="Finish the source.py repair",
        workspace=workspace,
        run_id="run_20260715_102",
        no_skills=True,
        repository_memory_enabled=False,
    )
    checkpoint_store = CheckpointStore(
        run_store.path_for(session.run_id) / "checkpoints"
    )
    history = [
        {"role": "user", "content": session.task},
        {
            "role": "assistant",
            "content": (
                "[MiniCode compacted execution]\n"
                "- write | source.py\n"
                "  status: passed"
            ),
        },
    ]
    checkpoint_store.save(
        RunCheckpoint(
            run_id=session.run_id,
            step=2,
            task=session.task,
            workspace=session.workspace,
            modified_files=["source.py"],
            workspace_digest=digest_workspace_files(workspace, ["source.py"]),
            tool_calls=1,
            status="running",
        ),
        message_history=history,
    )

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=ScriptedModelClient([ModelResponse(final_text="repair complete")]),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    assert source.read_text(encoding="utf-8") == "VALUE = 43\n"
    latest = checkpoint_store.load_latest()
    assert latest is not None
    assert latest.modified_files == ["source.py"]


def test_resume_after_verification_fault_preserves_command_and_result(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="Repair the failed focused test",
        workspace=workspace,
        run_id="run_20260715_103",
        no_write=True,
        no_skills=True,
        repository_memory_enabled=False,
    )
    checkpoint_store = CheckpointStore(
        run_store.path_for(session.run_id) / "checkpoints"
    )
    command = "python -m pytest -q tests/test_source.py"
    history = [
        {
            "role": "assistant",
            "content": (
                "[MiniCode compacted execution]\n"
                f"- run_command | {command}\n"
                "  status: failed\n"
                "  returncode: 1\n"
                "  result: assertion VALUE == 43 failed"
            ),
        },
        {"role": "user", "content": session.task},
    ]
    checkpoint_store.save(
        RunCheckpoint(
            run_id=session.run_id,
            step=3,
            task=session.task,
            workspace=session.workspace,
            run_state=RunState(
                verification=VerificationState(
                    status="failed",
                    command=command,
                    returncode=1,
                )
            ),
            tool_calls=1,
            status="running",
        ),
        message_history=history,
    )
    client = ScriptedModelClient([ModelResponse(final_text="failure evidence retained")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
    )

    assert result.status == "completed"
    first_request = json.dumps(client.requests[0], ensure_ascii=False)
    assert command in first_request
    assert "returncode: 1" in first_request
    latest = checkpoint_store.load_latest()
    assert latest is not None
    assert latest.run_state.verification.status == "failed"
    assert latest.run_state.verification.command == command
    assert latest.run_state.verification.returncode == 1
