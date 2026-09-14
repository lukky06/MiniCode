from __future__ import annotations

import hashlib

from minicode_harness.loop import AgentLoop
from minicode_harness.resume import resume_run
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.state import (
    ApprovalDecision,
    CheckpointStore,
    ExecutionJournal,
    RunCheckpoint,
    RunStore,
    StaticApprovalClient,
    digest_workspace_files,
)
from minicode_harness.tools import CommandRunResult
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def call_request(self, request):
        self.calls.append(list(request.as_chat_messages()))
        return self.responses.pop(0)


class RecordingCommandExecutor:
    sandboxed = False

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
        return CommandRunResult(
            argv=list(argv),
            command=" ".join(argv),
            returncode=0,
            stdout="secret command output must stay out of the journal",
            stderr="",
            duration_seconds=0.01,
            timeout_seconds=timeout_seconds,
            allowlist_rule="test",
        )


def test_execution_journal_round_trips_durable_lifecycle(tmp_path, monkeypatch) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(
        "minicode_harness.state.execution_journal.os.fsync",
        lambda fd: fsync_calls.append(fd),
    )
    journal = ExecutionJournal(tmp_path / "execution-journal.jsonl")

    prepared = journal.append_prepared(
        entry_id="1:call_1",
        run_id="run_1",
        step=1,
        tool_call_id="call_1",
        tool_name="write",
        argument_fingerprint="abc",
        effect_kind="filesystem",
        target_paths=["app.py"],
        before_hashes={"app.py": None},
        expected_after_hashes={"app.py": "after"},
    )
    journal.append_completed(
        prepared,
        after_hashes={"app.py": "after"},
        result_status="completed",
    )

    events = journal.load_events()
    assert [event.event for event in events] == ["PREPARED", "COMPLETED"]
    assert events[0].entry_id == events[1].entry_id
    assert events[1].after_hashes == {"app.py": "after"}
    assert len(fsync_calls) == 2


def test_write_execution_records_prepared_and_completed_hashes(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run_write" / "trace.jsonl"
    content = "hello journal\n"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_write",
                        name="write",
                        arguments={"path": "README.md", "content": content},
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Create README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        no_skills=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    )

    result = loop.run()

    assert result.status == "completed"
    journal = ExecutionJournal(trace_path.parent / "execution-journal.jsonl")
    events = journal.load_events()
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert [event.event for event in events] == ["PREPARED", "COMPLETED"]
    assert events[0].target_paths == ["README.md"]
    assert events[0].before_hashes == {"README.md": None}
    assert events[0].expected_after_hashes == {"README.md": expected_hash}
    assert events[1].after_hashes == {"README.md": expected_hash}


def test_rejected_write_does_not_enter_execution_journal(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run_rejected" / "trace.jsonl"
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
    loop = AgentLoop(
        task="Create README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        no_skills=True,
        approval_client=StaticApprovalClient(ApprovalDecision.REJECT),
    )

    loop.run()

    journal = ExecutionJournal(trace_path.parent / "execution-journal.jsonl")
    assert journal.load_events() == []
    assert not journal.path.exists()


def test_read_only_tool_does_not_enter_execution_journal(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    trace_path = tmp_path / "run_read" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_read",
                        name="read",
                        arguments={"source": "workspace", "target": "README.md"},
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Read README",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    )

    loop.run()

    journal = ExecutionJournal(trace_path.parent / "execution-journal.jsonl")
    assert journal.load_events() == []
    assert not journal.path.exists()


def test_command_journal_keeps_only_bounded_execution_facts(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run_command" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="call_cmd",
                        name="run_command",
                        arguments={
                            "argv": ["python", "-m", "pytest", "-q", "tests/test_x.py"],
                            "timeout_seconds": 30,
                        },
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Run focused test",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        enable_write=True,
        no_skills=True,
        command_executor=RecordingCommandExecutor(),
    )

    result = loop.run()

    assert result.status == "completed"
    journal = ExecutionJournal(trace_path.parent / "execution-journal.jsonl")
    events = journal.load_events()
    assert [event.event for event in events] == ["PREPARED", "COMPLETED"]
    assert events[0].effect_kind == "command"
    assert events[0].command_category == "verification"
    assert events[1].result_status == "completed"
    assert events[1].returncode == 0
    assert "secret command output" not in journal.path.read_text(encoding="utf-8")


def _interrupted_run(tmp_path, *, run_id: str = "run_20260907_901"):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="resume interrupted side effect",
        workspace=workspace,
        run_id=run_id,
        no_write=False,
    )
    run_path = run_store.path_for(run_id)
    checkpoint = RunCheckpoint(
        run_id=run_id,
        step=1,
        task=session.task,
        workspace=session.workspace,
        status="running",
    )
    CheckpointStore(run_path / "checkpoints").save(
        checkpoint,
        message_history=[{"role": "user", "content": session.task}],
    )
    return workspace, run_store, session, run_path


def test_resume_reconciles_prepared_before_effect_without_replay(tmp_path) -> None:
    workspace, run_store, session, run_path = _interrupted_run(tmp_path)
    content = "planned but not applied\n"
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    journal = ExecutionJournal(run_path / "execution-journal.jsonl")
    journal.append_prepared(
        entry_id="2:call_write",
        run_id=session.run_id,
        step=2,
        tool_call_id="call_write",
        tool_name="write",
        argument_fingerprint="write-hash",
        effect_kind="filesystem",
        target_paths=["README.md"],
        before_hashes={"README.md": None},
        expected_after_hashes={"README.md": expected_hash},
    )
    client = ScriptedModelClient([ModelResponse(final_text="resumed")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
        force_rebuild_context=True,
    )

    assert result.status == "completed"
    assert not (workspace / "README.md").exists()
    assert "No side effect is present" in str(client.calls[0])
    assert journal.load_events()[-1].resolution == "effect_not_applied"


def test_resume_reconciles_effect_after_prepared_before_completed(tmp_path) -> None:
    workspace, run_store, session, run_path = _interrupted_run(tmp_path)
    content = "already applied once\n"
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    journal = ExecutionJournal(run_path / "execution-journal.jsonl")
    journal.append_prepared(
        entry_id="2:call_write",
        run_id=session.run_id,
        step=2,
        tool_call_id="call_write",
        tool_name="write",
        argument_fingerprint="write-hash",
        effect_kind="filesystem",
        target_paths=["README.md"],
        before_hashes={"README.md": None},
        expected_after_hashes={"README.md": expected_hash},
    )
    (workspace / "README.md").write_bytes(content.encode("utf-8"))
    client = ScriptedModelClient([ModelResponse(final_text="resumed")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
        force_rebuild_context=True,
    )

    assert result.status == "completed"
    assert (workspace / "README.md").read_bytes() == content.encode("utf-8")
    assert "already contains the effect" in str(client.calls[0])
    assert journal.load_events()[-1].resolution == "effect_applied"


def test_resume_reconciles_completed_before_checkpoint_without_replay(tmp_path) -> None:
    workspace, run_store, session, run_path = _interrupted_run(tmp_path)
    content = "completed before checkpoint\n"
    after_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    journal = ExecutionJournal(run_path / "execution-journal.jsonl")
    prepared = journal.append_prepared(
        entry_id="2:call_write",
        run_id=session.run_id,
        step=2,
        tool_call_id="call_write",
        tool_name="write",
        argument_fingerprint="write-hash",
        effect_kind="filesystem",
        target_paths=["README.md"],
        before_hashes={"README.md": None},
        expected_after_hashes={"README.md": after_hash},
    )
    (workspace / "README.md").write_text(content, encoding="utf-8")
    journal.append_completed(
        prepared,
        after_hashes=digest_workspace_files(workspace, ["README.md"]),
        result_status="completed",
    )
    client = ScriptedModelClient([ModelResponse(final_text="resumed")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
        force_rebuild_context=True,
    )

    assert result.status == "completed"
    assert (workspace / "README.md").read_text(encoding="utf-8") == content
    assert journal.load_events()[-1].resolution == "effect_applied"
    latest = CheckpointStore(run_path / "checkpoints").load_latest()
    assert latest is not None
    assert "README.md" in latest.modified_files


def test_resume_blocks_unresolved_side_effect_without_model_call(tmp_path) -> None:
    workspace, run_store, session, run_path = _interrupted_run(tmp_path)
    target = workspace / "README.md"
    target.write_text("before\n", encoding="utf-8")
    journal = ExecutionJournal(run_path / "execution-journal.jsonl")
    journal.append_prepared(
        entry_id="2:call_patch",
        run_id=session.run_id,
        step=2,
        tool_call_id="call_patch",
        tool_name="apply_patch",
        argument_fingerprint="patch-hash",
        effect_kind="filesystem",
        target_paths=["README.md"],
        before_hashes=digest_workspace_files(workspace, ["README.md"]),
        expected_after_hashes={},
    )
    target.write_text("ambiguous external or partial effect\n", encoding="utf-8")
    client = ScriptedModelClient([ModelResponse(final_text="must not run")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
        force_rebuild_context=True,
    )

    assert result.status == "blocked"
    assert result.reason == "execution_effect_unknown"
    assert result.conflicts == ["README.md"]
    assert client.calls == []


def test_resume_blocks_uncheckpointed_command_even_when_completed(tmp_path) -> None:
    _, run_store, session, run_path = _interrupted_run(tmp_path)
    journal = ExecutionJournal(run_path / "execution-journal.jsonl")
    prepared = journal.append_prepared(
        entry_id="2:call_cmd",
        run_id=session.run_id,
        step=2,
        tool_call_id="call_cmd",
        tool_name="run_command",
        argument_fingerprint="command-hash",
        effect_kind="command",
        target_paths=[],
        before_hashes={},
        expected_after_hashes={},
        command_category="verification",
    )
    journal.append_completed(
        prepared,
        after_hashes={},
        result_status="completed",
        returncode=0,
    )
    client = ScriptedModelClient([ModelResponse(final_text="must not run")])

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=client,
    )

    assert result.status == "blocked"
    assert result.reason == "execution_effect_unknown"
    assert result.conflicts == ["run_command:call_cmd"]
    assert client.calls == []
