from __future__ import annotations

from minicode_harness.context import RunState, SessionCompactionState
from minicode_harness.runtime.run_lifecycle import RunLifecycle, RunSnapshot
from minicode_harness.state import CheckpointStore, ReplSessionStore, TaskListState
from minicode_harness.trace import TraceWriter


def _snapshot(*, workspace: str, messages: list[dict[str, object]]) -> RunSnapshot:
    return RunSnapshot(
        step=1,
        messages=messages,
        compaction_state=SessionCompactionState(),
        run_state=RunState(),
        task_state=TaskListState(),
        observations=[],
        modified_files=[],
        workspace_digest_paths=[],
        user_turn_id="turn_1",
        model_call_count=1,
        tool_calls=1,
        memory_snapshot_hash=None,
        memory_snapshot_path=None,
    )


def test_running_checkpoint_does_not_commit_conversation_session(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "session-data")
    session = session_store.create(workspace)
    run_path = tmp_path / "run"
    checkpoint_store = CheckpointStore(run_path / "checkpoints")
    lifecycle = RunLifecycle(
        run_id="run_20260914_001",
        task="inspect",
        workspace=str(workspace),
        trace_writer=TraceWriter(run_path / "trace.jsonl"),
        checkpoint_store=checkpoint_store,
        session_memory=session,
        emit_hook=lambda *args, **kwargs: None,
        resource_closers=[],
    )
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "working"},
    ]
    snapshot = _snapshot(workspace=str(workspace), messages=messages)

    lifecycle.checkpoint_progress(snapshot, status="running", reason="tool:read")

    assert session_store.load(workspace, session.session_id).load_message_history() == []
    checkpoint = checkpoint_store.load_latest()
    assert checkpoint is not None
    assert checkpoint_store.load_history(checkpoint) == messages

    lifecycle.finish_stopped(
        snapshot,
        reason="cancelled_by_test",
        final_text=None,
        stop_summary=None,
        modified_before_stop=[],
    )

    assert session_store.load(workspace, session.session_id).load_message_history() == messages
