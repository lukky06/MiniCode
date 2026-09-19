from __future__ import annotations

from types import SimpleNamespace

import pytest

import minicode_harness.resume as resume_module
from minicode_harness.loop import AgentRunResult
from minicode_harness.memory.snapshot import MemorySnapshotStore
from minicode_harness.memory.store import RepositoryMemoryStore
from minicode_harness.resume import resume_run
from minicode_harness.state import CheckpointStore, RunCheckpoint, RunStore


@pytest.mark.parametrize("force_rebuild_context", [False, True])
def test_resume_reuses_original_v3_memory_snapshot(
    tmp_path,
    monkeypatch,
    force_rebuild_context: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="resume me",
        workspace=workspace,
        run_id="run_20260918_901",
        repository_memory_enabled=True,
        no_write=True,
    )
    run_path = run_store.path_for(session.run_id)

    repository = RepositoryMemoryStore(workspace, data_dir=data_dir)
    repository.write_durable_memory(
        "# Memory\n\nold handbook\n",
        "v1\n- old summary\n",
    )
    source = repository.capture_snapshot_source()
    snapshot_store = MemorySnapshotStore(run_path)
    snapshot = snapshot_store.save(
        repository_id=repository.repository_id,
        memory_summary=source.memory_summary,
        memory_md=source.memory_md,
        rollout_summary_files=source.rollout_summary_files,
    )
    CheckpointStore(run_path / "checkpoints").save(
        RunCheckpoint(
            run_id=session.run_id,
            step=0,
            task=session.task,
            workspace=session.workspace,
            status="stopped",
            memory_snapshot_hash=snapshot.index_hash,
            memory_snapshot_path=snapshot_store.checkpoint_path,
        ),
        message_history=[],
    )
    run_store.update_session_state(session.run_id, status="stopped", current_step=0)

    repository.write_durable_memory(
        "# Memory\n\nNEW handbook\n",
        "v1\n- NEW summary\n",
    )

    captured: dict[str, object] = {}

    class FakeLoop:
        observations = []
        modified_files = []
        run_state = SimpleNamespace(verification=SimpleNamespace(status="not_run"))

        def run(self) -> AgentRunResult:
            return AgentRunResult(
                status="completed",
                final_text="done",
                steps=1,
                tool_calls=0,
                stop_reason="final_text",
            )

    def fake_build(session, **kwargs):
        captured.update(kwargs)
        return FakeLoop()

    monkeypatch.setattr(
        resume_module,
        "build_agent_loop_from_session",
        fake_build,
    )

    result = resume_run(
        session.run_id,
        run_store=run_store,
        model_client=SimpleNamespace(model="fake-model"),
        data_dir=str(data_dir),
        force_rebuild_context=force_rebuild_context,
    )

    assert result.status == "completed"
    assert captured["long_term_context"] == "v1\n- old summary\n"
    assert captured["memory_snapshot_hash"] == snapshot.index_hash
    assert snapshot_store.read_summary() == "v1\n- old summary\n"
    assert "NEW" not in snapshot_store.read_memory()
