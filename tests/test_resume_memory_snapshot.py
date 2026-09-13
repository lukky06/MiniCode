from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minicode_harness.loop import AgentRunResult
from minicode_harness.memory import MemorySnapshotStore, RepositoryMemoryStore
from minicode_harness.models import ModelClient, ModelResponse
from minicode_harness.output import NullOutputSink
from minicode_harness.resume import resume_run
from minicode_harness.runtime.run_executor import RunExecutionRequest, RunExecutor
from minicode_harness.state import CheckpointStore, RunCheckpoint, RunStore, StaticApprovalClient


class RecordingModelClient(ModelClient):
    def __init__(self, final_text: str = "resumed") -> None:
        self.final_text = final_text
        self.requests: list[list[dict]] = []

    def call_request(self, request):
        self.requests.append(list(request.as_chat_messages()))
        return ModelResponse(final_text=self.final_text)


def test_memory_snapshot_store_detects_tampering(tmp_path: Path) -> None:
    store = MemorySnapshotStore(tmp_path / "run")
    snapshot = store.save(
        repository_id="repo_test",
        rendered_index="Repository Memory Index:\n- instructions\n",
    )

    loaded = store.load(expected_hash=snapshot.index_hash)
    assert loaded is not None
    assert loaded.rendered_index == snapshot.rendered_index

    store.markdown_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown"):
        store.load(expected_hash=snapshot.index_hash)


def test_memory_snapshot_rejects_undeclared_topic_files(tmp_path: Path) -> None:
    store = MemorySnapshotStore(tmp_path / "run")
    store.save(
        repository_id="repo_test",
        rendered_index="index-v1",
        topic_payloads={
            "decisions": {
                "topic": "decisions",
                "content": "decision-v1",
                "content_hash": "hash-v1",
                "updated_at": "t1",
            }
        },
    )
    (store.topic_dir / "environment.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="do not match snapshot metadata"):
        store.load()


def test_memory_topic_snapshot_is_captured_with_the_run_index(
    tmp_path: Path,
) -> None:
    store = MemorySnapshotStore(tmp_path / "run")
    first_payload = {
        "topic": "decisions",
        "content": "decision-v1",
        "content_hash": "hash-v1",
        "updated_at": "t1",
    }
    snapshot = store.save(
        repository_id="repo_test",
        rendered_index="index-v1",
        topic_payloads={"decisions": first_payload},
    )

    assert snapshot.version == 2
    assert set(snapshot.topic_hashes) == {"decisions"}
    assert store.read_topic("decisions") == first_payload

    second_payload = {
        "topic": "decisions",
        "content": "decision-v2",
        "content_hash": "hash-v2",
        "updated_at": "t2",
    }
    rebuilt = store.save(
        repository_id="repo_test",
        rendered_index="index-v2",
        topic_payloads={"decisions": second_payload},
    )

    assert rebuilt.topic_hashes != snapshot.topic_hashes
    assert store.read_topic("decisions") == second_payload


def test_repository_snapshot_capture_retries_when_topic_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    repository.topic_store.add_entry(
        topic="decisions",
        entry_type="decision",
        summary="Use the original decision.",
        evidence_ids=["run_1"],
    )
    repository.refresh_index()
    first_payload = repository.read_topic("decisions")
    second_payload = {
        **first_payload,
        "content": first_payload["content"].replace("original", "updated"),
        "content_hash": "updated-hash",
        "updated_at": "updated",
    }
    calls = 0

    def changing_reader(topic):
        nonlocal calls
        calls += 1
        return first_payload if calls == 1 else second_payload

    monkeypatch.setattr(repository, "read_topic", changing_reader)

    source = repository.capture_snapshot_source()

    assert calls == 4
    assert source.topic_payloads["decisions"] == second_payload


def test_run_executor_creates_memory_snapshot_before_agent_loop(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))
    run_store = RunStore(tmp_path / "runs")
    repository = RepositoryMemoryStore(workspace, data_dir=data_dir)
    repository.topic_store.add_entry(
        topic="instructions",
        entry_type="user_instruction",
        summary="Use focused tests.",
        evidence_ids=["test"],
    )
    repository.refresh_index()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.create_model_client",
        lambda **kwargs: SimpleNamespace(model="fake-model"),
    )

    class FakeOrchestrator:
        def __init__(self, **kwargs) -> None:
            pass

        def recall_snapshot(self, **kwargs):
            return SimpleNamespace(
                rendered_index="Repository Memory Index:\n- instructions\n",
                topic_payloads=repository.snapshot_topics(),
            )

        def finalize_completed_run(self, **kwargs) -> None:
            pass

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            calls.update(kwargs)
            self.observations = []
            self.modified_files = []
            self.run_state = SimpleNamespace(
                inspected_files=[],
                verification=SimpleNamespace(status="not_run"),
            )

        def run(self) -> AgentRunResult:
            return AgentRunResult(
                status="completed",
                final_text="done",
                steps=1,
                tool_calls=0,
                stop_reason="final_text",
            )

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.RequestOrchestrator",
        FakeOrchestrator,
    )
    monkeypatch.setattr("minicode_harness.runtime.run_executor.AgentLoop", FakeLoop)

    result = RunExecutor(run_store=run_store).execute(
        RunExecutionRequest(task="task", workspace=workspace),
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
    )

    snapshot_store = MemorySnapshotStore(result.run_path)
    snapshot = snapshot_store.load()
    assert snapshot is not None
    assert snapshot.rendered_index.startswith("Repository Memory Index")
    assert set(snapshot.topic_hashes) == {"instructions"}
    assert "Use focused tests." in snapshot_store.read_topic("instructions")["content"]
    assert calls["memory_snapshot_hash"] == snapshot.index_hash
    assert calls["memory_snapshot_path"] == "memory-snapshot.json"


def _prepare_resumable_run(
    tmp_path: Path,
) -> tuple[RunStore, str, Path, RepositoryMemoryStore, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session = run_store.create_run(
        task="inspect repository memory",
        workspace=workspace,
        run_id="run_20260722_901",
        no_write=True,
    )
    data_dir = tmp_path / "data"
    repository = RepositoryMemoryStore(workspace, data_dir=data_dir)
    repository.topic_store.add_entry(
        topic="instructions",
        entry_type="user_instruction",
        summary="Use focused verification.",
        evidence_ids=["run_1:user"],
    )
    original_index = repository.refresh_index()
    run_path = run_store.path_for(session.run_id)
    snapshot_store = MemorySnapshotStore(run_path)
    snapshot = snapshot_store.save(
        repository_id=repository.repository_id,
        rendered_index=original_index,
        topic_payloads=repository.snapshot_topics(),
    )
    CheckpointStore(run_path / "checkpoints").save(
        RunCheckpoint(
            run_id=session.run_id,
            step=0,
            task=session.task,
            workspace=session.workspace,
            status="running",
            memory_snapshot_hash=snapshot.index_hash,
            memory_snapshot_path=snapshot_store.checkpoint_path,
        )
    )
    repository.topic_store.add_entry(
        topic="environment",
        entry_type="environment",
        summary="Python 3.12 is available.",
        evidence_ids=["run_2:command"],
    )
    repository.refresh_index()
    return run_store, session.run_id, data_dir, repository, original_index


def test_resume_uses_original_snapshot_until_force_rebuild(tmp_path: Path) -> None:
    run_store, run_id, data_dir, repository, original_index = _prepare_resumable_run(tmp_path)
    normal_model = RecordingModelClient("normal resume")

    normal = resume_run(
        run_id,
        run_store=run_store,
        model_client=normal_model,
        data_dir=str(data_dir),
    )

    assert normal.status == "completed"
    normal_system = normal_model.requests[0][0]["content"]
    assert "instructions" in normal_system
    assert "environment" not in normal_system
    assert MemorySnapshotStore(run_store.path_for(run_id)).load().rendered_index == original_index

    forced_model = RecordingModelClient("forced resume")
    forced = resume_run(
        run_id,
        run_store=run_store,
        force_rebuild_context=True,
        model_client=forced_model,
        data_dir=str(data_dir),
    )

    assert forced.status == "completed"
    forced_system = forced_model.requests[0][0]["content"]
    assert "environment" in forced_system
    rebuilt = MemorySnapshotStore(run_store.path_for(run_id)).load()
    assert rebuilt is not None
    assert rebuilt.rendered_index == repository.render_index()
    assert set(rebuilt.topic_hashes) == {"instructions", "environment"}
    assert "Python 3.12" in MemorySnapshotStore(
        run_store.path_for(run_id)
    ).read_topic("environment")["content"]
    latest_checkpoint = CheckpointStore(
        run_store.path_for(run_id) / "checkpoints"
    ).load_latest()
    assert latest_checkpoint is not None
    assert latest_checkpoint.memory_snapshot_hash == rebuilt.index_hash
    payload = json.loads(
        (
            run_store.path_for(run_id)
            / "checkpoints"
            / "latest.json"
        ).read_text(encoding="utf-8")
    )
    assert "rendered_index" not in payload
