from datetime import datetime
import json

import pytest

from minicode_harness.state import ReplSessionStore
from minicode_harness.state import RunStore, default_run_root, generate_run_id


def test_generate_run_id_uses_next_sequence_for_current_date(tmp_path) -> None:
    (tmp_path / "run_20260630_001").mkdir()
    (tmp_path / "run_20260630_003").mkdir()
    (tmp_path / "run_20260629_999").mkdir()
    (tmp_path / "notes").mkdir()

    run_id = generate_run_id(tmp_path, now=datetime(2026, 6, 30, 12, 0, 0))

    assert run_id == "run_20260630_004"


def test_run_store_can_create_only_the_top_level_run_directory(tmp_path) -> None:
    run_directory = RunStore(root=tmp_path).create_run_directory("run_20260630_001")

    assert run_directory.run_id == "run_20260630_001"
    assert run_directory.path == tmp_path / "run_20260630_001"
    assert run_directory.path.is_dir()
    assert list(run_directory.path.iterdir()) == []


def test_default_run_root_uses_user_data_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINICODE_HOME", str(tmp_path / "minicode-home"))
    monkeypatch.delenv("MINICODE_RUNS_DIR", raising=False)

    assert default_run_root() == tmp_path / "minicode-home" / "runs"
    assert RunStore().root == tmp_path / "minicode-home" / "runs"


def test_default_run_root_can_be_overridden(tmp_path, monkeypatch) -> None:
    configured = tmp_path / "external-runs"
    monkeypatch.setenv("MINICODE_RUNS_DIR", str(configured))

    assert default_run_root() == configured
    assert RunStore().root == configured


def test_run_store_creates_phase_one_session_layout(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session = RunStore(root=tmp_path / "runs").create_run(
        task="noop",
        workspace=workspace,
        run_id="run_20260630_001",
        conversation_session_id="session_abcdef123456",
    )

    run_path = tmp_path / "runs" / "run_20260630_001"
    assert session.run_id == "run_20260630_001"
    assert (run_path / "run.json").is_file()
    assert (run_path / "trace.jsonl").is_file()
    assert (run_path / "trace.jsonl").read_text(encoding="utf-8") == ""
    assert (run_path / "artifacts").is_dir()
    assert (run_path / "checkpoints").is_dir()
    assert (run_path / "approvals").is_dir()
    assert (run_path / "debug").is_dir()

    session_json = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    assert session_json["run_id"] == "run_20260630_001"
    assert session_json["task"] == "noop"
    assert session_json["workspace"] == str(workspace.resolve())
    assert session_json["conversation_session_id"] == "session_abcdef123456"
    assert session_json["status"] == "created"
    assert session_json["current_step"] == 0
    assert session_json["max_steps"] == 50


def test_default_run_store_places_runs_under_conversation_session(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    minicode_home = tmp_path / "minicode-home"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))
    monkeypatch.delenv("MINICODE_RUNS_DIR", raising=False)
    session_store = ReplSessionStore()
    conversation = session_store.create(workspace)
    store = RunStore()

    run = store.create_run(
        task="noop",
        workspace=workspace,
        run_id="run_20260727_001",
        conversation_session_id=conversation.session_id,
    )

    session_dir = session_store.session_directory(conversation)
    run_path = session_dir / "runs" / run.run_id
    assert store.path_for(run.run_id) == run_path
    assert (session_dir / "session.json").is_file()
    assert (run_path / "run.json").is_file()
    assert not (minicode_home / "runs" / run.run_id).exists()


def test_default_run_store_generates_unique_ids_across_sessions(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    minicode_home = tmp_path / "minicode-home"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))
    monkeypatch.delenv("MINICODE_RUNS_DIR", raising=False)
    session_store = ReplSessionStore()
    first_session = session_store.create(workspace)
    second_session = session_store.create(workspace)
    store = RunStore()

    first = store.create_run(
        task="first",
        workspace=workspace,
        conversation_session_id=first_session.session_id,
    )
    second = store.create_run(
        task="second",
        workspace=workspace,
        conversation_session_id=second_session.session_id,
    )

    assert first.run_id != second.run_id
    assert store.list_run_ids() == [first.run_id, second.run_id]
    assert store.path_for(first.run_id).parent.parent.name == first_session.session_id
    assert store.path_for(second.run_id).parent.parent.name == second_session.session_id


def test_default_run_store_filters_by_workspace_and_session(tmp_path, monkeypatch) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    minicode_home = tmp_path / "minicode-home"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))
    monkeypatch.delenv("MINICODE_RUNS_DIR", raising=False)
    session_store = ReplSessionStore()
    first_session = session_store.create(first_workspace)
    second_session = session_store.create(second_workspace)
    store = RunStore()

    first = store.create_run(
        task="first",
        workspace=first_workspace,
        conversation_session_id=first_session.session_id,
    )
    second = store.create_run(
        task="second",
        workspace=second_workspace,
        conversation_session_id=second_session.session_id,
    )
    standalone = store.create_run(task="standalone", workspace=first_workspace)

    assert store.list_run_ids(workspace=first_workspace) == [
        first.run_id,
        standalone.run_id,
    ]
    assert store.list_run_ids(
        conversation_session_id=first_session.session_id
    ) == [first.run_id]
    assert store.latest_run_id(workspace=second_workspace) == second.run_id


def test_run_store_filtering_fails_on_invalid_current_run_metadata(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RunStore(root=tmp_path / "runs")
    run = store.create_run(
        task="broken",
        workspace=workspace,
        run_id="run_20260918_001",
    )
    (store.path_for(run.run_id) / "run.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError):
        store.list_run_ids(workspace=workspace)


def test_run_store_updates_terminal_session_state(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RunStore(root=tmp_path / "runs")
    session = store.create_run(
        task="noop",
        workspace=workspace,
        run_id="run_20260630_001",
    )

    updated = store.update_session_state(
        session.run_id,
        status="completed",
        current_step=11,
    )

    assert updated.status == "completed"
    assert updated.current_step == 11
    persisted = store.load_session(session.run_id)
    assert persisted.status == "completed"
    assert persisted.current_step == 11
    assert persisted.updated_at >= session.updated_at


def test_run_store_rejects_root_inside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="must not be inside the workspace"):
        RunStore(root=workspace / "runs").create_run(
            task="noop",
            workspace=workspace,
            run_id="run_20260630_001",
        )


def test_run_store_rejects_invalid_run_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="run_YYYYMMDD_NNN"):
        RunStore(root=tmp_path).create_run_directory("not-a-run-id")


def test_run_store_returns_latest_run_id(tmp_path) -> None:
    store = RunStore(root=tmp_path)
    (tmp_path / "run_20260630_001").mkdir()
    (tmp_path / "run_20260701_002").mkdir()
    (tmp_path / "run_20260701_001").mkdir()
    (tmp_path / "notes").mkdir()

    assert store.list_run_ids() == [
        "run_20260630_001",
        "run_20260701_001",
        "run_20260701_002",
    ]
    assert store.latest_run_id() == "run_20260701_002"


def test_run_store_latest_run_id_requires_existing_run(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No runs found"):
        RunStore(root=tmp_path).latest_run_id()
