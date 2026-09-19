from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest

from minicode_harness.memory.store import RepositoryMemoryStore


def _store(tmp_path: Path) -> RepositoryMemoryStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")


def test_state_defaults_to_clean_v3_without_creating_file(tmp_path: Path) -> None:
    store = _store(tmp_path)

    state = store.load_state()

    assert state.schema_version == 3
    assert state.latest_stage1_seq == 0
    assert state.last_phase2_input_seq == 0
    assert state.last_phase2_success_at is None
    assert not store.state_path.exists()


def test_no_output_is_terminal_without_advancing_stage1_sequence(tmp_path: Path) -> None:
    store = _store(tmp_path)

    record = store.write_stage1_no_output("run_20260918_001")

    assert record.status == "no_output"
    assert record.seq is None
    assert store.load_state().latest_stage1_seq == 0
    assert store.load_stage1(record.run_id) == record


def test_memory_stage1_records_receive_monotonic_sequences(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = store.write_stage1_memory(
        run_id="run_20260918_001",
        raw_memory="Prefer focused tests.",
        rollout_summary="The user confirmed focused verification.",
        rollout_slug="focused-tests",
    )
    second = store.write_stage1_memory(
        run_id="run_20260918_002",
        raw_memory="Git commands can hang on Windows.",
        rollout_summary="Diagnosed a Windows Git subprocess hang.",
        rollout_slug="windows-git-hang",
    )

    assert first.seq == 1
    assert second.seq == 2
    assert store.load_state().latest_stage1_seq == 2
    assert [record.run_id for record in store.pending_stage1_records()] == [
        "run_20260918_001",
        "run_20260918_002",
    ]


def test_stage1_terminal_record_cannot_be_overwritten(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_stage1_no_output("run_20260918_001")

    with pytest.raises(FileExistsError):
        store.write_stage1_memory(
            run_id="run_20260918_001",
            raw_memory="new",
            rollout_summary="new",
            rollout_slug="new",
        )


def test_pipeline_lock_is_non_blocking_for_same_repository(tmp_path: Path) -> None:
    store = _store(tmp_path)
    other = RepositoryMemoryStore(
        store.workspace,
        data_dir=store.data_dir,
    )

    first = store.try_pipeline_lock()
    assert first is not None
    try:
        assert other.try_pipeline_lock() is None
    finally:
        first.release()

    acquired_again = other.try_pipeline_lock()
    assert acquired_again is not None
    acquired_again.release()


def test_pipeline_lock_is_cross_process(tmp_path: Path) -> None:
    store = _store(tmp_path)
    held = store.try_pipeline_lock()
    assert held is not None
    script = (
        "from minicode_harness.memory.store import RepositoryMemoryStore;"
        f"s=RepositoryMemoryStore({json.dumps(str(store.workspace))},"
        f"data_dir={json.dumps(str(store.data_dir))});"
        "lock=s.try_pipeline_lock();"
        "print('busy' if lock is None else 'acquired');"
        "lock.release() if lock is not None else None"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )
    finally:
        held.release()

    assert completed.stdout.strip() == "busy"
