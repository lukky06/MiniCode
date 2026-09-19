from __future__ import annotations

from pathlib import Path
import threading

import pytest

import minicode_harness.memory.store as store_module

from minicode_harness.memory.recall import MemorySnapshotReader
from minicode_harness.memory.snapshot import MemorySnapshotStore
from minicode_harness.memory.store import RepositoryMemoryStore


def _seed(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    repository.write_durable_memory(
        "# Repository Memory\n\n## Testing\nUse focused tests.\n",
        "v1\n- Testing preferences are documented in MEMORY.md.\n",
    )
    summary_path = repository.write_rollout_summary(
        run_id="run_20260918_001",
        rollout_slug="focused-tests",
        content="# Rollout Summary\n\nFocused tests were confirmed.\n",
    )
    source = repository.capture_snapshot_source()
    snapshot_store = MemorySnapshotStore(tmp_path / "run")
    snapshot = snapshot_store.save(
        repository_id=repository.repository_id,
        memory_summary=source.memory_summary,
        memory_md=source.memory_md,
        rollout_summary_files=source.rollout_summary_files,
    )
    return repository, snapshot_store, snapshot, summary_path.name


def test_snapshot_ignores_legacy_memory_file_without_v3_summary(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    repository.memory_path.parent.mkdir(parents=True, exist_ok=True)
    repository.memory_path.write_text(
        "# Repository Memory Index\n\n- build-and-test\n",
        encoding="utf-8",
    )
    repository.write_rollout_summary(
        run_id="legacy",
        rollout_slug="old",
        content="# Legacy rollout\n",
    )

    source = repository.capture_snapshot_source()

    assert source.memory_summary == ""
    assert source.memory_md == ""
    assert source.rollout_summary_files == {}


def test_snapshot_waits_for_complete_durable_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    repository.write_durable_memory("# Memory\nold\n", "v1\nold\n")

    memory_replaced = threading.Event()
    continue_write = threading.Event()
    capture_done = threading.Event()
    captured = {}
    original_atomic_write = store_module._atomic_write_text

    def slow_atomic_write(path: Path, content: str) -> None:
        original_atomic_write(path, content)
        if path == repository.memory_path and "new" in content:
            memory_replaced.set()
            assert continue_write.wait(timeout=2)

    monkeypatch.setattr(store_module, "_atomic_write_text", slow_atomic_write)

    writer = threading.Thread(
        target=repository.write_durable_memory,
        args=("# Memory\nnew\n", "v1\nnew\n"),
    )
    writer.start()
    assert memory_replaced.wait(timeout=2)

    def capture() -> None:
        captured["source"] = repository.capture_snapshot_source()
        capture_done.set()

    reader = threading.Thread(target=capture)
    reader.start()
    assert capture_done.wait(timeout=0.1) is False

    continue_write.set()
    writer.join(timeout=2)
    reader.join(timeout=2)

    assert writer.is_alive() is False
    assert reader.is_alive() is False
    source = captured["source"]
    assert source.memory_md == "# Memory\nnew\n"
    assert source.memory_summary == "v1\nnew\n"


def test_v3_snapshot_freezes_summary_and_memory_handbook(tmp_path: Path) -> None:
    repository, snapshot_store, snapshot, rollout_name = _seed(tmp_path)

    assert snapshot.version == 3
    assert snapshot_store.read_summary().startswith("v1\n")
    assert "Use focused tests." in snapshot_store.read_memory()
    assert snapshot.rollout_summary_files == [rollout_name]

    repository.write_durable_memory(
        "# Repository Memory\n\nchanged\n",
        "v1\nchanged\n",
    )

    assert "Use focused tests." in snapshot_store.read_memory()
    assert "changed" not in snapshot_store.read_summary()


def test_snapshot_reader_searches_handbook_and_reads_only_captured_resources(
    tmp_path: Path,
) -> None:
    repository, snapshot_store, _, rollout_name = _seed(tmp_path)
    reader = MemorySnapshotReader(snapshot_store)

    searched = reader.search("focused", limit=10)
    handbook = reader.read("MEMORY.md", start_line=1, end_line=4)
    rollout = reader.read(f"rollout_summaries/{rollout_name}")

    assert searched["matches"][0]["path"] == "MEMORY.md"
    assert "focused tests" in searched["matches"][0]["text"].lower()
    assert handbook["path"] == "MEMORY.md"
    assert handbook["start_line"] == 1
    assert "Repository Memory" in handbook["content"]
    assert rollout["path"] == f"rollout_summaries/{rollout_name}"
    assert "Focused tests were confirmed." in rollout["content"]


@pytest.mark.parametrize(
    "target",
    [
        "../MEMORY.md",
        "rollout_summaries/../../outside.md",
        "rollout_summaries/not-captured.md",
        "raw_memories.md",
    ],
)
def test_snapshot_reader_rejects_paths_outside_the_frozen_view(
    tmp_path: Path,
    target: str,
) -> None:
    repository, snapshot_store, _, _ = _seed(tmp_path)
    reader = MemorySnapshotReader(snapshot_store)

    with pytest.raises((FileNotFoundError, ValueError)):
        reader.read(target)
