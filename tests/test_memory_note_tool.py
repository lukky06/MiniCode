from __future__ import annotations

from pathlib import Path

from minicode_harness.memory import RepositoryMemoryPublisher, RepositoryMemoryStore


def test_repository_memory_publisher_remains_available_for_cli_workflow(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    publisher = RepositoryMemoryPublisher(memory)

    first = publisher.publish(
        topic="instructions",
        text="默认只运行聚焦测试。",
        evidence_ids=["cli:memory-remember"],
    )
    duplicate = publisher.publish(
        topic="instructions",
        text="默认只运行聚焦测试。",
        evidence_ids=["cli:memory-remember"],
    )

    assert first.status == "saved"
    assert duplicate.status == "already_exists"
    assert duplicate.entry_id == first.entry_id
    assert [entry.summary for entry in memory.topic_store.active_entries("instructions")] == [
        "默认只运行聚焦测试。"
    ]
    assert "instructions" in memory.render_index()
