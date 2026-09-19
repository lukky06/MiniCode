from __future__ import annotations

from pathlib import Path

from minicode_harness.memory.recall import MemorySnapshotReader
from minicode_harness.memory.snapshot import MemorySnapshotStore
from minicode_harness.memory.store import RepositoryMemoryStore
from minicode_harness.tools import ToolRegistry


def _registry(tmp_path: Path) -> ToolRegistry:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    repository.write_durable_memory(
        "# Memory\n\n## Windows Git\nUse bounded subprocess timeouts.\n",
        "v1\n- Windows Git notes are available in MEMORY.md.\n",
    )
    repository.write_rollout_summary(
        run_id="run_20260918_001",
        rollout_slug="windows-git",
        content="# Rollout Summary\n\nWindows Git hang was diagnosed.\n",
    )
    source = repository.capture_snapshot_source()
    snapshot_store = MemorySnapshotStore(tmp_path / "run")
    snapshot_store.save(
        repository_id=repository.repository_id,
        memory_summary=source.memory_summary,
        memory_md=source.memory_md,
        rollout_summary_files=source.rollout_summary_files,
    )
    reader = MemorySnapshotReader(snapshot_store)
    return ToolRegistry(
        str(workspace),
        memory_reader=reader.read,
        memory_searcher=reader.search,
    )


def test_read_memory_accepts_snapshot_paths_and_line_ranges(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    result = registry.execute_admitted(
        registry.admit(
            "read",
            {
                "source": "memory",
                "target": "MEMORY.md",
                "start_line": 3,
                "end_line": 4,
            },
        )
    )

    assert result["path"] == "MEMORY.md"
    assert result["start_line"] == 3
    assert "Windows Git" in result["content"]


def test_search_memory_uses_existing_search_tool_surface(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    result = registry.execute_admitted(
        registry.admit(
            "search",
            {
                "source": "memory",
                "kind": "text",
                "query": "git hang",
                "limit": 10,
                "case_sensitive": False,
            },
        )
    )

    assert result["matches"][0]["path"].startswith("rollout_summaries/")
    assert "Git hang" in result["matches"][0]["text"]


def test_search_memory_rejects_file_discovery_shape(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    admitted = registry.admit(
        "search",
        {
            "source": "memory",
            "kind": "text",
            "query": "Windows",
        },
    )
    result = registry.execute_admitted(admitted)

    assert result["matches"]
