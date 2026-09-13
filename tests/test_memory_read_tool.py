from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from minicode_harness.memory import RepositoryMemoryStore
from minicode_harness.tools import ToolRegistry


def _seed_memory(tmp_path: Path) -> tuple[Path, RepositoryMemoryStore]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    memory.topic_store.add_entry(
        topic="build-and-test",
        entry_type="procedure",
        summary="Run python -m pytest -q tests/test_memory_policy.py.",
        evidence_ids=["candidate_test"],
    )
    memory.refresh_index()
    return workspace, memory


def test_read_memory_is_static_read_only_and_index_bounded(tmp_path: Path) -> None:
    workspace, memory = _seed_memory(tmp_path)
    registry = ToolRegistry(
        str(workspace),
        memory_topic_reader=memory.read_topic,
    )

    names = [schema["function"]["name"] for schema in registry.schemas()]
    assert "read" in names
    assert "read_memory" not in names
    assert registry.history_effects()["read"] == {
        "read_only": True,
        "destructive": False,
        "result_reconstructible": True,
    }

    schema = next(
        item
        for item in registry.schemas()
        if item["function"]["name"] == "read"
    )
    description = schema["function"]["description"]
    assert "For Memory" in description
    assert "read at most two exact indexed Topics per user turn" in description
    assert "exact indexed Topics" in description

    result = registry.execute_admitted(
        registry.admit(
            "read",
            {"source": "memory", "target": "build-and-test"},
        )
    )
    assert result["topic"] == "build-and-test"
    assert result["content"].startswith(
        "Repository Memory Requirements (build-and-test):"
    )
    assert "Treat every listed requirement" in result["content"]
    assert "test_memory_policy.py" in result["content"]
    assert "---" not in result["content"]
    assert len(result["content"].encode("utf-8")) <= 8 * 1024
    assert result["content_hash"]


def test_read_memory_rejects_unindexed_topics_and_arbitrary_paths(tmp_path: Path) -> None:
    workspace, memory = _seed_memory(tmp_path)
    registry = ToolRegistry(
        str(workspace),
        memory_topic_reader=memory.read_topic,
    )

    with pytest.raises(FileNotFoundError, match="not registered"):
        registry.execute_admitted(
            registry.admit(
                "read",
                {"source": "memory", "target": "debugging"},
            )
        )

    with pytest.raises(ValidationError):
        registry.admit(
            "read",
            {"source": "memory", "target": "all"},
        )

    with pytest.raises(ValidationError):
        registry.admit(
            "read",
            {"source": "memory", "path": "MEMORY.md"},
        )
