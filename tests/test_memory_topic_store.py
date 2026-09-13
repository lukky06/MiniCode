from __future__ import annotations

from pathlib import Path

import pytest

from minicode_harness.memory import (
    MAX_MEMORY_INDEX_BYTES,
    MemoryTopicDocument,
    MemoryTopicEntry,
    RepositoryMemoryIndexStore,
    RepositoryMemoryTopicStore,
)


def test_topic_write_is_structured_atomic_and_index_is_bounded(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    topics = RepositoryMemoryTopicStore(memory_dir)
    entry = topics.add_entry(
        topic="build-and-test",
        entry_type="procedure",
        summary="Run python -m pytest -q tests/test_memory_policy.py for focused verification.",
        evidence_ids=["candidate_1"],
    )

    document = topics.read("build-and-test")
    assert document.entries[0].entry_id == entry.entry_id
    assert topics.path_for("build-and-test").read_text(encoding="utf-8").startswith("---\n")
    assert not topics.path_for("build-and-test").with_suffix(".md.tmp").exists()

    index = RepositoryMemoryIndexStore(memory_dir, topic_store=topics)
    rendered = index.refresh()
    event_count = len(index.event_store.list())
    unchanged = index.ensure()

    assert "build-and-test" in rendered
    assert "python -m pytest" not in rendered
    assert len(rendered.encode("utf-8")) <= MAX_MEMORY_INDEX_BYTES
    assert unchanged.status == "unchanged"
    assert unchanged.content == rendered
    assert len(index.event_store.list()) == event_count

    index.path.write_text("stale\n", encoding="utf-8")
    repaired = index.ensure()
    assert repaired.status == "repaired"
    assert repaired.content == rendered


def test_topic_store_rejects_unknown_topics_and_oversized_documents(tmp_path: Path) -> None:
    topics = RepositoryMemoryTopicStore(tmp_path / "memory")
    with pytest.raises(ValueError, match="Unknown memory topic"):
        topics.read("all")

    oversized = MemoryTopicDocument(
        topic="debugging",
        entries=[
            MemoryTopicEntry(
                entry_id=f"memory_{index}",
                type="pitfall",
                summary=(f"pitfall {index} " + "x" * 560),
                evidence_ids=[f"candidate_{index}"],
            )
            for index in range(16)
        ],
    )
    with pytest.raises(ValueError, match="exceeds"):
        topics.write(oversized)
