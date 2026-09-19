from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from minicode_harness.memory.migration import migrate_v2_topics
from minicode_harness.memory.store import RepositoryMemoryStore
from minicode_harness.models import ModelResponse


class FakeModelClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests = []

    def call_request(self, request):
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return ModelResponse(final_text=json.dumps(self.response, ensure_ascii=False))


def _legacy_topic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "topic": "decisions",
        "entries": [
            {
                "entry_id": "active_1",
                "type": "decision",
                "summary": "Keep canonical history as the single conversation source.",
                "status": "active",
                "evidence_ids": ["old"],
                "created_at": "2026-09-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
            },
            {
                "entry_id": "inactive_1",
                "type": "decision",
                "summary": "Do not migrate this inactive entry.",
                "status": "inactive",
                "evidence_ids": ["old"],
                "created_at": "2026-09-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
            },
        ],
        "updated_at": "2026-09-01T00:00:00+00:00",
    }
    front = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip()
    path.write_text(f"---\n{front}\n---\n\n# Decisions\n", encoding="utf-8")


def _store(tmp_path: Path) -> RepositoryMemoryStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")


def test_v2_active_topics_migrate_once_through_phase2_init(tmp_path: Path) -> None:
    store = _store(tmp_path)
    legacy = store.memory_dir / "topics" / "decisions.md"
    _legacy_topic(legacy)
    (store.memory_dir / "workflow.json").write_text(
        '{"candidates":[{"text":"do not migrate candidate"}]}',
        encoding="utf-8",
    )
    client = FakeModelClient(
        {
            "memory_md": "# Memory\n\nMigrated canonical-history decision.\n",
            "memory_summary_md": "v1\n- Migrated repository decision.\n",
        }
    )

    result = migrate_v2_topics(store, client)

    assert result.status == "completed"
    state = store.load_state()
    assert state.latest_stage1_seq == 1
    assert state.last_phase2_input_seq == 1
    record = store.load_stage1("migration_v2")
    assert record is not None
    assert "single conversation source" in record.raw_memory
    assert "inactive" not in record.raw_memory.lower()
    payload = json.loads(client.requests[0].messages[0]["content"])
    assert payload["existing_memory_md"] == ""
    assert "do not migrate candidate" not in json.dumps(payload)
    assert legacy.exists()


def test_v2_migration_failure_keeps_legacy_files_and_no_v3_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    legacy = store.memory_dir / "topics" / "decisions.md"
    _legacy_topic(legacy)
    client = FakeModelClient(RuntimeError("provider unavailable"))

    with pytest.raises(RuntimeError):
        migrate_v2_topics(store, client)

    assert legacy.exists()
    assert not store.state_path.exists()
    assert not store.memory_path.exists()
    assert not store.summary_path.exists()
