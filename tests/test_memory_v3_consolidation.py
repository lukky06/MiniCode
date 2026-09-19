from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from minicode_harness.memory.consolidation import Phase2Consolidator
from minicode_harness.memory.store import RepositoryMemoryStore
from minicode_harness.memory.types import MemoryPipelineState
from minicode_harness.models import ModelResponse


NOW = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)


class FakeModelClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests = []

    def call_request(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(final_text=json.dumps(response, ensure_ascii=False))


def _store(tmp_path: Path) -> RepositoryMemoryStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")


def _memory_output(label: str = "focused") -> dict[str, str]:
    return {
        "memory_md": f"# Durable Memory\n\n- {label}\n",
        "memory_summary_md": f"v1\n- {label}\n",
    }


def _add_stage1(store: RepositoryMemoryStore, run_id: str, label: str) -> None:
    store.write_stage1_memory(
        run_id=run_id,
        raw_memory=f"raw {label}",
        rollout_summary=f"summary {label}",
        rollout_slug=label,
    )


def test_phase2_materializes_rollout_summary_only_after_model_output(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _add_stage1(store, "run_20260918_001", "focused-tests")

    class InspectingClient:
        def call_request(self, request):
            assert list(store.rollout_summaries_dir.glob("*.md")) == []
            return ModelResponse(
                final_text=json.dumps(_memory_output(), ensure_ascii=False)
            )

    result = Phase2Consolidator(store, InspectingClient()).consolidate(now=NOW)

    assert result.status == "completed"
    assert [path.name for path in store.rollout_summaries_dir.glob("*.md")] == [
        "run_20260918_001--focused-tests.md"
    ]


def test_phase2_init_runs_immediately_and_publishes_pending_delta(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _add_stage1(store, "run_20260918_001", "focused-tests")
    client = FakeModelClient([_memory_output()])

    result = Phase2Consolidator(store, client).consolidate(now=NOW)

    assert result.status == "completed"
    assert result.mode == "init"
    assert store.memory_path.read_text(encoding="utf-8").startswith("# Durable Memory")
    assert store.summary_path.read_text(encoding="utf-8").startswith("v1\n")
    assert "raw focused-tests" in store.raw_memories_path.read_text(encoding="utf-8")
    summaries = list(store.rollout_summaries_dir.glob("*.md"))
    assert [path.name for path in summaries] == [
        "run_20260918_001--focused-tests.md"
    ]
    assert "summary focused-tests" in summaries[0].read_text(encoding="utf-8")
    state = store.load_state()
    assert state.latest_stage1_seq == 1
    assert state.last_phase2_input_seq == 1
    assert state.last_phase2_success_at == NOW.isoformat()
    request = client.requests[0]
    assert request.tools == []
    payload = json.loads(request.messages[0]["content"])
    assert "raw focused-tests" in payload["raw_memories_md"]
    assert payload["existing_memory_md"] == ""
    assert payload["existing_memory_summary_md"] == ""


def test_incremental_phase2_respects_six_hour_cooldown(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_durable_memory("# Memory\n", "v1\nold\n")
    _add_stage1(store, "run_20260918_001", "first")
    store.save_state(
        MemoryPipelineState(
            latest_stage1_seq=1,
            last_phase2_input_seq=0,
            last_phase2_success_at=(NOW - timedelta(hours=2)).isoformat(),
        )
    )
    client = FakeModelClient([_memory_output()])

    result = Phase2Consolidator(store, client).consolidate(now=NOW)

    assert result.status == "skipped"
    assert result.reason == "cooldown"
    assert client.requests == []
    assert store.load_state().last_phase2_input_seq == 0


def test_incremental_phase2_runs_after_cooldown_and_uses_only_pending_records(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.write_durable_memory("# Memory\nold\n", "v1\nold\n")
    _add_stage1(store, "run_20260918_001", "old")
    _add_stage1(store, "run_20260918_002", "new")
    store.save_state(
        MemoryPipelineState(
            latest_stage1_seq=2,
            last_phase2_input_seq=1,
            last_phase2_success_at=(NOW - timedelta(hours=6)).isoformat(),
        )
    )
    client = FakeModelClient([_memory_output("new")])

    result = Phase2Consolidator(store, client).consolidate(now=NOW)

    assert result.status == "completed"
    assert result.mode == "incremental"
    raw = store.raw_memories_path.read_text(encoding="utf-8")
    assert "raw new" in raw
    assert "raw old" not in raw
    payload = json.loads(client.requests[0].messages[0]["content"])
    assert payload["existing_memory_md"] == "# Memory\nold\n"
    assert payload["existing_memory_summary_md"] == "v1\nold\n"
    assert [item["run_id"] for item in payload["rollout_summaries"]] == [
        "run_20260918_002"
    ]


def test_explicit_consolidation_bypasses_cooldown_and_clean_state_is_noop(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.write_durable_memory("# Memory\nold\n", "v1\nold\n")
    _add_stage1(store, "run_20260918_001", "new")
    store.save_state(
        MemoryPipelineState(
            latest_stage1_seq=1,
            last_phase2_input_seq=0,
            last_phase2_success_at=NOW.isoformat(),
        )
    )
    client = FakeModelClient([_memory_output("new")])

    completed = Phase2Consolidator(store, client).consolidate(
        explicit=True,
        now=NOW,
    )
    noop = Phase2Consolidator(store, client).consolidate(
        explicit=True,
        now=NOW,
    )

    assert completed.status == "completed"
    assert noop.status == "noop"
    assert noop.reason == "clean"
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"memory_md": "# Memory\n", "memory_summary_md": "wrong-version\n"},
        {"memory_md": "# Memory\n", "memory_summary_md": "v1\n" + ("x" * 5000)},
    ],
)
def test_invalid_phase2_output_does_not_publish_or_advance_state(
    tmp_path: Path,
    response: dict[str, str],
) -> None:
    store = _store(tmp_path)
    store.write_durable_memory("# Memory\nold\n", "v1\nold\n")
    _add_stage1(store, "run_20260918_001", "new")
    original = MemoryPipelineState(
        latest_stage1_seq=1,
        last_phase2_input_seq=0,
        last_phase2_success_at=(NOW - timedelta(hours=8)).isoformat(),
    )
    store.save_state(original)
    client = FakeModelClient([response])

    with pytest.raises(ValueError):
        Phase2Consolidator(store, client).consolidate(now=NOW)

    assert store.memory_path.read_text(encoding="utf-8") == "# Memory\nold\n"
    assert store.summary_path.read_text(encoding="utf-8") == "v1\nold\n"
    assert store.load_state() == original


def test_rollout_summary_is_immutable_and_slug_cannot_escape_memory_root(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.write_stage1_memory(
        run_id="run_20260918_001",
        raw_memory="raw",
        rollout_summary="summary",
        rollout_slug="../../windows/git",
    )
    client = FakeModelClient([_memory_output()])

    Phase2Consolidator(store, client).consolidate(now=NOW)

    summaries = list(store.rollout_summaries_dir.glob("*.md"))
    assert len(summaries) == 1
    assert summaries[0].parent == store.rollout_summaries_dir
    assert summaries[0].name.startswith("run_20260918_001--")
    original = summaries[0].read_text(encoding="utf-8")

    store.save_state(
        store.load_state().model_copy(update={"last_phase2_input_seq": 0})
    )
    FakeModelClient2 = FakeModelClient([_memory_output("again")])
    Phase2Consolidator(store, FakeModelClient2).consolidate(
        explicit=True,
        now=NOW + timedelta(hours=1),
    )

    assert summaries[0].read_text(encoding="utf-8") == original
