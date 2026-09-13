from __future__ import annotations

import json
from pathlib import Path

import pytest

from minicode_harness.memory import (
    MAX_PENDING_CANDIDATES,
    MAX_PENDING_REVIEWS,
    MemoryApprovalCandidate,
    RepositoryMemoryStore,
    RepositoryMemoryWorkflowStore,
)



def _store(tmp_path: Path) -> RepositoryMemoryWorkflowStore:
    return RepositoryMemoryWorkflowStore(tmp_path / "memory")


def test_workflow_persists_review_and_deduplicates_run_id(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = store.append_review_record(
        source_run_id="run_1",
        user_text="remember focused tests",
        assistant_text="acknowledged",
    )
    duplicate = store.append_review_record(
        source_run_id="run_1",
        user_text="different",
        assistant_text="different",
    )

    assert first.status == "saved"
    assert duplicate.status == "already_saved"
    assert [item.source_run_id for item in store.pending_reviews()] == ["run_1"]
    assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == 1


def test_workflow_evicts_oldest_pending_reviews(tmp_path: Path) -> None:
    store = _store(tmp_path)

    for index in range(MAX_PENDING_REVIEWS + 3):
        store.append_review_record(
            source_run_id=f"run_{index}",
            user_text=f"turn {index}",
            assistant_text="done",
        )

    pending = store.pending_reviews()
    assert len(pending) == MAX_PENDING_REVIEWS
    assert pending[0].source_run_id == "run_3"


def test_review_result_consumes_records_and_creates_candidates_atomically(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for index in range(1, 3):
        store.append_review_record(
            source_run_id=f"run_{index}",
            user_text=f"turn {index}",
            assistant_text="done",
        )
    candidate = MemoryApprovalCandidate.create(
        topic="instructions",
        text="Use focused tests.",
        reason="Explicit repeated preference.",
        source_review_seqs=[1, 2],
    )

    store.save_review_result(reviewed_seqs=[1, 2], candidates=[candidate])

    assert store.pending_reviews() == []
    assert store.list_candidates(status="pending") == [candidate]


def test_candidate_capacity_is_checked_before_review_records_are_consumed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for index in range(MAX_PENDING_CANDIDATES):
        appended = store.append_review_record(
            source_run_id=f"run_{index}",
            user_text=f"candidate source {index}",
            assistant_text="done",
        )
        assert appended.record is not None
        candidate = MemoryApprovalCandidate.create(
            topic="instructions",
            text=f"candidate {index}",
            reason="test",
            source_review_seqs=[appended.record.seq],
        )
        store.save_review_result(
            reviewed_seqs=[appended.record.seq],
            candidates=[candidate],
        )

    extra_review = store.append_review_record(
        source_run_id="extra_run",
        user_text="extra candidate source",
        assistant_text="done",
    )
    assert extra_review.record is not None
    extra = MemoryApprovalCandidate.create(
        topic="instructions",
        text="extra candidate",
        reason="test",
        source_review_seqs=[extra_review.record.seq],
    )
    with pytest.raises(ValueError, match="capacity"):
        store.save_review_result(
            reviewed_seqs=[extra_review.record.seq],
            candidates=[extra],
        )

    assert [item.seq for item in store.pending_reviews()] == [
        extra_review.record.seq
    ]



def test_candidate_resolution_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_review_record(
        source_run_id="run_1",
        user_text="candidate source",
        assistant_text="done",
    )
    candidate = MemoryApprovalCandidate.create(
        topic="instructions",
        text="Use focused tests.",
        reason="Explicit preference.",
        source_review_seqs=[1],
    )
    store.save_review_result(reviewed_seqs=[1], candidates=[candidate])

    first = store.resolve_candidate(candidate.candidate_id, status="approved")
    second = store.resolve_candidate(candidate.candidate_id, status="approved")

    assert first.status == "approved"
    assert second == first
