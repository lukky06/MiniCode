from __future__ import annotations

from pathlib import Path

from minicode_harness.memory import MemoryApprovalCandidate, RepositoryMemoryStore


def test_pending_candidate_stays_in_workflow_until_resolved(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    appended = store.workflow_store.append_review_record(
        source_run_id="run_1",
        user_text="以后只运行聚焦测试。",
        assistant_text="已记录候选。",
    )
    assert appended.record is not None
    candidate = MemoryApprovalCandidate.create(
        topic="instructions",
        text="以后只运行聚焦测试。",
        reason="稳定用户约束。",
        source_review_seqs=[appended.record.seq],
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[appended.record.seq],
        candidates=[candidate],
    )

    assert [item.candidate_id for item in store.workflow_store.list_candidates(status="pending")] == [
        candidate.candidate_id
    ]
    assert store.topic_store.active_entries("instructions") == []
