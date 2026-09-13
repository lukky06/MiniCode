from __future__ import annotations

import json
from pathlib import Path

import pytest

from minicode_harness.memory import (
    MAX_PENDING_CANDIDATES,
    MemoryApprovalCandidate,
    MemoryCandidateEvidence,
    MemoryManualReviewError,
    MemoryManualReviewService,
    MemoryResolvedFailure,
    MemoryVerificationEvidence,
    RepositoryMemoryCandidateService,
    RepositoryMemoryStore,
)
from minicode_harness.memory.manual_review import MAX_REVIEW_INPUT_CHARS
from minicode_harness.models import ModelResponse


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests = []

    def call_request(self, request):
        self.requests.append(request)
        return ModelResponse(final_text=json.dumps(self.response))


def _store(tmp_path: Path) -> RepositoryMemoryStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")


def test_manual_review_uses_one_model_call_and_creates_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.workflow_store.append_review_record(
        source_run_id="run_1",
        user_text="以后修改代码后只运行聚焦测试。",
        assistant_text="已按要求完成。",
        verification=MemoryVerificationEvidence(
            status="passed",
            command="python -m pytest tests/test_memory_v2_manual_review.py -q",
            returncode=0,
        ),
        resolved_failure=MemoryResolvedFailure(
            failed_command="python -m pytest tests/test_memory_v2_manual_review.py -q",
            failure_summary="candidate evidence assertion failed",
            modified_files=["tests/test_memory_v2_manual_review.py"],
            passed_command="python -m pytest tests/test_memory_v2_manual_review.py -q",
        ),
    )
    client = FakeClient(
        {
            "candidates": [
                {
                    "topic": "build-and-test",
                    "text": "修改代码后只运行聚焦测试。",
                    "reason": "用户明确给出跨会话测试偏好。",
                    "source_review_seqs": [1],
                }
            ]
        }
    )

    result = MemoryManualReviewService(client, store=store).review()

    assert result.status == "completed"
    assert result.reviewed_seqs == [1]
    assert len(result.candidate_ids) == 1
    assert store.workflow_store.pending_reviews() == []
    candidates = store.workflow_store.list_candidates(status="pending")
    assert [item.candidate_id for item in candidates] == result.candidate_ids
    assert len(candidates[0].evidence) == 1
    assert candidates[0].evidence[0].source_run_id == "run_1"
    assert candidates[0].evidence[0].verification is not None
    assert candidates[0].evidence[0].verification.status == "passed"
    assert candidates[0].evidence[0].resolved_failure is not None
    assert candidates[0].evidence[0].resolved_failure.modified_files == [
        "tests/test_memory_v2_manual_review.py"
    ]
    assert len(client.requests) == 1
    assert client.requests[0].tools == []
    assert client.requests[0].metadata["temperature"] == 0.0
    payload = json.loads(client.requests[0].messages[0]["content"])
    assert set(payload) == {
        "repository_id",
        "allowed_topics",
        "memory_index",
        "active_memory_catalog",
        "max_candidates",
        "review_records",
    }
    assert "topics" not in payload
    assert {item["topic"] for item in payload["active_memory_catalog"]} == {
        "instructions",
        "build-and-test",
        "debugging",
        "decisions",
        "environment",
    }


def test_manual_review_accepts_zero_candidates_and_consumes_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.workflow_store.append_review_record(
        source_run_id="run_1",
        user_text="解释当前文件。",
        assistant_text="已解释。",
    )
    client = FakeClient({"candidates": []})

    result = MemoryManualReviewService(client, store=store).review()

    assert result.status == "reviewed_no_candidates"
    assert result.candidate_ids == []
    assert store.workflow_store.pending_reviews() == []
    assert len(client.requests) == 1


def test_manual_review_invalid_source_keeps_records_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.workflow_store.append_review_record(
        source_run_id="run_1",
        user_text="记住聚焦测试。",
        assistant_text="完成。",
    )
    client = FakeClient(
        {
            "candidates": [
                {
                    "topic": "instructions",
                    "text": "Use focused tests.",
                    "reason": "Stable preference.",
                    "source_review_seqs": [99],
                }
            ]
        }
    )

    with pytest.raises(MemoryManualReviewError, match="not in the review batch"):
        MemoryManualReviewService(client, store=store).review()

    assert [item.seq for item in store.workflow_store.pending_reviews()] == [1]
    assert store.workflow_store.list_candidates() == []
    assert len(client.requests) == 1


def test_manual_review_rejects_unmatched_auto_publish_quote(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.workflow_store.append_review_record(
        source_run_id="run_1",
        user_text="团队长期约定只运行聚焦测试。",
        assistant_text="完成。",
    )
    client = FakeClient(
        {
            "candidates": [
                {
                    "topic": "build-and-test",
                    "text": "只运行聚焦测试。",
                    "reason": "稳定规则。",
                    "source_review_seqs": [1],
                    "auto_publish": True,
                    "source_quotes": ["用户没有说过的引文"],
                }
            ]
        }
    )

    with pytest.raises(MemoryManualReviewError, match="source quote"):
        MemoryManualReviewService(client, store=store).review()

    assert [item.seq for item in store.workflow_store.pending_reviews()] == [1]
    assert store.workflow_store.list_candidates() == []


def test_manual_review_keeps_conflicting_candidate_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    existing = store.topic_store.add_entry(
        topic="decisions",
        entry_type="decision",
        summary="缓存身份只使用 ticket_id。",
        evidence_ids=["test:existing"],
    )
    store.refresh_index()
    source_quote = "团队长期约定缓存身份由 tenant_id 与 ticket_id 共同构成。"
    store.workflow_store.append_review_record(
        source_run_id="run_conflict",
        user_text=source_quote,
        assistant_text="完成。",
    )
    client = FakeClient(
        {
            "candidates": [
                {
                    "topic": "decisions",
                    "text": source_quote,
                    "reason": "新规则与现有决策冲突。",
                    "source_review_seqs": [1],
                    "auto_publish": True,
                    "source_quotes": [source_quote],
                    "reviewed_entry_ids": [existing.entry_id],
                    "conflicts_with_entry_ids": [existing.entry_id],
                }
            ]
        }
    )

    result = MemoryManualReviewService(client, store=store).review()

    assert result.status == "completed"
    candidate = store.workflow_store.list_candidates(status="pending")[0]
    assert candidate.auto_publish is False
    assert candidate.reviewed_entry_ids == [existing.entry_id]
    assert candidate.conflicts_with_entry_ids == [existing.entry_id]
    assert RepositoryMemoryCandidateService(store).auto_approve(candidate.candidate_id) is None
    assert [entry.summary for entry in store.topic_store.active_entries("decisions")] == [
        "缓存身份只使用 ticket_id。"
    ]


def test_auto_approve_rechecks_topic_snapshot_before_publication(tmp_path: Path) -> None:
    store = _store(tmp_path)
    existing = store.topic_store.add_entry(
        topic="instructions",
        entry_type="user_instruction",
        summary="只运行聚焦测试。",
        evidence_ids=["test:existing"],
    )
    source_quote = "不要修改生成目录。"
    review = store.workflow_store.append_review_record(
        source_run_id="run_snapshot",
        user_text=source_quote,
        assistant_text="完成。",
    )
    assert review.record is not None
    candidate = MemoryApprovalCandidate.create(
        topic="instructions",
        text=source_quote,
        reason="明确的仓库约定。",
        source_review_seqs=[review.record.seq],
        evidence=[MemoryCandidateEvidence.from_review_record(review.record)],
        auto_publish=True,
        source_quotes=[source_quote],
        reviewed_entry_ids=[existing.entry_id],
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[review.record.seq],
        candidates=[candidate],
    )
    store.topic_store.add_entry(
        topic="instructions",
        entry_type="user_instruction",
        summary="提交前运行静态检查。",
        evidence_ids=["test:concurrent"],
    )

    assert RepositoryMemoryCandidateService(store).auto_approve(candidate.candidate_id) is None
    assert store.workflow_store.get_candidate(candidate.candidate_id).status == "pending"


def test_memory_candidate_rejects_sensitive_source_quote(tmp_path: Path) -> None:
    store = _store(tmp_path)
    review = store.workflow_store.append_review_record(
        source_run_id="run_sensitive",
        user_text="发布说明不得包含凭据。",
        assistant_text="完成。",
    )
    assert review.record is not None

    with pytest.raises(ValueError, match="sensitive content"):
        MemoryApprovalCandidate.create(
            topic="instructions",
            text="发布说明不得包含凭据。",
            reason="安全要求。",
            source_review_seqs=[review.record.seq],
            source_quotes=["api_key=sk-test-do-not-store"],
            auto_publish=True,
        )


def test_manual_review_shrinks_large_batch_to_input_budget(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(8):
        store.workflow_store.append_review_record(
            source_run_id=f"large_{index}",
            user_text=f"规则 {index}: " + ("u" * 3900),
            assistant_text="a" * 3900,
        )
    client = FakeClient({"candidates": []})

    result = MemoryManualReviewService(client, store=store).review()

    assert 1 <= len(result.reviewed_seqs) < 8
    assert result.pending_reviews == 8 - len(result.reviewed_seqs)
    assert len(client.requests) == 1
    serialized = client.requests[0].messages[0]["content"]
    assert len(serialized) <= MAX_REVIEW_INPUT_CHARS


def test_manual_review_does_not_call_model_when_candidate_store_is_full(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for index in range(MAX_PENDING_CANDIDATES):
        appended = store.workflow_store.append_review_record(
            source_run_id=f"source_{index}",
            user_text=f"source {index}",
            assistant_text="done",
        )
        assert appended.record is not None
        candidate = MemoryApprovalCandidate.create(
            topic="instructions",
            text=f"candidate {index}",
            reason="test",
            source_review_seqs=[appended.record.seq],
        )
        store.workflow_store.save_review_result(
            reviewed_seqs=[appended.record.seq],
            candidates=[candidate],
        )
    store.workflow_store.append_review_record(
        source_run_id="waiting",
        user_text="waiting review",
        assistant_text="done",
    )
    client = FakeClient({"candidates": []})

    result = MemoryManualReviewService(client, store=store).review()

    assert result.status == "candidate_store_full"
    assert len(store.workflow_store.pending_reviews()) == 1
    assert client.requests == []


def test_candidate_approve_and_reject_are_deterministic_and_idempotent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first_review = store.workflow_store.append_review_record(
        source_run_id="approve_source",
        user_text="remember focused tests",
        assistant_text="done",
    )
    assert first_review.record is not None
    approved_candidate = MemoryApprovalCandidate.create(
        topic="build-and-test",
        text="Use focused tests.",
        reason="Explicit preference.",
        source_review_seqs=[first_review.record.seq],
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[first_review.record.seq],
        candidates=[approved_candidate],
    )
    service = RepositoryMemoryCandidateService(store)

    first = service.approve(approved_candidate.candidate_id)
    second = service.approve(approved_candidate.candidate_id)

    assert first.status == "applied"
    assert first.entry_id == f"memory_{approved_candidate.candidate_id}"
    assert second.status == "idempotent_noop"
    assert len(store.topic_store.active_entries("build-and-test")) == 1
    assert store.workflow_store.get_candidate(approved_candidate.candidate_id).status == (
        "approved"
    )

    second_review = store.workflow_store.append_review_record(
        source_run_id="reject_source",
        user_text="temporary detail",
        assistant_text="done",
    )
    assert second_review.record is not None
    rejected_candidate = MemoryApprovalCandidate.create(
        topic="instructions",
        text="Temporary detail.",
        reason="User should decide.",
        source_review_seqs=[second_review.record.seq],
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[second_review.record.seq],
        candidates=[rejected_candidate],
    )

    rejected = service.reject(rejected_candidate.candidate_id)
    repeated = service.reject(rejected_candidate.candidate_id)

    assert rejected.status == "applied"
    assert repeated.status == "idempotent_noop"
    assert store.workflow_store.get_candidate(rejected_candidate.candidate_id).status == (
        "rejected"
    )
    assert store.topic_store.active_entries("instructions") == []


def test_auto_approve_accepts_multiple_exact_source_quotes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    quotes = [
        "缓存身份包含 tenant_id。",
        "缓存身份同时包含 ticket_id。",
    ]
    records = [
        store.workflow_store.append_review_record(
            source_run_id=f"quoted_{index}",
            user_text=quote,
            assistant_text="完成。",
        ).record
        for index, quote in enumerate(quotes, start=1)
    ]
    assert all(record is not None for record in records)
    resolved_records = [record for record in records if record is not None]
    candidate = MemoryApprovalCandidate.create(
        topic="decisions",
        text=" ".join(quotes),
        reason="两条明确规则属于同一决策。",
        source_review_seqs=[record.seq for record in resolved_records],
        evidence=[
            MemoryCandidateEvidence.from_review_record(record)
            for record in resolved_records
        ],
        source_quotes=quotes,
        auto_publish=True,
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[record.seq for record in resolved_records],
        candidates=[candidate],
    )

    approved = RepositoryMemoryCandidateService(store).auto_approve(candidate.candidate_id)

    assert approved is not None
    assert approved.candidate_status == "approved"
    assert [entry.summary for entry in store.topic_store.active_entries("decisions")] == [
        " ".join(quotes)
    ]


def test_auto_approve_rejects_candidate_text_not_equal_to_source_quotes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source_quote = "团队长期约定只运行聚焦测试。"
    review = store.workflow_store.append_review_record(
        source_run_id="quoted_source",
        user_text=source_quote,
        assistant_text="完成。",
    )
    assert review.record is not None
    candidate = MemoryApprovalCandidate.create(
        topic="build-and-test",
        text="团队长期约定只运行聚焦测试，并且禁止所有集成测试。",
        reason="包含未经用户确认的附加结论。",
        source_review_seqs=[review.record.seq],
        evidence=[MemoryCandidateEvidence.from_review_record(review.record)],
        source_quotes=[source_quote],
        auto_publish=True,
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[review.record.seq],
        candidates=[candidate],
    )

    assert RepositoryMemoryCandidateService(store).auto_approve(candidate.candidate_id) is None
    assert store.workflow_store.get_candidate(candidate.candidate_id).status == "pending"
    assert store.topic_store.active_entries("build-and-test") == []


def test_auto_approve_requires_structured_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    review = store.workflow_store.append_review_record(
        source_run_id="verified_source",
        user_text="验证修改。",
        assistant_text="聚焦测试已通过。",
        verification=MemoryVerificationEvidence(
            status="passed",
            command="python -m pytest tests/test_cache.py -q",
            returncode=0,
        ),
    )
    assert review.record is not None
    safe = MemoryApprovalCandidate.create(
        topic="build-and-test",
        text="先运行 python -m pytest tests/test_cache.py -q。",
        reason="已验证的聚焦测试入口。",
        source_review_seqs=[review.record.seq],
        evidence=[MemoryCandidateEvidence.from_review_record(review.record)],
        auto_publish=True,
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[review.record.seq],
        candidates=[safe],
    )

    approved = RepositoryMemoryCandidateService(store).auto_approve(safe.candidate_id)

    assert approved is not None
    assert approved.candidate_status == "approved"
    assert len(store.topic_store.active_entries("build-and-test")) == 1

    second_review = store.workflow_store.append_review_record(
        source_run_id="unverified_source",
        user_text="分析新的构建流程。",
        assistant_text="可能使用某个命令。",
    )
    assert second_review.record is not None
    unsafe = MemoryApprovalCandidate.create(
        topic="build-and-test",
        text="运行任意测试命令。",
        reason="没有验证证据。",
        source_review_seqs=[second_review.record.seq],
        evidence=[MemoryCandidateEvidence.from_review_record(second_review.record)],
        auto_publish=True,
    )
    store.workflow_store.save_review_result(
        reviewed_seqs=[second_review.record.seq],
        candidates=[unsafe],
    )

    assert RepositoryMemoryCandidateService(store).auto_approve(unsafe.candidate_id) is None
    assert store.workflow_store.get_candidate(unsafe.candidate_id).status == "pending"
