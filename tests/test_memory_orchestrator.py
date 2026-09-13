from __future__ import annotations

import json
from pathlib import Path

from minicode_harness.context import ContextObservation, VerificationState
from minicode_harness.memory import MAX_PENDING_REVIEWS, RepositoryMemoryStore
from minicode_harness.models import ModelResponse
from minicode_harness.runtime.request_orchestrator import RequestOrchestrator


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests = []

    def call_request(self, request):
        self.requests.append(request)
        return ModelResponse(final_text=json.dumps(self.response, ensure_ascii=False))


class FailingClient:
    def call_request(self, request):
        raise RuntimeError("review unavailable")


def _repository(tmp_path: Path) -> RepositoryMemoryStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")


def test_orchestrator_renders_repository_index_without_model_dependency(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.topic_store.add_entry(
        topic="decisions",
        entry_type="decision",
        summary="ContextBuilder only builds the System Prompt.",
        evidence_ids=["test:index"],
    )
    repository.refresh_index()

    context = RequestOrchestrator(repository_memory=repository).recall_context()

    assert "decisions" in context
    assert "ContextBuilder only builds the System Prompt." not in context


def test_completed_run_records_review_without_model_call(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = RequestOrchestrator(repository_memory=repository)

    result = orchestrator.finalize_completed_run(
        run_id="run_1",
        user_input="以后修改代码后只运行聚焦测试。",
        assistant_text="已完成。",
    )

    assert result.status == "review_recorded"
    assert result.pending_user_turns == 1
    assert [turn.user_text for turn in repository.workflow_store.pending_reviews()] == [
        "以后修改代码后只运行聚焦测试。"
    ]
    assert repository.workflow_store.list_candidates() == []


def test_completed_run_is_idempotent_for_same_run_id(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = RequestOrchestrator(repository_memory=repository)

    first = orchestrator.finalize_completed_run(
        run_id="run_1",
        user_input="第一次。",
        assistant_text="完成。",
    )
    second = orchestrator.finalize_completed_run(
        run_id="run_1",
        user_input="重复。",
        assistant_text="完成。",
    )

    assert first.status == "review_recorded"
    assert second.status == "idempotent_noop"
    assert len(repository.workflow_store.pending_reviews()) == 1


def test_eight_completed_runs_never_trigger_memory_model_calls(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = RequestOrchestrator(repository_memory=repository)

    for index in range(1, 9):
        result = orchestrator.finalize_completed_run(
            run_id=f"run_{index}",
            user_input=f"turn {index}",
            assistant_text="完成。",
        )
        assert result.status == "review_recorded"

    assert len(repository.workflow_store.pending_reviews()) == 8
    assert repository.topic_store.registered_topics() == []


def test_auto_review_runs_once_at_threshold_and_publishes_safe_add(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    source_quote = "团队长期约定缓存身份由 tenant_id 与 ticket_id 共同构成。"
    client = FakeClient(
        {
            "candidates": [
                {
                    "topic": "decisions",
                    "text": source_quote,
                    "reason": "用户明确给出长期仓库约定。",
                    "source_review_seqs": [1],
                    "auto_publish": True,
                    "source_quotes": [source_quote],
                }
            ]
        }
    )
    orchestrator = RequestOrchestrator(
        repository_memory=repository,
        review_model_client=client,
    )

    for index in range(1, 8):
        result = orchestrator.finalize_completed_run(
            run_id=f"run_{index}",
            user_input=(source_quote if index == 1 else f"临时讨论 {index}"),
            assistant_text="完成。",
        )
        assert result.review_status is None
        assert client.requests == []

    result = orchestrator.finalize_completed_run(
        run_id="run_8",
        user_input="临时讨论 8",
        assistant_text="完成。",
    )

    assert result.status == "review_recorded"
    assert result.review_status == "completed"
    assert result.reviewed_turns == 8
    assert result.candidate_count == 1
    assert result.auto_published_count == 1
    assert result.pending_user_turns == 0
    assert result.pending_candidates == 0
    assert len(client.requests) == 1
    entries = repository.topic_store.active_entries("decisions")
    assert [entry.summary for entry in entries] == [source_quote]
    candidate = repository.workflow_store.list_candidates()[0]
    assert candidate.status == "approved"
    assert candidate.auto_publish is True
    assert candidate.source_quotes == [source_quote]


def test_auto_review_keeps_uncertain_candidate_pending(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    client = FakeClient(
        {
            "candidates": [
                {
                    "topic": "decisions",
                    "text": "可能采用新的缓存策略。",
                    "reason": "该结论仍需要用户确认。",
                    "source_review_seqs": [1],
                    "auto_publish": False,
                }
            ]
        }
    )
    orchestrator = RequestOrchestrator(
        repository_memory=repository,
        review_model_client=client,
        auto_review_record_threshold=1,
    )

    result = orchestrator.finalize_completed_run(
        run_id="run_pending",
        user_input="分析缓存策略。",
        assistant_text="给出若干候选方案。",
    )

    assert result.review_status == "completed"
    assert result.auto_published_count == 0
    assert result.pending_candidates == 1
    assert repository.topic_store.active_entries("decisions") == []
    assert repository.workflow_store.list_candidates(status="pending")[0].text == (
        "可能采用新的缓存策略。"
    )


def test_auto_review_failure_does_not_fail_completed_run(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = RequestOrchestrator(
        repository_memory=repository,
        review_model_client=FailingClient(),
        auto_review_record_threshold=1,
    )

    result = orchestrator.finalize_completed_run(
        run_id="run_failure",
        user_input="以后只运行聚焦测试。",
        assistant_text="完成。",
    )

    assert result.status == "review_recorded"
    assert result.review_status == "failed"
    assert result.pending_user_turns == 1
    assert repository.workflow_store.list_candidates() == []


def test_pending_review_turns_are_bounded(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = RequestOrchestrator(repository_memory=repository)

    for index in range(MAX_PENDING_REVIEWS + 5):
        orchestrator.finalize_completed_run(
            run_id=f"run_{index}",
            user_input=f"turn {index}",
            assistant_text="完成。",
        )

    pending = repository.workflow_store.pending_reviews()
    assert len(pending) == MAX_PENDING_REVIEWS
    assert pending[0].source_run_id == "run_5"
    events = repository.event_store.list()
    assert any(event["event"] == "memory_review_record_evicted" for event in events)


def test_review_record_keeps_only_bounded_verification_evidence(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = RequestOrchestrator(repository_memory=repository)
    observations = [
        ContextObservation(
            tool_call_id="failed",
            tool_name="run_command",
            content="full failure output",
            output_preview="assertion failed",
            token_estimate=10,
            summary="Cache invalidation assertion failed.",
            metadata={
                "argv": ["pytest", "tests/test_cache.py", "-q"],
                "returncode": 1,
            },
        ),
        ContextObservation(
            tool_call_id="write",
            tool_name="apply_patch",
            content="patch applied",
            output_preview="patch applied",
            token_estimate=4,
        ),
    ]

    orchestrator.finalize_completed_run(
        run_id="run_evidence",
        user_input="修复缓存测试。",
        assistant_text="已修复并通过聚焦测试。",
        observations=observations,
        modified_files=["cache.py"],
        verification=VerificationState(
            status="passed",
            command="pytest tests/test_cache.py -q",
            returncode=0,
        ),
    )

    record = repository.workflow_store.pending_reviews()[0]
    assert record.verification is not None
    assert record.verification.status == "passed"
    assert record.resolved_failure is not None
    assert record.resolved_failure.failed_command == "pytest tests/test_cache.py -q"
    assert record.resolved_failure.modified_files == ["cache.py"]
    serialized = record.model_dump_json()
    assert "full failure output" not in serialized
