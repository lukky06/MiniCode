"""Deterministic approval and rejection of memory candidates."""

from __future__ import annotations

from pydantic import BaseModel

from .publishing import RepositoryMemoryPublisher
from .repository_memory import RepositoryMemoryStore
from .types import MemoryTopicName
from .workflow import MemoryApprovalCandidate


class MemoryCandidateDecisionResult(BaseModel):
    status: str
    candidate_id: str
    candidate_status: str
    topic: MemoryTopicName
    entry_id: str | None = None
    publish_status: str | None = None
    index_status: str | None = None
    topic_size_bytes: int | None = None
    topic_limit_bytes: int | None = None
    capacity_warning: str | None = None


class RepositoryMemoryCandidateService:
    def __init__(self, store: RepositoryMemoryStore) -> None:
        self.store = store
        self.publisher = RepositoryMemoryPublisher(store)

    def approve(self, candidate_id: str) -> MemoryCandidateDecisionResult:
        candidate = self._candidate(candidate_id)
        if candidate.status == "rejected":
            raise ValueError(
                "Rejected memory candidate cannot be approved; run memory review again."
            )

        entry_id = f"memory_{candidate.candidate_id}"
        published = self.publisher.publish(
            topic=candidate.topic,
            text=candidate.text,
            entry_id=entry_id,
            evidence_ids=[
                f"review:{seq}" for seq in candidate.source_review_seqs
            ],
        )
        if candidate.status == "approved":
            return MemoryCandidateDecisionResult(
                status="idempotent_noop",
                candidate_id=candidate.candidate_id,
                candidate_status="approved",
                topic=candidate.topic,
                entry_id=published.entry_id,
                publish_status=published.status,
                index_status=published.index_status,
                topic_size_bytes=published.topic_size_bytes,
                topic_limit_bytes=published.topic_limit_bytes,
                capacity_warning=published.capacity_warning,
            )

        resolved = self.store.workflow_store.resolve_candidate(
            candidate.candidate_id,
            status="approved",
        )
        return MemoryCandidateDecisionResult(
            status="applied",
            candidate_id=resolved.candidate_id,
            candidate_status=resolved.status,
            topic=resolved.topic,
            entry_id=published.entry_id,
            publish_status=published.status,
            index_status=published.index_status,
            topic_size_bytes=published.topic_size_bytes,
            topic_limit_bytes=published.topic_limit_bytes,
            capacity_warning=published.capacity_warning,
        )

    def auto_approve(
        self,
        candidate_id: str,
    ) -> MemoryCandidateDecisionResult | None:
        candidate = self._candidate(candidate_id)
        if candidate.status != "pending" or not candidate.auto_publish:
            return None
        current_entry_ids = {
            item.entry_id
            for item in self.store.topic_store.active_entries(candidate.topic)
        }
        if (
            set(candidate.reviewed_entry_ids) != current_entry_ids
            or candidate.conflicts_with_entry_ids
        ):
            return None
        if self.store.topic_store.capacity_status(candidate.topic).warning:
            return None
        if not _has_auto_publish_evidence(candidate):
            return None
        return self.approve(candidate_id)

    def reject(self, candidate_id: str) -> MemoryCandidateDecisionResult:
        candidate = self._candidate(candidate_id)
        if candidate.status == "approved":
            raise ValueError(
                "Approved memory candidate is active; use memory forget on its entry."
            )
        if candidate.status == "rejected":
            return MemoryCandidateDecisionResult(
                status="idempotent_noop",
                candidate_id=candidate.candidate_id,
                candidate_status="rejected",
                topic=candidate.topic,
            )
        resolved = self.store.workflow_store.resolve_candidate(
            candidate.candidate_id,
            status="rejected",
        )
        return MemoryCandidateDecisionResult(
            status="applied",
            candidate_id=resolved.candidate_id,
            candidate_status=resolved.status,
            topic=resolved.topic,
        )

    def _candidate(self, candidate_id: str):
        candidate = self.store.workflow_store.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"Memory candidate not found: {candidate_id}")
        return candidate


def _has_auto_publish_evidence(candidate: MemoryApprovalCandidate) -> bool:
    normalized_text = _normalized(candidate.text)
    if candidate.source_quotes:
        quoted_text = " ".join(candidate.source_quotes)
        return normalized_text == _normalized(quoted_text)
    if candidate.topic == "build-and-test":
        return any(
            evidence.verification is not None
            and evidence.verification.status == "passed"
            and evidence.verification.command
            and _normalized(evidence.verification.command) in normalized_text
            for evidence in candidate.evidence
        )
    if candidate.topic == "debugging":
        for evidence in candidate.evidence:
            failure = evidence.resolved_failure
            if failure is None or not failure.modified_files:
                continue
            required_anchors = [
                failure.failed_command,
                failure.passed_command,
                failure.modified_files[0],
            ]
            if all(
                _normalized(anchor) in normalized_text
                for anchor in required_anchors
            ):
                return True
    return False


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()
