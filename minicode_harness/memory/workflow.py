"""Bounded atomic workflow state for repository memory review and approval."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .events import RepositoryMemoryEventStore
from .types import MemoryTopicName, contains_sensitive_content, utc_now

MAX_PENDING_REVIEWS = 64
MAX_PENDING_CANDIDATES = 64
MAX_RESOLVED_CANDIDATES = 64
MAX_WORKFLOW_BYTES = 1024 * 1024
MAX_REVIEW_TEXT_CHARS = 4 * 1024
MAX_CANDIDATE_TEXT_CHARS = 600


class MemoryVerificationEvidence(BaseModel):
    status: Literal["not_run", "passed", "failed", "rolled_back"]
    command: str | None = Field(default=None, max_length=2000)
    returncode: int | None = None
    reason: str | None = Field(default=None, max_length=1200)


class MemoryResolvedFailure(BaseModel):
    failed_command: str = Field(min_length=1, max_length=2000)
    failure_summary: str = Field(min_length=1, max_length=1200)
    modified_files: list[str] = Field(default_factory=list, max_length=32)
    passed_command: str = Field(min_length=1, max_length=2000)


class MemoryReviewRecord(BaseModel):
    seq: int = Field(ge=1)
    source_run_id: str = Field(min_length=1, max_length=160)
    user_text: str = Field(min_length=1, max_length=MAX_REVIEW_TEXT_CHARS)
    assistant_text: str = Field(default="", max_length=MAX_REVIEW_TEXT_CHARS)
    verification: MemoryVerificationEvidence | None = None
    resolved_failure: MemoryResolvedFailure | None = None
    created_at: str = Field(default_factory=utc_now)

    @field_validator("user_text", "assistant_text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if contains_sensitive_content(normalized):
            raise ValueError("Memory review record contains sensitive content.")
        return normalized


class MemoryCandidateEvidence(BaseModel):
    review_seq: int = Field(ge=1)
    source_run_id: str = Field(min_length=1, max_length=160)
    verification: MemoryVerificationEvidence | None = None
    resolved_failure: MemoryResolvedFailure | None = None

    @classmethod
    def from_review_record(cls, record: MemoryReviewRecord) -> "MemoryCandidateEvidence":
        return cls(
            review_seq=record.seq,
            source_run_id=record.source_run_id,
            verification=record.verification,
            resolved_failure=record.resolved_failure,
        )


MemoryCandidateStatus = Literal["pending", "approved", "rejected"]


class MemoryApprovalCandidate(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=160)
    topic: MemoryTopicName
    text: str = Field(min_length=1, max_length=MAX_CANDIDATE_TEXT_CHARS)
    reason: str = Field(min_length=1, max_length=800)
    source_review_seqs: list[int] = Field(min_length=1, max_length=8)
    evidence: list[MemoryCandidateEvidence] = Field(default_factory=list, max_length=8)
    auto_publish: bool = False
    source_quotes: list[str] = Field(default_factory=list, max_length=8)
    reviewed_entry_ids: list[str] = Field(default_factory=list, max_length=32)
    conflicts_with_entry_ids: list[str] = Field(default_factory=list, max_length=32)
    status: MemoryCandidateStatus = "pending"
    created_at: str = Field(default_factory=utc_now)
    resolved_at: str | None = None

    @field_validator("text", "reason")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if contains_sensitive_content(normalized):
            raise ValueError("Memory approval candidate contains sensitive content.")
        return normalized

    @field_validator("source_review_seqs")
    @classmethod
    def normalize_sources(cls, value: list[int]) -> list[int]:
        normalized = sorted(set(value))
        if any(seq < 1 for seq in normalized):
            raise ValueError("source_review_seqs must contain positive integers.")
        return normalized

    @field_validator("source_quotes")
    @classmethod
    def normalize_source_quotes(cls, value: list[str]) -> list[str]:
        normalized_quotes: list[str] = []
        for item in value:
            normalized = " ".join(item.split())
            if not normalized:
                continue
            if contains_sensitive_content(normalized):
                raise ValueError("Memory candidate source quote contains sensitive content.")
            normalized_quotes.append(normalized)
        return normalized_quotes

    @field_validator("reviewed_entry_ids", "conflicts_with_entry_ids")
    @classmethod
    def normalize_entry_ids(cls, value: list[str]) -> list[str]:
        return sorted({item.strip() for item in value if item.strip()})

    @model_validator(mode="after")
    def validate_evidence_sources(self) -> "MemoryApprovalCandidate":
        if self.evidence:
            evidence_seqs = sorted({item.review_seq for item in self.evidence})
            if evidence_seqs != self.source_review_seqs:
                raise ValueError(
                    "Candidate evidence must cover exactly the source review sequences."
                )
        unknown_conflicts = set(self.conflicts_with_entry_ids) - set(
            self.reviewed_entry_ids
        )
        if unknown_conflicts:
            raise ValueError(
                "Candidate conflicts must be a subset of reviewed memory entries."
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        topic: MemoryTopicName,
        text: str,
        reason: str,
        source_review_seqs: list[int],
        evidence: list[MemoryCandidateEvidence] | None = None,
        auto_publish: bool = False,
        source_quotes: list[str] | None = None,
        reviewed_entry_ids: list[str] | None = None,
        conflicts_with_entry_ids: list[str] | None = None,
    ) -> "MemoryApprovalCandidate":
        normalized_text = " ".join(text.split())
        normalized_sources = sorted(set(source_review_seqs))
        payload = {
            "topic": topic,
            "text": normalized_text,
            "source_review_seqs": normalized_sources,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        return cls(
            candidate_id=f"candidate_{digest}",
            topic=topic,
            text=normalized_text,
            reason=reason,
            source_review_seqs=normalized_sources,
            evidence=list(evidence or []),
            auto_publish=auto_publish,
            source_quotes=list(source_quotes or []),
            reviewed_entry_ids=list(reviewed_entry_ids or []),
            conflicts_with_entry_ids=list(conflicts_with_entry_ids or []),
        )


class MemoryWorkflowSnapshot(BaseModel):
    version: Literal[1] = 1
    next_review_seq: int = Field(default=1, ge=1)
    recent_run_ids: list[str] = Field(default_factory=list, max_length=128)
    pending_reviews: list[MemoryReviewRecord] = Field(default_factory=list)
    candidates: list[MemoryApprovalCandidate] = Field(default_factory=list)


@dataclass(frozen=True)
class MemoryReviewRecordAppendResult:
    status: str
    record: MemoryReviewRecord | None = None
    evicted_review_seqs: tuple[int, ...] = ()


class RepositoryMemoryWorkflowStore:
    """Own all transient memory workflow state in one bounded atomic file."""

    def __init__(
        self,
        memory_dir: Path | str,
        *,
        event_store: RepositoryMemoryEventStore | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.path = self.memory_dir / "workflow.json"
        self.event_store = event_store or RepositoryMemoryEventStore(self.memory_dir)

    def snapshot(self) -> MemoryWorkflowSnapshot:
        if not self.path.is_file():
            return MemoryWorkflowSnapshot()
        size = self.path.stat().st_size
        if size > MAX_WORKFLOW_BYTES:
            raise ValueError(
                f"Memory workflow exceeds {MAX_WORKFLOW_BYTES} bytes: {size}."
            )
        try:
            return MemoryWorkflowSnapshot.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ValueError(f"Invalid workflow.json: {exc}") from exc

    def append_review_record(
        self,
        *,
        source_run_id: str,
        user_text: str,
        assistant_text: str,
        verification: MemoryVerificationEvidence | None = None,
        resolved_failure: MemoryResolvedFailure | None = None,
    ) -> MemoryReviewRecordAppendResult:
        snapshot = self.snapshot()
        if source_run_id in snapshot.recent_run_ids:
            return MemoryReviewRecordAppendResult(status="already_saved")

        normalized_user = _bounded_text(user_text)
        normalized_assistant = _bounded_text(assistant_text, allow_empty=True)
        if not normalized_user:
            return MemoryReviewRecordAppendResult(status="skipped_empty")
        try:
            record = MemoryReviewRecord(
                seq=snapshot.next_review_seq,
                source_run_id=source_run_id,
                user_text=normalized_user,
                assistant_text=normalized_assistant,
                verification=verification,
                resolved_failure=resolved_failure,
            )
        except ValueError as exc:
            if "sensitive" not in str(exc).lower():
                raise
            self.event_store.append(
                "memory_review_record_skipped",
                source_run_id=source_run_id,
                reason="sensitive_content",
            )
            self._write(
                snapshot.model_copy(
                    update={
                        "recent_run_ids": _append_bounded(
                            snapshot.recent_run_ids,
                            source_run_id,
                            128,
                        )
                    }
                )
            )
            return MemoryReviewRecordAppendResult(status="skipped_sensitive")

        pending = [*snapshot.pending_reviews, record]
        evicted = pending[:-MAX_PENDING_REVIEWS]
        kept = pending[-MAX_PENDING_REVIEWS:]
        updated = snapshot.model_copy(
            update={
                "next_review_seq": record.seq + 1,
                "recent_run_ids": _append_bounded(
                    snapshot.recent_run_ids,
                    source_run_id,
                    128,
                ),
                "pending_reviews": kept,
            }
        )
        self._write(updated)
        self.event_store.append(
            "memory_review_record_appended",
            seq=record.seq,
            source_run_id=source_run_id,
        )
        if evicted:
            self.event_store.append(
                "memory_review_record_evicted",
                review_seqs=[item.seq for item in evicted],
                retained_count=len(kept),
            )
        return MemoryReviewRecordAppendResult(
            status="saved",
            record=record,
            evicted_review_seqs=tuple(item.seq for item in evicted),
        )

    def pending_reviews(self) -> list[MemoryReviewRecord]:
        return list(self.snapshot().pending_reviews)

    def review_batch(self, *, limit: int = 8) -> list[MemoryReviewRecord]:
        bounded_limit = min(8, max(1, int(limit)))
        return self.pending_reviews()[:bounded_limit]

    def candidate_capacity(self) -> int:
        pending = sum(
            1 for item in self.snapshot().candidates if item.status == "pending"
        )
        return max(0, MAX_PENDING_CANDIDATES - pending)

    def list_candidates(
        self,
        *,
        status: MemoryCandidateStatus | None = None,
    ) -> list[MemoryApprovalCandidate]:
        candidates = list(self.snapshot().candidates)
        if status is None:
            return candidates
        return [item for item in candidates if item.status == status]

    def get_candidate(self, candidate_id: str) -> MemoryApprovalCandidate | None:
        return next(
            (
                item
                for item in self.snapshot().candidates
                if item.candidate_id == candidate_id
            ),
            None,
        )

    def save_review_result(
        self,
        *,
        reviewed_seqs: list[int],
        candidates: list[MemoryApprovalCandidate],
    ) -> None:
        snapshot = self.snapshot()
        normalized_reviewed = sorted(set(reviewed_seqs))
        available_reviews = {item.seq for item in snapshot.pending_reviews}
        if not normalized_reviewed or not set(normalized_reviewed).issubset(
            available_reviews
        ):
            raise ValueError("Reviewed sequences must reference pending review records.")

        existing_ids = {item.candidate_id for item in snapshot.candidates}
        new_candidates = [
            item for item in candidates if item.candidate_id not in existing_ids
        ]
        pending_count = sum(
            1 for item in snapshot.candidates if item.status == "pending"
        )
        if pending_count + len(new_candidates) > MAX_PENDING_CANDIDATES:
            raise ValueError("Pending memory candidate capacity exceeded.")

        remaining_reviews = [
            item
            for item in snapshot.pending_reviews
            if item.seq not in set(normalized_reviewed)
        ]
        combined_candidates = _trim_candidates(
            [*snapshot.candidates, *new_candidates]
        )
        self._write(
            snapshot.model_copy(
                update={
                    "pending_reviews": remaining_reviews,
                    "candidates": combined_candidates,
                }
            )
        )
        self.event_store.append(
            "memory_review_completed",
            reviewed_seqs=normalized_reviewed,
            candidate_ids=[item.candidate_id for item in new_candidates],
        )

    def resolve_candidate(
        self,
        candidate_id: str,
        *,
        status: Literal["approved", "rejected"],
    ) -> MemoryApprovalCandidate:
        snapshot = self.snapshot()
        candidates = list(snapshot.candidates)
        for index, item in enumerate(candidates):
            if item.candidate_id != candidate_id:
                continue
            if item.status == status:
                return item
            updated = item.model_copy(
                update={"status": status, "resolved_at": utc_now()}
            )
            candidates[index] = updated
            self._write(
                snapshot.model_copy(update={"candidates": _trim_candidates(candidates)})
            )
            self.event_store.append(
                f"memory_candidate_{status}",
                candidate_id=candidate_id,
                topic=item.topic,
            )
            return updated
        raise KeyError(f"Memory candidate not found: {candidate_id}")

    def _write(self, snapshot: MemoryWorkflowSnapshot) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(
                snapshot.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        size = len(serialized.encode("utf-8"))
        if size > MAX_WORKFLOW_BYTES:
            raise ValueError(
                f"Memory workflow exceeds {MAX_WORKFLOW_BYTES} bytes: {size}."
            )
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(self.path)


def _bounded_text(value: str, *, allow_empty: bool = False) -> str:
    normalized = value.strip()
    if not normalized:
        return "" if allow_empty else normalized
    if len(normalized) <= MAX_REVIEW_TEXT_CHARS:
        return normalized
    return normalized[: MAX_REVIEW_TEXT_CHARS - 16].rstrip() + "\n[turn clipped]"


def _append_bounded(values: list[str], value: str, limit: int) -> list[str]:
    return [*values, value][-limit:]


def _trim_candidates(
    candidates: list[MemoryApprovalCandidate],
) -> list[MemoryApprovalCandidate]:
    pending = [item for item in candidates if item.status == "pending"]
    resolved = [item for item in candidates if item.status != "pending"]
    if len(pending) > MAX_PENDING_CANDIDATES:
        raise ValueError("Pending memory candidate capacity exceeded.")
    return [*pending, *resolved[-MAX_RESOLVED_CANDIDATES:]]
