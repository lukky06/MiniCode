"""Low-frequency one-shot review that proposes approval candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from minicode_harness.models import ModelClient, ModelRequest
from minicode_harness.trace import TraceWriter

from .repository_memory import RepositoryMemoryStore
from .types import MemoryTopicName, TOPIC_NAMES, contains_sensitive_content
from .workflow import MemoryApprovalCandidate, MemoryCandidateEvidence

MAX_REVIEW_CANDIDATES = 3
MAX_REVIEW_INPUT_CHARS = 64_000
MAX_ACTIVE_MEMORY_CATALOG_CHARS = 12_000


class MemoryCandidateProposal(BaseModel):
    topic: MemoryTopicName
    text: str = Field(min_length=1, max_length=600)
    reason: str = Field(min_length=1, max_length=800)
    source_review_seqs: list[int] = Field(min_length=1, max_length=8)
    auto_publish: bool = False
    source_quotes: list[str] = Field(default_factory=list, max_length=8)
    reviewed_entry_ids: list[str] = Field(default_factory=list, max_length=32)
    conflicts_with_entry_ids: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("text", "reason")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("source_review_seqs")
    @classmethod
    def normalize_sources(cls, value: list[int]) -> list[int]:
        normalized = sorted(set(value))
        if any(seq < 1 for seq in normalized):
            raise ValueError("source_review_seqs must contain positive integers.")
        return normalized

    @field_validator("reviewed_entry_ids", "conflicts_with_entry_ids")
    @classmethod
    def normalize_entry_ids(cls, value: list[str]) -> list[str]:
        return sorted({item.strip() for item in value if item.strip()})

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


class MemoryReviewCandidateProposal(BaseModel):
    candidates: list[MemoryCandidateProposal] = Field(
        default_factory=list,
        max_length=MAX_REVIEW_CANDIDATES,
    )


class MemoryManualReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryManualReviewResult:
    status: str
    reviewed_seqs: list[int] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    pending_reviews: int = 0
    pending_candidates: int = 0


class MemoryManualReviewService:
    """Run exactly one sidecar model call for the oldest 1-8 review records."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        store: RepositoryMemoryStore,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.model_client = model_client
        self.store = store
        self.trace_writer = trace_writer

    def review(self, *, limit: int = 8) -> MemoryManualReviewResult:
        capacity = self.store.workflow_store.candidate_capacity()
        if capacity == 0:
            return self._result(status="candidate_store_full")
        records = self.store.workflow_store.review_batch(limit=limit)
        if not records:
            return self._result(status="no_reviews")

        active_memory_catalog = _active_memory_catalog(self.store)
        payload_base = {
            "repository_id": self.store.repository_id,
            "allowed_topics": list(TOPIC_NAMES),
            "memory_index": self.store.index_store.ensure().content,
            "active_memory_catalog": active_memory_catalog,
            "max_candidates": min(MAX_REVIEW_CANDIDATES, capacity),
        }
        records = _bounded_review_records(records, payload_base=payload_base)
        payload = {
            **payload_base,
            "review_records": [item.model_dump(mode="json") for item in records],
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        self._event(
            "memory_manual_review_started",
            repository_id=self.store.repository_id,
            review_seqs=[item.seq for item in records],
        )
        try:
            response = self.model_client.call_request(
                ModelRequest(
                    system=_system_prompt(),
                    messages=[{"role": "user", "content": serialized}],
                    tools=[],
                    metadata={
                        "purpose": "memory_manual_review_v2",
                        "repository_id": self.store.repository_id,
                        "temperature": 0.0,
                    },
                )
            ).enforce_turn_contract()
            if response.tool_calls:
                raise MemoryManualReviewError("Memory reviewer returned tool calls.")
            if not response.final_text:
                raise MemoryManualReviewError(
                    response.invalid_reason or "Memory reviewer returned no JSON."
                )
            proposal = MemoryReviewCandidateProposal.model_validate(
                _parse_json_object(response.final_text)
            )
            if len(proposal.candidates) > payload["max_candidates"]:
                raise MemoryManualReviewError(
                    "Memory reviewer exceeded the available candidate capacity."
                )
            allowed_seqs = {item.seq for item in records}
            _validate_sources(proposal, allowed_seqs=allowed_seqs)
            records_by_seq = {item.seq: item for item in records}
            _validate_auto_publish_evidence(
                proposal,
                records_by_seq=records_by_seq,
            )
            catalog_by_topic = {
                item["topic"]: item for item in active_memory_catalog
            }
            _validate_entry_references(
                proposal,
                catalog_by_topic=catalog_by_topic,
            )
            candidates = [
                MemoryApprovalCandidate.create(
                    topic=item.topic,
                    text=item.text,
                    reason=item.reason,
                    source_review_seqs=item.source_review_seqs,
                    evidence=[
                        MemoryCandidateEvidence.from_review_record(records_by_seq[seq])
                        for seq in item.source_review_seqs
                    ],
                    auto_publish=_auto_publish_allowed_by_catalog(
                        item,
                        catalog_by_topic=catalog_by_topic,
                    ),
                    source_quotes=item.source_quotes,
                    reviewed_entry_ids=item.reviewed_entry_ids,
                    conflicts_with_entry_ids=item.conflicts_with_entry_ids,
                )
                for item in proposal.candidates
            ]
            reviewed_seqs = [item.seq for item in records]
            self.store.workflow_store.save_review_result(
                reviewed_seqs=reviewed_seqs,
                candidates=candidates,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._event(
                "memory_manual_review_failed",
                repository_id=self.store.repository_id,
                review_seqs=[item.seq for item in records],
                reason=reason,
            )
            if isinstance(exc, MemoryManualReviewError):
                raise
            raise MemoryManualReviewError(reason) from exc

        status = "completed" if candidates else "reviewed_no_candidates"
        self._event(
            "memory_manual_review_completed",
            repository_id=self.store.repository_id,
            review_seqs=reviewed_seqs,
            candidate_ids=[item.candidate_id for item in candidates],
        )
        return self._result(
            status=status,
            reviewed_seqs=reviewed_seqs,
            candidate_ids=[item.candidate_id for item in candidates],
        )

    def _result(
        self,
        *,
        status: str,
        reviewed_seqs: list[int] | None = None,
        candidate_ids: list[str] | None = None,
    ) -> MemoryManualReviewResult:
        return MemoryManualReviewResult(
            status=status,
            reviewed_seqs=list(reviewed_seqs or []),
            candidate_ids=list(candidate_ids or []),
            pending_reviews=len(self.store.workflow_store.pending_reviews()),
            pending_candidates=len(
                self.store.workflow_store.list_candidates(status="pending")
            ),
        )

    def _event(self, event: str, **payload: object) -> None:
        self.store.event_store.append(event, **payload)
        if self.trace_writer is not None:
            self.trace_writer.write_event(event, **payload)


def _bounded_review_records(
    records: list[Any],
    *,
    payload_base: dict[str, Any],
) -> list[Any]:
    selected: list[Any] = []
    for record in records:
        candidate = [*selected, record]
        payload = {
            **payload_base,
            "review_records": [item.model_dump(mode="json") for item in candidate],
        }
        size = len(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        if size > MAX_REVIEW_INPUT_CHARS:
            if selected:
                break
            raise MemoryManualReviewError(
                "One memory review record exceeds the bounded input budget."
            )
        selected.append(record)
    return selected


def _active_memory_catalog(
    store: RepositoryMemoryStore,
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    used_chars = 0
    for topic in TOPIC_NAMES:
        rendered_entries: list[dict[str, str]] = []
        entries = store.topic_store.active_entries(topic)
        complete = True
        for entry in entries:
            rendered = {
                "entry_id": entry.entry_id,
                "summary": entry.summary,
            }
            cost = len(
                json.dumps(
                    rendered,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if used_chars + cost > MAX_ACTIVE_MEMORY_CATALOG_CHARS:
                complete = False
                break
            rendered_entries.append(rendered)
            used_chars += cost
        catalog.append(
            {
                "topic": topic,
                "complete": complete and len(rendered_entries) == len(entries),
                "entries": rendered_entries,
            }
        )
    return catalog


def _validate_sources(
    proposal: MemoryReviewCandidateProposal,
    *,
    allowed_seqs: set[int],
) -> None:
    referenced: list[int] = [
        seq
        for candidate in proposal.candidates
        for seq in candidate.source_review_seqs
    ]
    unknown = sorted(set(referenced) - allowed_seqs)
    if unknown:
        raise MemoryManualReviewError(
            "Candidate source_review_seqs are not in the review batch: "
            + ", ".join(str(seq) for seq in unknown)
        )
    duplicate_sources = sorted(
        {seq for seq in referenced if referenced.count(seq) > 1}
    )
    if duplicate_sources:
        raise MemoryManualReviewError(
            "A review record may support at most one candidate: "
            + ", ".join(str(seq) for seq in duplicate_sources)
        )


def _validate_entry_references(
    proposal: MemoryReviewCandidateProposal,
    *,
    catalog_by_topic: dict[str, dict[str, Any]],
) -> None:
    for candidate in proposal.candidates:
        known_entry_ids = {
            entry["entry_id"]
            for entry in catalog_by_topic[candidate.topic]["entries"]
        }
        referenced = {
            *candidate.reviewed_entry_ids,
            *candidate.conflicts_with_entry_ids,
        }
        unknown = sorted(referenced - known_entry_ids)
        if unknown:
            raise MemoryManualReviewError(
                "Candidate references entries outside its selected Topic catalog: "
                + ", ".join(unknown)
            )
        if not set(candidate.conflicts_with_entry_ids).issubset(
            candidate.reviewed_entry_ids
        ):
            raise MemoryManualReviewError(
                "Candidate conflicts must be a subset of reviewed memory entries."
            )


def _auto_publish_allowed_by_catalog(
    candidate: MemoryCandidateProposal,
    *,
    catalog_by_topic: dict[str, dict[str, Any]],
) -> bool:
    catalog = catalog_by_topic[candidate.topic]
    active_ids = {
        entry["entry_id"] for entry in catalog["entries"]
    }
    return (
        candidate.auto_publish
        and bool(catalog["complete"])
        and set(candidate.reviewed_entry_ids) == active_ids
        and not candidate.conflicts_with_entry_ids
    )


def _validate_auto_publish_evidence(
    proposal: MemoryReviewCandidateProposal,
    *,
    records_by_seq: dict[int, Any],
) -> None:
    for candidate in proposal.candidates:
        if not candidate.source_quotes:
            continue
        source_texts = [
            " ".join(records_by_seq[seq].user_text.split())
            for seq in candidate.source_review_seqs
        ]
        for quote in candidate.source_quotes:
            if not any(quote in source_text for source_text in source_texts):
                raise MemoryManualReviewError(
                    "Candidate source quote is not present in its referenced user text."
                )


def _system_prompt() -> str:
    schema = json.dumps(
        MemoryReviewCandidateProposal.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Review completed repository work for durable cross-session memory. Return one JSON "
        f"object matching this schema and nothing else: {schema}. Propose zero to three new ADD "
        "candidates. Use only supplied review records and Topic names. Save explicit user "
        "preferences, verified build/test procedures, evidence-backed debugging lessons, confirmed "
        "decisions with rationale, and stable environment facts. Skip task progress, source facts, "
        "diffs, raw output, temporary blockers, secrets, unverified commands, and anything readily "
        "reconstructible from the repository. Do not update, merge, deactivate, or quote sensitive "
        "content. Each review record may support at most one candidate. Set auto_publish=true only "
        "for an unambiguous new durable rule backed by exact source quotes from referenced user "
        "text, a passed verification command copied into the candidate text, or a complete resolved "
        "failure chain. active_memory_catalog contains bounded existing entries and a complete flag "
        "per Topic. For each Candidate, reviewed_entry_ids must list every supplied existing entry "
        "examined in that Topic, and conflicts_with_entry_ids must identify any contradiction, "
        "replacement, or incompatible instruction. Never set auto_publish=true when the Topic "
        "catalog is incomplete, an existing entry was not reviewed, or any conflict exists. When "
        "source_quotes are used, candidate text must equal those quotes joined with one space and "
        "contain no generated claim. A failure-chain candidate must include the failed command, "
        "passed command, and at least one modified file. Otherwise leave auto_publish=false. "
        "source_quotes must be exact bounded excerpts from referenced user text."
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].lstrip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise MemoryManualReviewError(f"Invalid memory review JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise MemoryManualReviewError("Memory review response must be a JSON object.")
    return value
