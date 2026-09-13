"""Repository Memory lifecycle around one completed MiniCode Run.

The active runtime injects one bounded Repository Memory Index before a Run,
appends one bounded Review Record after completion, and may perform one
low-frequency sidecar review when enough records have accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass

from minicode_harness.context import ContextObservation, VerificationState
from minicode_harness.memory import (
    MemoryManualReviewService,
    MemoryResolvedFailure,
    MemoryVerificationEvidence,
    RepositoryMemoryCandidateService,
    RepositoryMemoryIndexService,
    RepositoryMemorySnapshotSource,
    RepositoryMemoryStore,
)
from minicode_harness.models import ModelClient
from minicode_harness.trace import TraceWriter


AUTO_REVIEW_RECORD_THRESHOLD = 8
AUTO_REVIEW_CHAR_THRESHOLD = 32_000


@dataclass(frozen=True)
class MemoryFinalizationResult:
    status: str
    pending_user_turns: int
    review_status: str | None = None
    reviewed_turns: int = 0
    candidate_count: int = 0
    auto_published_count: int = 0
    pending_candidates: int = 0


class RequestOrchestrator:
    """Own the active Repository Memory pre/post-Run lifecycle."""

    def __init__(
        self,
        *,
        repository_memory: RepositoryMemoryStore,
        trace_writer: TraceWriter | None = None,
        review_model_client: ModelClient | None = None,
        auto_review_record_threshold: int = AUTO_REVIEW_RECORD_THRESHOLD,
        auto_review_char_threshold: int = AUTO_REVIEW_CHAR_THRESHOLD,
    ) -> None:
        self.repository_memory = repository_memory
        self.trace_writer = trace_writer
        self.review_model_client = review_model_client
        self.auto_review_record_threshold = max(1, int(auto_review_record_threshold))
        self.auto_review_char_threshold = max(1, int(auto_review_char_threshold))
        self.index_service = RepositoryMemoryIndexService(trace_writer=trace_writer)

    def recall_context(self) -> str:
        """Render the bounded immutable Memory Index selected for this Run."""

        return self.index_service.render(store=self.repository_memory)

    def recall_snapshot(self) -> RepositoryMemorySnapshotSource:
        source = self.repository_memory.capture_snapshot_source()
        self.index_service.record(
            store=self.repository_memory,
            rendered=source.rendered_index,
            registered_topics=list(source.topic_payloads),
        )
        return source

    def finalize_completed_run(
        self,
        *,
        run_id: str,
        user_input: str,
        assistant_text: str,
        observations: list[ContextObservation] | None = None,
        modified_files: list[str] | None = None,
        verification: VerificationState | None = None,
    ) -> MemoryFinalizationResult:
        """Append one bounded record and run one non-fatal low-frequency review."""

        review_append = self.repository_memory.workflow_store.append_review_record(
            source_run_id=run_id,
            user_text=user_input,
            assistant_text=assistant_text,
            verification=_verification_evidence(verification),
            resolved_failure=_resolved_failure_evidence(
                observations=list(observations or []),
                modified_files=list(modified_files or []),
                verification=verification,
            ),
        )
        review_status: str | None = None
        reviewed_turns = 0
        candidate_count = 0
        auto_published_count = 0
        if review_append.status == "saved":
            try:
                (
                    review_status,
                    reviewed_turns,
                    candidate_count,
                    auto_published_count,
                ) = self.review_pending()
            except Exception as exc:
                review_status = "failed"
                if self.trace_writer is not None:
                    self.trace_writer.write_event(
                        "memory_auto_review_failed",
                        source_run_id=run_id,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
        pending_review_turns = len(
            self.repository_memory.workflow_store.pending_reviews()
        )
        pending_candidates = len(
            self.repository_memory.workflow_store.list_candidates(status="pending")
        )
        if self.trace_writer is not None:
            self.trace_writer.write_event(
                "memory_review_recorded",
                source_run_id=run_id,
                review_append_status=review_append.status,
                pending_user_turns=pending_review_turns,
                review_status=review_status,
                reviewed_turns=reviewed_turns,
                candidate_count=candidate_count,
                auto_published_count=auto_published_count,
                pending_candidates=pending_candidates,
            )
        return MemoryFinalizationResult(
            status=(
                "review_recorded"
                if review_append.status == "saved"
                else "idempotent_noop"
            ),
            pending_user_turns=pending_review_turns,
            review_status=review_status,
            reviewed_turns=reviewed_turns,
            candidate_count=candidate_count,
            auto_published_count=auto_published_count,
            pending_candidates=pending_candidates,
        )

    def review_pending(
        self,
        *,
        force: bool = False,
    ) -> tuple[str | None, int, int, int]:
        if self.review_model_client is None:
            return None, 0, 0, 0
        pending = self.repository_memory.workflow_store.pending_reviews()
        if not pending:
            return None, 0, 0, 0
        pending_chars = sum(
            len(item.user_text) + len(item.assistant_text)
            for item in pending
        )
        if (
            not force
            and len(pending) < self.auto_review_record_threshold
            and pending_chars < self.auto_review_char_threshold
        ):
            return None, 0, 0, 0

        reviewed = MemoryManualReviewService(
            self.review_model_client,
            store=self.repository_memory,
            trace_writer=self.trace_writer,
        ).review(limit=min(8, len(pending)))
        candidate_service = RepositoryMemoryCandidateService(self.repository_memory)
        auto_published = 0
        for candidate_id in reviewed.candidate_ids:
            try:
                if candidate_service.auto_approve(candidate_id) is not None:
                    auto_published += 1
            except Exception as exc:
                if self.trace_writer is not None:
                    self.trace_writer.write_event(
                        "memory_auto_publish_failed",
                        candidate_id=candidate_id,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
        if self.trace_writer is not None:
            self.trace_writer.write_event(
                "memory_auto_review_completed",
                review_status=reviewed.status,
                reviewed_turns=len(reviewed.reviewed_seqs),
                candidate_count=len(reviewed.candidate_ids),
                auto_published_count=auto_published,
                pending_reviews=reviewed.pending_reviews,
                pending_candidates=len(
                    self.repository_memory.workflow_store.list_candidates(
                        status="pending"
                    )
                ),
            )
        return (
            reviewed.status,
            len(reviewed.reviewed_seqs),
            len(reviewed.candidate_ids),
            auto_published,
        )


def _verification_evidence(
    verification: VerificationState | None,
) -> MemoryVerificationEvidence | None:
    if verification is None:
        return None
    return MemoryVerificationEvidence(
        status=verification.status,
        command=verification.command,
        returncode=verification.returncode,
        reason=verification.reason,
    )


def _resolved_failure_evidence(
    *,
    observations: list[ContextObservation],
    modified_files: list[str],
    verification: VerificationState | None,
) -> MemoryResolvedFailure | None:
    if (
        verification is None
        or verification.status != "passed"
        or not verification.command
        or not modified_files
    ):
        return None

    write_tools = {"edit", "apply_patch", "write"}
    failed_observation: ContextObservation | None = None
    for index, observation in enumerate(observations):
        if observation.tool_name != "run_command":
            continue
        if observation.metadata.get("returncode") in (None, 0):
            continue
        if not any(
            later.tool_name in write_tools for later in observations[index + 1 :]
        ):
            continue
        failed_observation = observation

    if failed_observation is None:
        return None
    command = _observation_command(failed_observation)
    if not command:
        return None
    summary = (
        failed_observation.summary
        or failed_observation.output_preview
        or "Verification command failed before the successful repair."
    )
    try:
        return MemoryResolvedFailure(
            failed_command=command,
            failure_summary=summary[:1200],
            modified_files=sorted(set(modified_files))[:32],
            passed_command=verification.command,
        )
    except ValueError:
        return None


def _observation_command(observation: ContextObservation) -> str:
    command = observation.metadata.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()
    argv = observation.metadata.get("argv")
    if isinstance(argv, list):
        return " ".join(str(item) for item in argv if str(item).strip())
    return ""
