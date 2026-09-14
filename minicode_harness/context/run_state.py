"""Deterministic updates for minimal checkpointed run state."""

from __future__ import annotations

from typing import Any

from minicode_harness.tools.semantics import is_artifact_read, is_workspace_read, read_target

from .limits import MAX_INSPECTED_FILES
from .types import ContextObservation, InspectedFile, RunState, VerificationState


def initialize_run_state(state: RunState | None = None) -> RunState:
    """Return an isolated run-state value for a new or resumed loop."""

    return state.model_copy(deep=True) if state is not None else RunState()


def mark_verification_not_run(state: RunState) -> None:
    """Reset verification after a workspace modification."""

    state.verification = VerificationState()


def mark_verification_passed(
    state: RunState,
    *,
    command: str,
    returncode: int = 0,
) -> None:
    """Record a successful verification command."""

    state.verification = VerificationState(
        status="passed",
        command=command or None,
        returncode=returncode,
    )


def mark_verification_failed(
    state: RunState,
    *,
    command: str,
    returncode: int,
    reason: str | None = None,
) -> None:
    """Record a failed verification command."""

    state.verification = VerificationState(
        status="failed",
        command=command or None,
        returncode=returncode,
        reason=reason,
    )


def mark_verification_rolled_back(
    state: RunState,
    *,
    reason: str,
) -> None:
    """Record that unfinished workspace changes were rolled back."""

    state.verification = VerificationState(
        status="rolled_back",
        reason=reason,
    )


def record_inspected_file(
    *,
    state: RunState,
    tool_name: str,
    tool_call_id: str,
    arguments: dict[str, Any],
    observation: ContextObservation,
    step: int,
    workspace_generation: int = 0,
) -> None:
    """Record the strongest freshness-scoped read coverage for one file."""

    status = str(observation.metadata.get("status", "ok"))
    if status != "ok":
        return
    if not (
        is_workspace_read(tool_name, arguments)
        or is_artifact_read(tool_name, arguments)
    ):
        return
    path = str(
        observation.metadata.get("path")
        or read_target(tool_name, arguments)
        or ""
    ).strip()
    if not path:
        return

    line_start = _optional_int(observation.metadata.get("start_line"))
    line_end = _optional_int(observation.metadata.get("end_line"))
    total_lines = _optional_int(observation.metadata.get("total_lines"))
    summary = observation.summary or _first_line(
        observation.output_preview or observation.content
    )
    current = InspectedFile(
        path=path,
        summary=summary,
        last_tool_call_id=tool_call_id,
        last_step=step,
        line_start=line_start,
        line_end=line_end,
        total_lines=total_lines,
        artifact_path=observation.artifact_path,
        content_status=_content_status(
            observation=observation,
            line_start=line_start,
            line_end=line_end,
            total_lines=total_lines,
        ),
        content_sha256=(
            str(observation.metadata.get("content_sha256"))
            if observation.metadata.get("content_sha256")
            else None
        ),
        workspace_generation=workspace_generation,
    )
    existing = next(
        (item for item in reversed(state.inspected_files) if item.path == path),
        None,
    )
    if existing is not None and existing.workspace_generation == workspace_generation:
        current = _strongest_coverage(existing, current)
    state.inspected_files = [item for item in state.inspected_files if item.path != path]
    state.inspected_files.append(current)
    del state.inspected_files[:-MAX_INSPECTED_FILES]


def _strongest_coverage(existing: InspectedFile, current: InspectedFile) -> InspectedFile:
    if (
        existing.content_sha256
        and current.content_sha256
        and existing.content_sha256 != current.content_sha256
    ):
        return current
    existing_full = _is_full(existing)
    current_full = _is_full(current)
    if existing_full and not current_full:
        return existing
    if current_full:
        return current
    if _range_contains(existing, current):
        return existing
    if _range_contains(current, existing):
        return current
    existing_span = _range_span(existing.line_start, existing.line_end)
    current_span = _range_span(current.line_start, current.line_end)
    if existing_span != current_span:
        return existing if existing_span > current_span else current
    return current if _content_rank(current.content_status) >= _content_rank(existing.content_status) else existing


def _is_full(item: InspectedFile) -> bool:
    return (
        item.line_start == 1
        and item.line_end is not None
        and item.total_lines is not None
        and item.line_end >= item.total_lines
        and item.content_status in {
            "full_content_available_in_context",
            "full_content_available_via_artifact",
        }
    )


def _range_contains(outer: InspectedFile, inner: InspectedFile) -> bool:
    if None in {outer.line_start, outer.line_end, inner.line_start, inner.line_end}:
        return False
    return int(outer.line_start) <= int(inner.line_start) and int(outer.line_end) >= int(inner.line_end)


def _range_span(start: int | None, end: int | None) -> int:
    if start is None or end is None or end < start:
        return 0
    return end - start + 1


def _content_rank(status: str) -> int:
    return {
        "summary_only": 0,
        "partial_content_available_in_context": 1,
        "full_content_available_via_artifact": 2,
        "full_content_available_in_context": 3,
    }.get(status, 0)


def _content_status(
    *,
    observation: ContextObservation,
    line_start: int | None,
    line_end: int | None,
    total_lines: int | None,
) -> str:
    full_range = (
        line_start == 1
        and total_lines is not None
        and line_end is not None
        and line_end >= total_lines
    )
    if observation.lossiness == "none":
        return "full_content_available_in_context" if full_range else "partial_content_available_in_context"
    if observation.artifact_path and full_range:
        return "full_content_available_via_artifact"
    return "summary_only"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        compact = " ".join(line.split())
        if compact:
            return compact
    return ""
