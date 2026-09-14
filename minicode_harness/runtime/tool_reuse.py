"""Run-local read and search reuse tracking."""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
from threading import Lock
from typing import Any

from minicode_harness.context import ContextObservation, RunState
from minicode_harness.models import NormalizedToolCall
from minicode_harness.tools.semantics import (
    is_text_search,
    is_workspace_read,
    read_target,
    search_source,
)


_INCOMPLETE_SEARCH_REASONS = {"timeout", "scan_limit", "cancelled"}


@dataclass
class ReadCoverageEntry:
    path: str
    intervals: list[tuple[int, int]]
    total_lines: int | None = None
    source_tool_call_id: str | None = None


@dataclass(frozen=True)
class SearchReuseEntry:
    source_tool_call_id: str
    summary: str
    match_count: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class ReadReuseMatch:
    path: str
    requested: tuple[int, int]
    covered_by: tuple[int, int]
    total_lines: int | None
    source_tool_call_id: str | None


@dataclass(frozen=True)
class SearchReuseMatch:
    arguments: dict[str, Any]
    summary: str
    match_count: int
    files: tuple[str, ...]
    source_tool_call_id: str


class ToolReuseTracker:
    """Own read/search reuse state for one Run."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._read_coverage: dict[tuple[int, str], ReadCoverageEntry] = {}
        self._search_reuse: dict[tuple[Any, ...], SearchReuseEntry] = {}

    def lookup(
        self,
        tool_call: NormalizedToolCall,
        *,
        workspace_generation: int,
    ) -> ReadReuseMatch | SearchReuseMatch | None:
        if is_workspace_read(tool_call.name, tool_call.arguments):
            path = normalize_workspace_path(
                read_target(tool_call.name, tool_call.arguments)
            )
            if not path:
                return None
            with self._lock:
                entry = self._read_coverage.get((workspace_generation, path))
                if entry is None:
                    return None
                requested = read_request_range(
                    tool_call.arguments,
                    total_lines=entry.total_lines,
                )
                if requested is None:
                    return None
                covered_by = next(
                    (
                        interval
                        for interval in entry.intervals
                        if interval[0] <= requested[0] and interval[1] >= requested[1]
                    ),
                    None,
                )
                if covered_by is None:
                    return None
                return ReadReuseMatch(
                    path=entry.path,
                    requested=requested,
                    covered_by=covered_by,
                    total_lines=entry.total_lines,
                    source_tool_call_id=entry.source_tool_call_id,
                )

        if is_text_search(tool_call.name, tool_call.arguments):
            key = search_reuse_key(
                tool_call.arguments,
                workspace_generation=workspace_generation,
            )
            with self._lock:
                entry = self._search_reuse.get(key)
                if entry is None:
                    return None
                return SearchReuseMatch(
                    arguments=normalized_search_arguments(tool_call.arguments),
                    summary=entry.summary,
                    match_count=entry.match_count,
                    files=entry.files,
                    source_tool_call_id=entry.source_tool_call_id,
                )
        return None

    def overlap_detected(
        self,
        tool_call: NormalizedToolCall,
        *,
        workspace_generation: int,
    ) -> bool:
        if not is_workspace_read(tool_call.name, tool_call.arguments):
            return False
        path = normalize_workspace_path(read_target(tool_call.name, tool_call.arguments))
        if not path:
            return False
        with self._lock:
            entry = self._read_coverage.get((workspace_generation, path))
            if entry is None:
                return False
            requested = read_request_range(
                tool_call.arguments,
                total_lines=entry.total_lines,
            )
            if requested is None:
                return False
            for interval in entry.intervals:
                overlaps = max(interval[0], requested[0]) <= min(
                    interval[1], requested[1]
                )
                contains = (
                    interval[0] <= requested[0] and interval[1] >= requested[1]
                )
                if overlaps and not contains:
                    return True
        return False

    def record(
        self,
        tool_call: NormalizedToolCall,
        observation: ContextObservation,
        *,
        workspace_generation: int,
    ) -> None:
        if is_workspace_read(tool_call.name, tool_call.arguments):
            self._record_read(
                tool_call,
                observation,
                workspace_generation=workspace_generation,
            )
        elif is_text_search(tool_call.name, tool_call.arguments):
            self._record_search(
                tool_call,
                observation,
                workspace_generation=workspace_generation,
            )

    def restore(
        self,
        *,
        observations: list[ContextObservation],
        run_state: RunState,
        workspace_generation: int,
    ) -> None:
        with self._lock:
            self._read_coverage.clear()
            self._search_reuse.clear()
            self._restore_locked(
                observations=observations,
                run_state=run_state,
                workspace_generation=workspace_generation,
                visible_result_ids=None,
            )

    def synchronize(
        self,
        messages: list[dict[str, Any]],
        *,
        observations: list[ContextObservation],
        run_state: RunState,
        workspace_generation: int,
    ) -> None:
        visible_result_ids = {
            str(message.get("tool_call_id") or "")
            for message in messages
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        with self._lock:
            self._read_coverage.clear()
            self._search_reuse.clear()
            self._restore_locked(
                observations=observations,
                run_state=run_state,
                workspace_generation=workspace_generation,
                visible_result_ids=visible_result_ids,
            )

    def _record_read(
        self,
        tool_call: NormalizedToolCall,
        observation: ContextObservation,
        *,
        workspace_generation: int,
    ) -> None:
        status = str(observation.metadata.get("status") or "ok")
        if status != "ok":
            return
        path = normalize_workspace_path(
            observation.metadata.get("path")
            or read_target(tool_call.name, tool_call.arguments)
        )
        start = coerce_optional_int(observation.metadata.get("start_line"))
        end = coerce_optional_int(observation.metadata.get("end_line"))
        total_lines = coerce_optional_int(observation.metadata.get("total_lines"))
        if not path or start is None or end is None or end < start:
            return
        observation.metadata["workspace_generation"] = workspace_generation
        with self._lock:
            self._merge_read_coverage(
                path=path,
                start=start,
                end=end,
                total_lines=total_lines,
                source_tool_call_id=tool_call.id,
                workspace_generation=workspace_generation,
            )

    def _record_search(
        self,
        tool_call: NormalizedToolCall,
        observation: ContextObservation,
        *,
        workspace_generation: int,
    ) -> None:
        if not search_observation_reusable(observation):
            return
        arguments = normalized_search_arguments(tool_call.arguments)
        observation.metadata["search_arguments"] = arguments
        observation.metadata["workspace_generation"] = workspace_generation
        key = search_reuse_key(arguments, workspace_generation=workspace_generation)
        with self._lock:
            self._search_reuse[key] = SearchReuseEntry(
                source_tool_call_id=tool_call.id,
                summary=observation.summary,
                match_count=int(observation.metadata.get("match_count") or 0),
                files=tuple(
                    str(path) for path in observation.metadata.get("files") or []
                ),
            )

    def _restore_locked(
        self,
        *,
        observations: list[ContextObservation],
        run_state: RunState,
        workspace_generation: int,
        visible_result_ids: set[str] | None,
    ) -> None:
        for inspected in run_state.inspected_files:
            if (
                visible_result_ids is not None
                and inspected.last_tool_call_id not in visible_result_ids
            ):
                continue
            if int(inspected.workspace_generation) != workspace_generation:
                continue
            path = normalize_workspace_path(inspected.path)
            start = coerce_optional_int(inspected.line_start)
            end = coerce_optional_int(inspected.line_end)
            if not path or start is None or end is None or end < start:
                continue
            self._merge_read_coverage(
                path=path,
                start=start,
                end=end,
                total_lines=coerce_optional_int(inspected.total_lines),
                source_tool_call_id=inspected.last_tool_call_id,
                workspace_generation=workspace_generation,
            )

        for observation in observations:
            if (
                visible_result_ids is not None
                and observation.tool_call_id not in visible_result_ids
            ):
                continue
            generation = coerce_optional_int(
                observation.metadata.get("workspace_generation")
            )
            if generation != workspace_generation:
                continue
            status = str(observation.metadata.get("status") or "ok")
            if observation_is_workspace_read(observation) and status == "ok":
                path = normalize_workspace_path(observation.metadata.get("path"))
                start = coerce_optional_int(observation.metadata.get("start_line"))
                end = coerce_optional_int(observation.metadata.get("end_line"))
                if path and start is not None and end is not None and end >= start:
                    self._merge_read_coverage(
                        path=path,
                        start=start,
                        end=end,
                        total_lines=coerce_optional_int(
                            observation.metadata.get("total_lines")
                        ),
                        source_tool_call_id=observation.tool_call_id,
                        workspace_generation=generation,
                    )
            elif search_observation_reusable(observation):
                arguments = observation.metadata.get("search_arguments")
                if not isinstance(arguments, dict):
                    continue
                key = search_reuse_key(
                    arguments,
                    workspace_generation=generation,
                )
                self._search_reuse[key] = SearchReuseEntry(
                    source_tool_call_id=observation.tool_call_id,
                    summary=observation.summary,
                    match_count=int(observation.metadata.get("match_count") or 0),
                    files=tuple(
                        str(path) for path in observation.metadata.get("files") or []
                    ),
                )

    def _merge_read_coverage(
        self,
        *,
        path: str,
        start: int,
        end: int,
        total_lines: int | None,
        source_tool_call_id: str | None,
        workspace_generation: int,
    ) -> None:
        key = (workspace_generation, path)
        entry = self._read_coverage.get(key)
        if entry is None:
            entry = ReadCoverageEntry(path=path, intervals=[])
            self._read_coverage[key] = entry
        entry.intervals = merge_line_intervals([*entry.intervals, (start, end)])
        if total_lines is not None:
            entry.total_lines = total_lines
        if source_tool_call_id:
            entry.source_tool_call_id = source_tool_call_id


def normalize_workspace_path(value: Any) -> str:
    raw = str(value or ".").strip().replace("\\", "/")
    normalized = posixpath.normpath(raw)
    if normalized == ".":
        return "."
    return normalized.removeprefix("./")


def coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def observation_is_workspace_read(observation: ContextObservation) -> bool:
    return (
        observation.tool_name == "read"
        and observation.metadata.get("read_source") == "workspace"
    )


def observation_is_text_search(observation: ContextObservation) -> bool:
    return (
        observation.tool_name == "search"
        and observation.metadata.get("search_kind") == "text"
    )


def read_request_range(
    arguments: dict[str, Any],
    *,
    total_lines: int | None,
) -> tuple[int, int] | None:
    start = coerce_optional_int(arguments.get("start_line")) or 1
    explicit_end = coerce_optional_int(arguments.get("end_line"))
    end = explicit_end if explicit_end is not None else total_lines
    if start < 1 or end is None or end < start:
        return None
    return start, end


def merge_line_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def normalized_search_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    file_glob = arguments.get("file_glob")
    return {
        "source": search_source("search", arguments) or "workspace",
        "query": str(arguments.get("query") or ""),
        "path": normalize_workspace_path(arguments.get("path") or "."),
        "file_glob": (
            str(file_glob).strip().replace("\\", "/")
            if file_glob is not None
            else None
        ),
        "use_regex": bool(arguments.get("use_regex", False)),
        "case_sensitive": bool(arguments.get("case_sensitive", True)),
        "max_matches": int(
            arguments.get("limit", arguments.get("max_matches", 50))
        ),
        "max_depth": int(arguments.get("max_depth", 12)),
    }


def search_reuse_key(
    arguments: dict[str, Any],
    *,
    workspace_generation: int,
) -> tuple[Any, ...]:
    normalized = normalized_search_arguments(arguments)
    return (
        normalized["source"],
        normalized["query"],
        normalized["path"],
        normalized["file_glob"],
        normalized["use_regex"],
        normalized["case_sensitive"],
        normalized["max_matches"],
        normalized["max_depth"],
        workspace_generation,
    )


def search_observation_reusable(observation: ContextObservation) -> bool:
    if not observation_is_text_search(observation):
        return False
    status = str(observation.metadata.get("status") or "ok")
    reason = str(observation.metadata.get("truncation_reason") or "")
    return status == "ok" and reason not in _INCOMPLETE_SEARCH_REASONS
