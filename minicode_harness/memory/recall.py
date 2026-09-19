"""Deterministic Memory Index injection for one Run snapshot."""

from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any

from .snapshot import MemorySnapshotStore


class MemorySnapshotReader:
    """Read and search only the immutable Memory view frozen for one Run."""

    def __init__(self, snapshot_store: MemorySnapshotStore) -> None:
        self.snapshot_store = snapshot_store

    def read(
        self,
        target: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        normalized = _normalize_snapshot_target(target)
        snapshot = self.snapshot_store.load()
        if snapshot is None:
            raise FileNotFoundError("Memory Run snapshot is missing.")

        if normalized == "MEMORY.md":
            content = self.snapshot_store.read_memory()
        elif normalized == "memory_summary.md":
            content = self.snapshot_store.read_summary()
        elif normalized.startswith("rollout_summaries/"):
            filename = normalized.removeprefix("rollout_summaries/")
            content = self.snapshot_store.read_rollout_summary(filename)
        else:
            raise FileNotFoundError(
                f"Memory resource is not present in the Run snapshot: {normalized}"
            )
        return _line_range(
            normalized,
            content,
            start_line=start_line,
            end_line=end_line,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        use_regex: bool = False,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        if not query:
            raise ValueError("Memory search query must not be empty.")
        if limit < 1 or limit > 500:
            raise ValueError("Memory search limit must be between 1 and 500.")
        snapshot = self.snapshot_store.load()
        if snapshot is None:
            raise FileNotFoundError("Memory Run snapshot is missing.")

        resources = [
            ("MEMORY.md", self.snapshot_store.read_memory()),
            *[
                (
                    f"rollout_summaries/{filename}",
                    self.snapshot_store.read_rollout_summary(filename),
                )
                for filename in snapshot.rollout_summary_files
            ],
        ]
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query, flags) if use_regex else None
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        for path, content in resources:
            for line_number, line in enumerate(content.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                matched = (
                    bool(pattern.search(line))
                    if pattern is not None
                    else needle in haystack
                )
                if not matched:
                    continue
                matches.append(
                    {"path": path, "line": line_number, "text": line}
                )
                if len(matches) >= limit:
                    return {
                        "query": query,
                        "matches": matches,
                        "truncated": True,
                    }
        return {
            "query": query,
            "matches": matches,
            "truncated": False,
        }


def _normalize_snapshot_target(target: str) -> str:
    normalized = target.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"Invalid Memory snapshot path: {target}")
    return path.as_posix()


def _line_range(
    path: str,
    content: str,
    *,
    start_line: int | None,
    end_line: int | None,
) -> dict[str, Any]:
    lines = content.splitlines()
    total_lines = len(lines)
    start = start_line if start_line is not None else 1
    if start < 1:
        raise ValueError("start_line must be at least 1.")
    if end_line is not None and end_line < start:
        raise ValueError("end_line must be greater than or equal to start_line.")
    if total_lines == 0:
        return {
            "path": path,
            "content": "",
            "start_line": 1,
            "end_line": 0,
            "total_lines": 0,
            "returned_lines": 0,
        }
    if start > total_lines:
        raise ValueError("start_line must not exceed the number of lines.")
    end = min(end_line if end_line is not None else total_lines, total_lines)
    return {
        "path": path,
        "content": "\n".join(lines[start - 1 : end]),
        "start_line": start,
        "end_line": end,
        "total_lines": total_lines,
        "returned_lines": end - start + 1,
    }
