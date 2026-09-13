"""Read-only workspace tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fnmatch
import hashlib
import os
import re
import subprocess
import time

from pydantic import BaseModel, Field

from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.workspace import WorkspaceAccessError, WorkspaceGuard


DEFAULT_MAX_MATCHES = 50
FILE_SEARCH_TIMEOUT_SECONDS = 5.0
FILE_SEARCH_SCAN_LIMIT = 100_000
TEXT_SEARCH_TIMEOUT_SECONDS = 5.0
TEXT_SEARCH_SCAN_LIMIT = 100_000
ESTIMATED_BYTES_PER_LINE = 80
LARGE_READ_FEEDBACK_LINE_THRESHOLD = 500
LARGE_READ_FEEDBACK_CHAR_THRESHOLD = 48_000
DEFAULT_SEARCH_IGNORED_DIRECTORIES = frozenset(
    {
        "node_modules",
        "target",
        "build",
        "dist",
        ".venv",
        "venv",
        "__pycache__",
        ".gradle",
        ".pytest_cache",
    }
)


class FileSearchEntry(BaseModel):
    """Lightweight cost metadata for one discovered file."""

    path: str
    size_bytes: int
    estimated_lines: int


class FindFilesResult(BaseModel):
    """Result returned by the unified find_files tool."""

    root: str
    pattern: str
    files: list[str] = Field(default_factory=list)
    file_details: list[FileSearchEntry] = Field(default_factory=list)
    truncated: bool = False
    truncation_reason: str | None = None
    scanned_entries: int = 0
    exists: bool = True
    is_directory: bool = True


@dataclass
class _FileSearchState:
    """Internal execution bounds shared by Git and filesystem discovery."""

    deadline: float
    scan_limit: int
    scanned_entries: int = 0
    truncation_reason: str | None = None
    cancellation_token: CancellationToken | None = None

    def admit_next(self) -> bool:
        if self.interrupted():
            return False
        if self.scanned_entries >= self.scan_limit:
            self.stop("scan_limit")
            return False
        self.scanned_entries += 1
        return True

    def interrupted(self) -> bool:
        if self.cancellation_token is not None and self.cancellation_token.is_cancelled:
            self.stop("cancelled")
            return True
        if time.monotonic() >= self.deadline:
            self.stop("timeout")
            return True
        return False

    def stop(self, reason: str) -> None:
        if self.truncation_reason is None:
            self.truncation_reason = reason

class TextRange(BaseModel):
    """Shared bounded UTF-8 text range returned by read-side tools."""

    path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int
    returned_lines: int = 0
    returned_chars: int = 0
    total_chars: int = 0
    full_resource_read: bool = False
    guidance: str | None = None


class FileReadResult(TextRange):
    """Result returned by the read_file tool."""

    content_sha256: str | None = Field(default=None, exclude=True)


class ArtifactReadResult(TextRange):
    """Result returned by the read_artifact tool."""


class SearchMatch(BaseModel):
    """One text search match."""

    path: str
    line: int
    text: str


class SearchResult(BaseModel):
    """Result returned by the search_text tool."""

    query: str
    matches: list[SearchMatch] = Field(default_factory=list)
    truncated: bool = False
    truncation_reason: str | None = None
    scanned_entries: int = 0


class GitDiffResult(BaseModel):
    """Result returned by the inspect_git_diff tool."""

    command: str = "git diff && git status --short"
    returncode: int
    diff: str
    stderr: str = ""
    status_short: str = ""
    untracked_files: list[str] = Field(default_factory=list)


def find_files(
    workspace: Path | str,
    path: Path | str = ".",
    pattern: str = "**/*",
    *,
    max_results: int = 200,
    max_depth: int = 12,
    cancellation_token: CancellationToken | None = None,
) -> FindFilesResult:
    """Find non-sensitive workspace files through bounded filesystem discovery."""

    normalized_pattern = pattern.strip()
    if not normalized_pattern:
        raise ValueError("pattern must not be empty.")
    if max_results < 1 or max_depth < 0:
        raise ValueError("max_results must be positive and max_depth must be non-negative.")

    guard = WorkspaceGuard(workspace)
    root = guard.resolve(path)
    root_display = guard.relative_path(root)
    if not root.exists():
        return FindFilesResult(
            root=root_display,
            pattern=normalized_pattern,
            exists=False,
            is_directory=False,
        )
    if not root.is_dir():
        return FindFilesResult(
            root=root_display,
            pattern=normalized_pattern,
            exists=True,
            is_directory=False,
        )

    state = _FileSearchState(
        deadline=time.monotonic() + FILE_SEARCH_TIMEOUT_SECONDS,
        scan_limit=FILE_SEARCH_SCAN_LIMIT,
        cancellation_token=cancellation_token,
    )
    matches: list[Path] = []
    for candidate in _iter_file_candidates(
        guard,
        root,
        max_depth,
        state,
    ):
        relative_to_root = candidate.relative_to(root).as_posix()
        if not _matches_file_pattern(relative_to_root, candidate.name, normalized_pattern):
            continue
        if len(matches) >= max_results:
            state.stop("result_limit")
            break
        matches.append(candidate)

    return FindFilesResult(
        root=root_display,
        pattern=normalized_pattern,
        files=[guard.relative_path(candidate) for candidate in matches],
        file_details=[
            _file_search_entry(guard, candidate)
            for candidate in matches
        ],
        truncated=state.truncation_reason is not None,
        truncation_reason=state.truncation_reason,
        scanned_entries=state.scanned_entries,
    )


def _iter_file_candidates(
    guard: WorkspaceGuard,
    root: Path,
    max_depth: int,
    state: _FileSearchState,
):
    """Yield candidates from the bounded prunable filesystem walk."""

    yield from _iter_workspace_files(guard, root, max_depth, state)


def _iter_workspace_files(
    guard: WorkspaceGuard,
    root: Path,
    max_depth: int,
    state: _FileSearchState,
):
    """Yield safe files with directory pruning and deterministic local ordering."""

    stack: list[tuple[Path, int, bool, bool]] = [(root, 0, True, False)]
    while stack:
        if state.interrupted():
            return
        candidate, depth, is_directory, is_file = stack.pop()
        if candidate != root and not state.admit_next():
            return
        if not guard.is_within_workspace(candidate) or guard.is_sensitive(candidate):
            continue
        if is_directory:
            if candidate != root and (
                depth >= max_depth or _should_prune_directory(candidate)
            ):
                continue
            try:
                with os.scandir(candidate) as directory_entries:
                    entries = sorted(
                        directory_entries,
                        key=lambda entry: (entry.name.casefold(), entry.name),
                        reverse=True,
                    )
            except OSError:
                continue
            for entry in entries:
                entry_path = Path(entry.path)
                if entry.is_symlink() or _is_junction(entry_path):
                    continue
                try:
                    entry_is_directory = entry.is_dir(follow_symlinks=False)
                    entry_is_file = entry.is_file(follow_symlinks=False)
                except OSError:
                    continue
                if entry_is_directory or entry_is_file:
                    stack.append(
                        (entry_path, depth + 1, entry_is_directory, entry_is_file)
                    )
            continue
        if is_file and depth <= max_depth:
            yield candidate


def _is_safe_candidate(
    guard: WorkspaceGuard,
    root: Path,
    candidate: Path,
) -> bool:
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return False
    return guard.is_within_workspace(candidate) and not guard.is_sensitive(candidate)


def _should_prune_directory(path: Path) -> bool:
    return path.name.casefold() in DEFAULT_SEARCH_IGNORED_DIRECTORIES


def _file_search_entry(guard: WorkspaceGuard, path: Path) -> FileSearchEntry:
    try:
        size_bytes = max(0, int(path.stat().st_size))
    except OSError:
        size_bytes = 0
    estimated_lines = (
        0
        if size_bytes == 0
        else max(1, (size_bytes + ESTIMATED_BYTES_PER_LINE - 1) // ESTIMATED_BYTES_PER_LINE)
    )
    return FileSearchEntry(
        path=guard.relative_path(path),
        size_bytes=size_bytes,
        estimated_lines=estimated_lines,
    )


def is_large_text_resource(*, total_lines: int, total_chars: int) -> bool:
    """Return whether one successful read should surface high-cost guidance."""

    return (
        total_lines >= LARGE_READ_FEEDBACK_LINE_THRESHOLD
        or total_chars >= LARGE_READ_FEEDBACK_CHAR_THRESHOLD
    )


def _is_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _matches_file_pattern(relative_path: str, file_name: str, pattern: str) -> bool:
    normalized_pattern = pattern.strip().replace("\\", "/")
    normalized_path = relative_path.strip().replace("\\", "/").strip("/")
    while normalized_pattern.startswith("./"):
        normalized_pattern = normalized_pattern[2:]
    normalized_pattern = normalized_pattern.strip("/")
    if not normalized_pattern:
        return False
    if "/" not in normalized_pattern:
        return fnmatch.fnmatch(file_name, normalized_pattern)

    path_parts = tuple(part for part in normalized_path.split("/") if part)
    pattern_parts = tuple(part for part in normalized_pattern.split("/") if part)
    memo: dict[tuple[int, int], bool] = {}

    def matches(path_index: int, pattern_index: int) -> bool:
        key = (path_index, pattern_index)
        if key in memo:
            return memo[key]
        if pattern_index == len(pattern_parts):
            result = path_index == len(path_parts)
        elif pattern_parts[pattern_index] == "**":
            result = matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts)
                and matches(path_index + 1, pattern_index)
            )
        else:
            result = (
                path_index < len(path_parts)
                and fnmatch.fnmatch(path_parts[path_index], pattern_parts[pattern_index])
                and matches(path_index + 1, pattern_index + 1)
            )
        memo[key] = result
        return result

    return matches(0, 0)


def read_file(
    workspace: Path | str,
    path: Path | str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> FileReadResult:
    """Read a UTF-8 text file inside the workspace."""

    guard = WorkspaceGuard(workspace)
    file_path = guard.resolve(path)
    if file_path.is_dir():
        display_path = guard.relative_path(file_path)
        raise IsADirectoryError(
            f"Path is a directory: {display_path}. "
            f'Use search(kind="files", query="*", path="{display_path}", max_depth=1) '
            "to inspect its files."
        )
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {guard.relative_path(file_path)}")

    raw_content = file_path.read_bytes()
    content = raw_content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    result = _read_text_line_range(
        file_path,
        display_path=guard.relative_path(file_path),
        start_line=start_line,
        end_line=end_line,
        content=content,
    )
    return FileReadResult(
        **result.model_dump(),
        content_sha256=hashlib.sha256(raw_content).hexdigest(),
    )


def read_artifact(
    artifact_root: Path | str,
    path: Path | str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ArtifactReadResult:
    """Read a UTF-8 text artifact produced under the current run artifacts directory."""

    if _looks_like_evidence_id(path):
        raise ValueError(
            "read_artifact path must be an artifact path, not an evidence id. "
            "Use the artifact_path value attached to the evidence item."
        )

    root = Path(artifact_root).resolve()
    file_path = _resolve_artifact_read_path(root, path)
    try:
        relative_path = file_path.relative_to(root)
    except ValueError as exc:
        raise WorkspaceAccessError(f"Path escapes artifact directory: {path}") from exc
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not an artifact file: {relative_path.as_posix()}")

    result = _read_text_line_range(
        file_path,
        display_path=relative_path.as_posix(),
        start_line=start_line,
        end_line=end_line,
    )
    return ArtifactReadResult.model_validate(result.model_dump())


def _read_text_line_range(
    file_path: Path,
    *,
    display_path: str,
    start_line: int | None,
    end_line: int | None,
    content: str | None = None,
) -> TextRange:
    """Read one validated UTF-8 line range with shared file/artifact semantics."""

    content = file_path.read_text(encoding="utf-8") if content is None else content
    lines = content.splitlines()
    total_lines = len(lines)
    start = start_line if start_line is not None else 1
    if start < 1:
        raise ValueError("start_line must be at least 1.")
    if end_line is not None and end_line < start:
        raise ValueError("end_line must be greater than or equal to start_line.")
    total_chars = len(content)
    if total_lines == 0:
        return TextRange(
            path=display_path,
            content="",
            start_line=1,
            end_line=0,
            total_lines=0,
            returned_lines=0,
            returned_chars=0,
            total_chars=total_chars,
            full_resource_read=True,
        )
    if start > total_lines:
        raise ValueError("start_line must not exceed the number of lines in the text resource.")

    end = min(end_line if end_line is not None else total_lines, total_lines)
    returned_content = "\n".join(lines[start - 1 : end])
    full_resource_read = start == 1 and end == total_lines
    guidance = (
        "High-cost full-resource read. For additional large candidate source files or "
        "Artifacts, use text search on the exact resource before reading a local range; "
        "file discovery alone does not locate relevant content."
        if full_resource_read
        and is_large_text_resource(total_lines=total_lines, total_chars=total_chars)
        else None
    )
    return TextRange(
        path=display_path,
        content=returned_content,
        start_line=start,
        end_line=end,
        total_lines=total_lines,
        returned_lines=end - start + 1,
        returned_chars=len(returned_content),
        total_chars=total_chars,
        full_resource_read=full_resource_read,
        guidance=guidance,
    )


def _resolve_artifact_read_path(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _looks_like_evidence_id(path: Path | str) -> bool:
    return str(path).strip().startswith("ev_")


def search_text(
    workspace: Path | str,
    query: str,
    path: Path | str = ".",
    *,
    max_matches: int = DEFAULT_MAX_MATCHES,
    max_depth: int = 12,
    use_regex: bool = False,
    case_sensitive: bool = True,
    file_glob: str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> SearchResult:
    """Search text through bounded prunable filesystem candidates."""

    if not query:
        raise ValueError("query must not be empty.")
    if max_matches < 1 or max_depth < 0:
        raise ValueError("max_matches must be positive and max_depth must be non-negative.")

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query, flags) if use_regex else None
    needle = query if case_sensitive else query.casefold()
    guard = WorkspaceGuard(workspace)
    search_root = guard.resolve(path)
    state = _FileSearchState(
        deadline=time.monotonic() + TEXT_SEARCH_TIMEOUT_SECONDS,
        scan_limit=TEXT_SEARCH_SCAN_LIMIT,
        cancellation_token=cancellation_token,
    )
    matches: list[SearchMatch] = []

    for file_path in _iter_text_search_candidates(
        guard,
        search_root,
        file_glob=file_glob,
        max_depth=max_depth,
        state=state,
    ):
        if state.interrupted():
            break
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if state.interrupted():
                        return _search_result(query, matches, state)
                    line = raw_line.rstrip("\r\n")
                    haystack = line if case_sensitive else line.casefold()
                    matched = bool(pattern.search(line)) if pattern is not None else needle in haystack
                    if not matched:
                        continue
                    matches.append(
                        SearchMatch(
                            path=guard.relative_path(file_path),
                            line=line_number,
                            text=line,
                        )
                    )
                    if len(matches) >= max_matches:
                        state.stop("result_limit")
                        return _search_result(query, matches, state)
        except (OSError, UnicodeDecodeError):
            continue

    return _search_result(query, matches, state)


def _iter_text_search_candidates(
    guard: WorkspaceGuard,
    search_root: Path,
    *,
    file_glob: str | None,
    max_depth: int,
    state: _FileSearchState,
):
    """Yield filename-filtered text candidates before opening file contents."""

    if search_root.is_file():
        if not state.admit_next() or not _is_safe_candidate(guard, search_root.parent, search_root):
            return
        if file_glob and not _matches_file_pattern(search_root.name, search_root.name, file_glob):
            return
        yield search_root
        return
    if not search_root.is_dir():
        return

    for candidate in _iter_file_candidates(
        guard,
        search_root,
        max_depth,
        state,
    ):
        relative = candidate.relative_to(search_root)
        if _is_under_ignored_directory(relative):
            continue
        relative_path = relative.as_posix()
        if file_glob and not _matches_file_pattern(relative_path, candidate.name, file_glob):
            continue
        yield candidate


def _is_under_ignored_directory(relative_path: Path) -> bool:
    return any(
        part.casefold() in DEFAULT_SEARCH_IGNORED_DIRECTORIES
        for part in relative_path.parts[:-1]
    )


def _search_result(
    query: str,
    matches: list[SearchMatch],
    state: _FileSearchState,
) -> SearchResult:
    return SearchResult(
        query=query,
        matches=matches,
        truncated=state.truncation_reason is not None,
        truncation_reason=state.truncation_reason,
        scanned_entries=state.scanned_entries,
    )


def inspect_git_diff(workspace: Path | str) -> GitDiffResult:
    """Return the current workspace Git diff without inheriting a parent repository."""

    guard = WorkspaceGuard(workspace)
    try:
        root_completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=guard.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return GitDiffResult(returncode=127, diff="", stderr="git executable was not found.")

    if root_completed.returncode != 0:
        return GitDiffResult(
            returncode=root_completed.returncode,
            diff="",
            stderr=root_completed.stderr or "Workspace is not a Git repository.",
        )

    git_root = Path(root_completed.stdout.strip()).resolve()
    if os.path.normcase(str(git_root)) != os.path.normcase(str(guard.root.resolve())):
        return GitDiffResult(
            returncode=2,
            diff="",
            stderr=(
                "Workspace is nested inside another Git repository; refusing to inspect "
                f"parent repository diff at {git_root}."
            ),
        )

    try:
        diff_completed = subprocess.run(
            ["git", "diff"],
            cwd=guard.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        status_args = ["git", "status", "--short"]
        status_args.append("--untracked-" + "files=all")
        status_completed = subprocess.run(
            status_args,
            cwd=guard.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return GitDiffResult(returncode=127, diff="", stderr="git executable was not found.")

    stderr = chr(10).join(
        part for part in [diff_completed.stderr, status_completed.stderr] if part
    )
    return GitDiffResult(
        returncode=diff_completed.returncode
        if diff_completed.returncode != 0
        else status_completed.returncode,
        diff=diff_completed.stdout,
        stderr=stderr,
        status_short=status_completed.stdout,
        untracked_files=_untracked_files_from_status(status_completed.stdout),
    )


def _untracked_files_from_status(status_short: str) -> list[str]:
    files: list[str] = []
    for line in status_short.splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:].strip().replace("\\", "/")
        if path:
            files.append(path)
    return files
