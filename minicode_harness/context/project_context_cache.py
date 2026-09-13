"""Project-scoped cache for fresh workspace read results.

Cached entries live outside the workspace and are invalidated by the current
file hash. This is execution infrastructure, not model-visible memory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from minicode_harness.models import NormalizedToolCall
from minicode_harness.storage import default_data_dir, workspace_hash
from minicode_harness.tools.read_tools import FileReadResult
from minicode_harness.workspace import WorkspaceGuard


class CachedToolResult(BaseModel):
    workspace_hash: str
    run_id: str
    step: int
    tool_name: str
    arguments_hash: str
    arguments: dict[str, Any]
    summary: str
    output_preview: str
    result_payload: dict[str, Any] = Field(default_factory=dict)
    file_path: str | None = None
    file_hash: str | None = None
    created_at: str


class ProjectContextCache:
    """Persist reusable workspace-read results outside the repository."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()

    def project_dir(self, workspace: Path | str) -> Path:
        self._ensure_outside_workspace(workspace)
        return self.data_dir / "projects" / workspace_hash(workspace)

    def lookup_workspace_read(
        self,
        *,
        workspace: Path | str,
        arguments: dict[str, Any],
    ) -> CachedToolResult | None:
        normalized_arguments, file_path = self._normalize_workspace_read_arguments(
            workspace,
            arguments,
        )
        cache_path = self._tool_cache_path(workspace)
        if not cache_path.is_file():
            return None
        arguments_hash = _hash_json(normalized_arguments)
        current_hash = _file_hash(file_path)
        for line in reversed(cache_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                cached = CachedToolResult.model_validate_json(line)
            except ValueError:
                continue
            if cached.arguments_hash == arguments_hash:
                return cached if cached.file_hash == current_hash else None
        return None

    def save_workspace_read_result(
        self,
        *,
        workspace: Path | str,
        run_id: str,
        step: int,
        tool_call: NormalizedToolCall,
        result: FileReadResult,
    ) -> CachedToolResult:
        normalized_arguments, absolute_path = self._normalize_workspace_read_arguments(
            workspace,
            tool_call.arguments,
        )
        cached = CachedToolResult(
            workspace_hash=workspace_hash(workspace),
            run_id=run_id,
            step=step,
            tool_name="read",
            arguments_hash=_hash_json(normalized_arguments),
            arguments=normalized_arguments,
            summary=(
                f"Read file {result.path} lines {result.start_line}-{result.end_line} "
                f"of {result.total_lines}."
            ),
            output_preview=result.content[:1000],
            result_payload=result.model_dump(mode="json"),
            file_path=result.path,
            file_hash=_file_hash(absolute_path),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        cache_path = self._tool_cache_path(workspace)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as cache_file:
            cache_file.write(cached.model_dump_json())
            cache_file.write("\n")
        return cached

    def _tool_cache_path(self, workspace: Path | str) -> Path:
        return self.project_dir(workspace) / "tool-cache" / "workspace-read.jsonl"

    def _normalize_workspace_read_arguments(
        self,
        workspace: Path | str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        guard = WorkspaceGuard(workspace)
        raw_path = arguments.get("target")
        if not raw_path:
            raise ValueError("workspace read cache lookup requires a target argument.")
        file_path = guard.resolve(str(raw_path))
        return (
            {
                "target": guard.relative_path(file_path),
                "start_line": arguments.get("start_line"),
                "end_line": arguments.get("end_line"),
            },
            file_path,
        )

    def _ensure_outside_workspace(self, workspace: Path | str) -> None:
        workspace_path = Path(workspace).resolve()
        data_path = self.data_dir.resolve()
        if data_path == workspace_path or data_path.is_relative_to(workspace_path):
            raise ValueError("Project context cache data_dir must not be inside the workspace.")


def _hash_json(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
