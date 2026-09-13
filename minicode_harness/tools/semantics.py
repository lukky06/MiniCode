"""Canonical model-facing Tool semantics."""

from __future__ import annotations

from typing import Any


def read_source(tool_name: str, arguments: dict[str, Any] | None = None) -> str | None:
    if tool_name != "read":
        return None
    source = str((arguments or {}).get("source") or "")
    return source or None


def read_target(tool_name: str, arguments: dict[str, Any] | None = None) -> str | None:
    if tool_name != "read":
        return None
    target = (arguments or {}).get("target")
    text = str(target or "").strip()
    return text or None


def search_source(tool_name: str, arguments: dict[str, Any] | None = None) -> str | None:
    if tool_name != "search":
        return None
    source = str((arguments or {}).get("source") or "workspace")
    return source or None


def search_kind(tool_name: str, arguments: dict[str, Any] | None = None) -> str | None:
    if tool_name != "search":
        return None
    kind = str((arguments or {}).get("kind") or "")
    return kind or None


def is_workspace_read(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    return read_source(tool_name, arguments) == "workspace"


def is_artifact_read(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    return read_source(tool_name, arguments) == "artifact"


def is_memory_read(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    return read_source(tool_name, arguments) == "memory"


def is_skill_read(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    return read_source(tool_name, arguments) == "skill"


def is_diff_read(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    return read_source(tool_name, arguments) == "diff"


def is_file_search(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    return search_kind(tool_name, arguments) == "files"


def is_text_search(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    return search_kind(tool_name, arguments) == "text"


def is_edit_tool(tool_name: str) -> bool:
    return tool_name == "edit"


def is_write_tool(tool_name: str) -> bool:
    return tool_name in {"edit", "write", "apply_patch"}


def is_task_tool(tool_name: str) -> bool:
    return tool_name == "task"
