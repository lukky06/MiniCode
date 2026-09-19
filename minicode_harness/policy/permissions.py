"""Execution-mode decisions for side-effecting tools."""

from __future__ import annotations

from enum import StrEnum


class ExecutionMode(StrEnum):
    """User-facing execution behavior for admitted side effects."""

    DEFAULT = "default"
    REVIEW_CHANGES = "review-changes"
    FULL_ACCESS = "full-access"


class PermissionDecision(StrEnum):
    """Effective interactive decision after deterministic admission."""

    ALLOW = "allow"
    ASK = "ask"


DEFAULT_EXECUTION_MODE = ExecutionMode.DEFAULT

_WORKSPACE_MUTATION_TOOLS = {"edit", "write", "apply_patch"}


def decide_permission(
    *,
    tool_name: str,
    requires_approval: bool,
    execution_mode: ExecutionMode,
) -> PermissionDecision:
    """Resolve whether an already-admitted tool needs interactive approval."""

    if not requires_approval:
        return PermissionDecision.ALLOW
    if execution_mode == ExecutionMode.FULL_ACCESS:
        return PermissionDecision.ALLOW
    if (
        execution_mode == ExecutionMode.DEFAULT
        and tool_name in _WORKSPACE_MUTATION_TOOLS
    ):
        return PermissionDecision.ALLOW
    return PermissionDecision.ASK
