"""Unified approval-policy and permission-mode decisions."""

from __future__ import annotations

from enum import StrEnum


class ApprovalPolicy(StrEnum):
    """When MiniCode may ask the user to approve a side effect."""

    ON_REQUEST = "on-request"
    NEVER = "never"


class PermissionMode(StrEnum):
    """Which admitted side effects may execute without per-call approval."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    FULL_ACCESS = "full-access"


class PermissionDecision(StrEnum):
    """Effective decision after deterministic admission and session grants."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


DEFAULT_APPROVAL_POLICY = ApprovalPolicy.ON_REQUEST
DEFAULT_PERMISSION_MODE = PermissionMode.READ_ONLY

_WORKSPACE_MUTATION_TOOLS = {"edit", "write", "apply_patch"}


def decide_permission(
    *,
    tool_name: str,
    requires_approval: bool,
    is_command: bool,
    approval_policy: ApprovalPolicy,
    permission_mode: PermissionMode,
) -> PermissionDecision:
    """Resolve only the interactive-approval layer.

    Deterministic tool/command admission runs before this function and cannot be
    bypassed by any permission mode.
    """

    if not requires_approval:
        return PermissionDecision.ALLOW

    if (
        permission_mode == PermissionMode.WORKSPACE_WRITE
        and tool_name in _WORKSPACE_MUTATION_TOOLS
    ):
        return PermissionDecision.ALLOW

    if is_command:
        return (
            PermissionDecision.ASK
            if approval_policy == ApprovalPolicy.ON_REQUEST
            else PermissionDecision.DENY
        )

    if permission_mode == PermissionMode.FULL_ACCESS:
        return PermissionDecision.ALLOW

    return (
        PermissionDecision.ASK
        if approval_policy == ApprovalPolicy.ON_REQUEST
        else PermissionDecision.DENY
    )
