"""Tool risk classification."""

from __future__ import annotations

from enum import StrEnum


class RiskLevel(StrEnum):
    """MVP risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


LOW_RISK_TOOLS = {
    "read",
    "search",
    "task",
    "delegate_task",
    "runtime_task_status",
}
MEDIUM_RISK_TOOLS = {
    "apply_patch",
    "edit",
    "write",
    "delegate_worktree",
    "runtime_task_stop",
}
HIGH_RISK_TOOLS = {"run_command"}


def risk_level_for_tool(tool_name: str) -> RiskLevel:
    """Return the risk level for a tool."""

    if tool_name in LOW_RISK_TOOLS:
        return RiskLevel.LOW
    if tool_name in MEDIUM_RISK_TOOLS:
        return RiskLevel.MEDIUM
    if tool_name in HIGH_RISK_TOOLS:
        return RiskLevel.HIGH
    return RiskLevel.HIGH


def tool_requires_approval(tool_name: str) -> bool:
    """Return whether a tool requires interactive approval in normal runs."""

    return risk_level_for_tool(tool_name) in {RiskLevel.MEDIUM, RiskLevel.HIGH}
