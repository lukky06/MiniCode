"""Deterministic hooks used by the active AgentLoop."""

from __future__ import annotations

import re
from typing import Any

from minicode_harness.context import ContextObservation
from minicode_harness.context.token import estimate_tokens
from minicode_harness.policy import render_argv

from .manager import HookManager
from .types import HookDecision, HookEvent, HookEventName


class InvalidWriteTargetHook:
    """Block write targets that are not real workspace project files."""

    name = "invalid_write_target_guard"
    events: tuple[HookEventName, ...] = ("pre_tool_use",)

    def handle(self, event: HookEvent) -> HookDecision:
        tool_call = event.payload["tool_call"]
        loop = event.payload["loop"]
        if tool_call.name not in {"edit", "write"}:
            return HookDecision.allow()
        path = tool_call.arguments["path"].strip()
        if not _is_invalid_write_target_path(path) and not _is_artifact_output_path(path, loop):
            return HookDecision.allow()

        repeat_count = 1 + sum(
            observation.metadata.get("status")
            in {"invalid_write_target", "invalid_write_target_duplicate"}
            for observation in loop.observations
        )
        status = "invalid_write_target_duplicate" if repeat_count > 1 else "invalid_write_target"
        content = (
            f"Invalid {tool_call.name} target {path!r}. Write tools accept only workspace-relative "
            "project files, not output channels, temporary system paths, absolute paths, or run artifacts."
        )
        observation = _observation(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=content,
            status=status,
            metadata={
                "error_type": "invalid_write_target",
                "path": path,
                "repeat_count": repeat_count,
                "hook": self.name,
            },
        )
        return HookDecision.block(
            hook_name=self.name,
            reason="invalid_write_file_target",
            observation=observation,
        )


class CommandPolicyHook:
    """Block commands classified as deterministically dangerous."""

    name = "command_policy_guard"
    events: tuple[HookEventName, ...] = ("pre_tool_use",)

    def handle(self, event: HookEvent) -> HookDecision:
        tool_call = event.payload["tool_call"]
        loop = event.payload["loop"]
        if tool_call.name != "run_command":
            return HookDecision.allow()

        argv = list(tool_call.arguments["argv"])
        policy_result = event.payload["command_policy"]
        command = render_argv(argv)
        if policy_result.allowed:
            return HookDecision.allow()

        repeat_count = 1 + sum(
            observation.tool_name == "run_command"
            and observation.metadata.get("rejected_command") == command
            and observation.metadata.get("status")
            in {"command_rejected_by_policy", "command_rejected_duplicate"}
            for observation in loop.observations
        )
        status = "command_rejected_duplicate" if repeat_count > 1 else "command_rejected_by_policy"
        observation = _observation(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=_render_command_rejection_message(
                command=command,
                reason=policy_result.reason,
            ),
            status=status,
            summary=_command_rejection_summary(command, repeat_count),
            metadata={
                "error_type": "command_policy_rejected",
                "rejected_argv": argv,
                "rejected_command": command,
                "reason": policy_result.reason,
                "repeat_count": repeat_count,
                "policy_action": policy_result.action.value,
                "policy_category": policy_result.category.value,
                "risk_level": policy_result.risk_level.value,
                "effects": list(policy_result.effects),
                "hook": self.name,
            },
        )
        return HookDecision.block(
            hook_name=self.name,
            reason="command_denied_by_policy",
            observation=observation,
        )


def default_hook_manager(*, trace_writer: Any | None = None) -> HookManager:
    return HookManager(
        hooks=[InvalidWriteTargetHook(), CommandPolicyHook()],
        trace_writer=trace_writer,
    )


def _observation(
    *,
    tool_call_id: str,
    tool_name: str,
    content: str,
    status: str,
    metadata: dict[str, Any],
    summary: str | None = None,
) -> ContextObservation:
    return ContextObservation(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=content,
        output_preview=content,
        token_estimate=estimate_tokens(content),
        summary=summary or content,
        is_important=True,
        metadata={"status": status, **metadata},
    )


def _is_invalid_write_target_path(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    lowered = normalized.lower()
    if not normalized or normalized.startswith(("/", "~")):
        return True
    if re.match(r"^[A-Za-z]:/", normalized):
        return True
    if lowered in {"-", "nul", "null", "stdout", "stderr"}:
        return True
    return lowered.startswith(("dev/", "proc/", "tmp/", "/dev/", "/proc/", "/tmp/"))


def _is_artifact_output_path(path: str, loop: Any) -> bool:
    normalized = path.strip().replace("\\", "/")
    if not normalized:
        return False
    if re.match(r"^[A-Za-z_]+_call_[A-Za-z0-9_]+\.txt$", normalized):
        return True
    return any(
        normalized == str(observation.artifact_path or "").replace("\\", "/")
        for observation in loop.observations
    )


def _render_command_rejection_message(*, command: str, reason: str) -> str:
    return "\n".join(
        [
            f"Command denied by policy: {command}",
            f"Reason: {reason}",
            "Choose a workspace-local diagnostic, focused verification command, or a safer structured tool.",
        ]
    )


def _command_rejection_summary(command: str, repeat_count: int) -> str:
    if repeat_count <= 1:
        return f"Denied dangerous run_command {command!r}."
    return f"Denied repeated dangerous run_command {command!r} {repeat_count} time(s)."