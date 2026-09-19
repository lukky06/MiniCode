"""Hook protocol and decision types for MiniCodeHarness.

Hooks are deterministic lifecycle extensions executed by the harness, not by the
model. Every declared lifecycle event is emitted for audit and instrumentation.
Only ``pre_tool_use`` decisions currently alter execution; decisions from other
events are observational so they cannot create a hidden workflow state machine.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from minicode_harness.context import ContextObservation


HookEventName = Literal[
    "session_start",
    "before_context_build",
    "after_context_build",
    "before_model_call",
    "after_model_response",
    "pre_tool_use",
    "post_tool_use",
    "before_checkpoint",
    "stop",
    "error",
]

class HookEvent(BaseModel):
    """One harness lifecycle event delivered to registered hooks."""

    name: HookEventName
    run_id: str
    task: str
    workspace: str
    step: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class HookDecision(BaseModel):
    """Decision returned by a hook.

    ``allow`` continues execution; ``block`` returns a deterministic observation.
    """

    action: Literal["allow", "block"] = "allow"
    hook_name: str = ""
    reason: str = ""
    observation: ContextObservation | None = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def allow(cls) -> "HookDecision":
        return cls(action="allow")

    @classmethod
    def block(
        cls,
        *,
        hook_name: str,
        reason: str,
        observation: ContextObservation,
    ) -> "HookDecision":
        return cls(
            action="block",
            hook_name=hook_name,
            reason=reason,
            observation=observation,
        )


class Hook(Protocol):
    """Protocol implemented by all deterministic hooks."""

    name: str
    events: tuple[HookEventName, ...]

    def handle(self, event: HookEvent) -> HookDecision:
        """Handle one lifecycle event."""
        ...
