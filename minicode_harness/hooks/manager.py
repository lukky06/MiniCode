"""Deterministic hook manager."""

from __future__ import annotations

from collections.abc import Iterable

from minicode_harness.trace import TraceWriter

from .types import Hook, HookDecision, HookEvent


class HookManager:
    """Run lifecycle hooks in registration order.

    A block decision short-circuits later hooks for the same event.
    """

    def __init__(
        self,
        hooks: Iterable[Hook],
        *,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.hooks = list(hooks)
        self.trace_writer = trace_writer

    def emit(self, event: HookEvent) -> HookDecision:
        """Emit one event and return the first non-allow decision."""

        for hook in self.hooks:
            if event.name not in hook.events:
                continue
            decision = hook.handle(event)
            if decision.action != "allow":
                self._trace_decision(event, decision)
                return decision
        return HookDecision.allow()

    def _trace_decision(self, event: HookEvent, decision: HookDecision) -> None:
        if self.trace_writer is None:
            return
        observation = decision.observation
        self.trace_writer.write_event(
            "hook_decision",
            step=event.step,
            hook=decision.hook_name,
            lifecycle_event=event.name,
            action=decision.action,
            reason=decision.reason,
            observation_status=(observation.metadata.get("status") if observation else None),
            tool=(event.payload.get("tool_call").name if event.payload.get("tool_call") else None),
        )
