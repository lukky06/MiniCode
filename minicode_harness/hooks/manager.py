"""Deterministic hook manager."""

from __future__ import annotations

from collections.abc import Iterable

from minicode_harness.trace import TraceWriter

from .types import Hook, HookDecision, HookEvent


class HookManager:
    """Run lifecycle hooks in registration order.

    A non-allow decision short-circuits later hooks for the same event. Hook
    failures are recorded in trace and treated as ``allow`` so a broken hook does
    not corrupt the main agent loop.
    """

    def __init__(
        self,
        hooks: Iterable[Hook] | None = None,
        *,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.hooks = list(hooks or [])
        self.trace_writer = trace_writer

    def emit(self, event: HookEvent) -> HookDecision:
        """Emit one event and return the first non-allow decision."""

        for hook in self.hooks:
            if event.name not in hook.events:
                continue
            try:
                decision = hook.handle(event)
            except Exception as exc:  # pragma: no cover - defensive audit path
                self._trace_error(event, hook.name, exc)
                continue
            if decision.action != "allow":
                if not decision.hook_name:
                    decision.hook_name = hook.name
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

    def _trace_error(self, event: HookEvent, hook_name: str, exc: Exception) -> None:
        if self.trace_writer is None:
            return
        self.trace_writer.write_event(
            "hook_error",
            step=event.step,
            hook=hook_name,
            lifecycle_event=event.name,
            error_type=type(exc).__name__,
            error=str(exc),
        )
