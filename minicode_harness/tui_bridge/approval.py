"""ApprovalClient implementation backed by JSONL request/response messages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from threading import Event, Lock

from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.state import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)

from .output import JsonlOutputSink
from .protocol import ApprovalRequired


@dataclass
class _PendingApproval:
    request_id: str
    completed: Event
    response: ApprovalResponse | None = None


class JsonlApprovalClient:
    """Keep approval semantics in Python while letting the TUI choose a decision."""

    def __init__(self, output_sink: JsonlOutputSink) -> None:
        self.output_sink = output_sink
        self._decision_lock = Lock()
        self._pending_lock = Lock()
        self._pending: _PendingApproval | None = None
        self._cancellation_token: CancellationToken | None = None

    def bind_cancellation(self, token: CancellationToken | None) -> None:
        self._cancellation_token = token

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        with self._decision_lock:
            if self._is_cancelled():
                return _abort_response("Run was cancelled before approval.")

            pending = _PendingApproval(request_id=request.id, completed=Event())
            with self._pending_lock:
                self._pending = pending

            summary, details = _approval_copy(request)
            self.output_sink.writer.emit(
                ApprovalRequired(
                    id=request.id,
                    tool_call_id=request.tool_call_id,
                    tool=request.tool_name,
                    summary=summary,
                    details=details,
                    can_approve_session=request.can_approve_session,
                )
            )

            while not pending.completed.wait(0.05):
                if self._is_cancelled():
                    with self._pending_lock:
                        if (
                            self._pending is pending
                            and not pending.completed.is_set()
                        ):
                            pending.response = _abort_response(
                                "Run was cancelled while approval was pending."
                            )
                            pending.completed.set()

            with self._pending_lock:
                if self._pending is pending:
                    self._pending = None
            return pending.response or _abort_response("Approval ended without a decision.")

    def resolve(
        self,
        request_id: str,
        decision: str,
    ) -> tuple[bool, str | None]:
        with self._pending_lock:
            pending = self._pending
            if pending is None:
                return False, "No approval is currently pending."
            if pending.request_id != request_id:
                return False, f"Stale approval response: {request_id}."
            if pending.completed.is_set():
                return False, f"Approval is already resolved: {request_id}."
            try:
                resolved = ApprovalDecision(decision)
            except ValueError:
                return False, f"Unsupported approval decision: {decision}."
            pending.response = ApprovalResponse(
                decision=resolved,
                reason="Decision supplied by the TypeScript TUI.",
            )
            pending.completed.set()
            return True, None

    def has_pending(self) -> bool:
        with self._pending_lock:
            return self._pending is not None and not self._pending.completed.is_set()

    def _is_cancelled(self) -> bool:
        return bool(
            self._cancellation_token is not None
            and self._cancellation_token.is_cancelled
        )


def _approval_copy(request: ApprovalRequest) -> tuple[str, str]:
    """Render only Python-owned preview facts; raw model arguments never cross."""

    preview = dict(request.preview)
    summary = str(preview.get("summary") or "").strip()
    if not summary:
        if request.tool_name == "run_command":
            summary = str(preview.get("command") or "Run command")
        elif request.tool_name in {"edit", "write"}:
            summary = str(preview.get("path") or request.tool_name)
        elif request.tool_name == "apply_patch":
            files = preview.get("files") or []
            summary = ", ".join(str(path) for path in files) or "Apply patch"
        else:
            summary = request.tool_name

    safe_details = {
        "risk_level": request.risk_level,
        "preview": preview,
    }
    details = json.dumps(
        safe_details,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return _bound(summary, 500), _bound(details, 12_000)


def _bound(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _abort_response(reason: str) -> ApprovalResponse:
    return ApprovalResponse(
        decision=ApprovalDecision.ABORT,
        reason=reason,
    )
