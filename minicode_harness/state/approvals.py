"""Approval request persistence and decision helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


PENDING_APPROVAL_FILE = "pending_approval.json"


class ApprovalDecision(StrEnum):
    """Supported approval decisions."""

    APPROVE = "approve"
    APPROVE_SESSION = "approve_session"
    REJECT = "reject"
    SKIP = "skip"
    ABORT = "abort"


class ApprovalRequest(BaseModel):
    """Pending approval request persisted for resume."""

    id: str
    tool_call_id: str
    tool_name: str
    risk_level: str
    step: int | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    preview: dict[str, Any] = Field(default_factory=dict)
    can_approve_session: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApprovalResponse(BaseModel):
    """Approval decision."""

    decision: ApprovalDecision
    reason: str | None = None


class ApprovalClient(Protocol):
    """Decision provider for approval requests."""

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        """Return an approval decision."""


class ApprovalStore:
    """Persist pending approvals under ``runs/<run_id>/approvals``."""

    def __init__(self, approvals_dir: Path | str) -> None:
        self.approvals_dir = Path(approvals_dir)

    @property
    def pending_path(self) -> Path:
        return self.approvals_dir / PENDING_APPROVAL_FILE

    def save_pending(self, request: ApprovalRequest) -> None:
        """Persist the current pending approval request."""

        self.approvals_dir.mkdir(parents=True, exist_ok=True)
        payload = request.model_dump(mode="json")
        self.pending_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def load_pending(self) -> ApprovalRequest | None:
        """Load a pending approval request if one exists."""

        if not self.pending_path.is_file():
            return None
        payload = json.loads(self.pending_path.read_text(encoding="utf-8"))
        return ApprovalRequest.model_validate(payload)

    def clear_pending(self) -> None:
        """Clear any pending approval request."""

        if self.pending_path.exists():
            self.pending_path.unlink()


class StaticApprovalClient:
    """Approval client useful for tests or benchmark auto-approval."""

    def __init__(
        self,
        decision: ApprovalDecision = ApprovalDecision.APPROVE,
        reason: str | None = None,
    ) -> None:
        self.decision = decision
        self.reason = reason
        self.requests: list[ApprovalRequest] = []

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        return ApprovalResponse(decision=self.decision, reason=self.reason)


class NonInteractiveApprovalClient:
    """Reject side effects when no interactive approval channel exists."""

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            decision=ApprovalDecision.REJECT,
            reason="Approval requires an interactive terminal.",
        )


class InteractiveApprovalClient:
    """Console approval client for normal CLI runs."""

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        _print_preview(request, full=False)
        while True:
            try:
                grant_hint = ", [g] approve for session" if request.can_approve_session else ""
                answer = input(
                    f"Approve tool call? [y] approve once{grant_hint}, [n] reject, [s] skip, [a] abort, [v] view: "
                )
            except EOFError:
                return ApprovalResponse(
                    decision=ApprovalDecision.REJECT,
                    reason="No interactive input was available.",
                )
            normalized = answer.strip().lower()
            if normalized == "y":
                return ApprovalResponse(decision=ApprovalDecision.APPROVE)
            if normalized == "g" and request.can_approve_session:
                return ApprovalResponse(decision=ApprovalDecision.APPROVE_SESSION)
            if normalized == "n":
                return ApprovalResponse(decision=ApprovalDecision.REJECT)
            if normalized == "s":
                return ApprovalResponse(decision=ApprovalDecision.SKIP)
            if normalized == "a":
                return ApprovalResponse(decision=ApprovalDecision.ABORT)
            if normalized == "v":
                _print_preview(request, full=True)


def _print_preview(request: ApprovalRequest, *, full: bool) -> None:
    print("")
    print(f"Approval required: {request.tool_name} ({request.risk_level})")
    print(f"Tool call: {request.tool_call_id}")
    if full:
        print(json.dumps(request.preview, indent=2, ensure_ascii=False))
    else:
        summary = request.preview.get("summary") or request.preview
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("")
