"""Rich approval client for the Python minicode exec path."""

from __future__ import annotations

from collections.abc import Callable
import json
import sys

from prompt_toolkit import prompt
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from minicode_harness.state import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)
from minicode_harness.terminal.rendering import DiffRenderer
from minicode_harness.terminal.transient_status import TransientStatusLine
from minicode_harness.terminal.types import TerminalRunState


class TerminalApprovalClient:
    """Render structured approvals without changing runtime decision semantics."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        answer_reader: Callable[[str], str] | None = None,
        interactive: bool | None = None,
        diff_renderer: DiffRenderer | None = None,
        run_state: TerminalRunState | None = None,
        transient_status: TransientStatusLine | None = None,
    ) -> None:
        self.console = console or Console()
        self.answer_reader = answer_reader or prompt
        self.interactive = (
            sys.stdin.isatty() and sys.stdout.isatty()
            if interactive is None
            else interactive
        )
        self.diff_renderer = diff_renderer or DiffRenderer(self.console)
        self.run_state = run_state
        self.transient_status = transient_status

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        if not self.interactive:
            return ApprovalResponse(
                decision=ApprovalDecision.REJECT,
                reason="Approval requires an interactive terminal.",
            )

        previous_phase = self.run_state.phase if self.run_state is not None else None
        previous_activity = self.run_state.activity if self.run_state is not None else None
        previous_target = self.run_state.activity_target if self.run_state is not None else None
        if self.transient_status is not None:
            self.transient_status.suspend()
        if self.run_state is not None:
            self.run_state.phase = "approval"
            self.run_state.activity = "waiting approval"
            self.run_state.activity_target = None
        try:
            self._render_request(request, full=False)
            while True:
                try:
                    grant_hint = "  [g] session" if request.can_approve_session else ""
                    answer = self.answer_reader(
                        "Approve? [y] once" + grant_hint + "  [n] reject  [s] skip  "
                        "[a] abort  [v] details: "
                    )
                except EOFError:
                    return ApprovalResponse(
                        decision=ApprovalDecision.REJECT,
                        reason="Approval input was unavailable.",
                    )
                except KeyboardInterrupt:
                    return ApprovalResponse(
                        decision=ApprovalDecision.ABORT,
                        reason="Approval input was interrupted.",
                    )
                normalized = answer.strip().lower()
                if normalized in {"y", "yes", "approve"}:
                    return ApprovalResponse(decision=ApprovalDecision.APPROVE)
                if normalized in {"g", "grant", "session"} and request.can_approve_session:
                    return ApprovalResponse(decision=ApprovalDecision.APPROVE_SESSION)
                if normalized in {"n", "no", "reject"}:
                    return ApprovalResponse(decision=ApprovalDecision.REJECT)
                if normalized in {"s", "skip"}:
                    return ApprovalResponse(decision=ApprovalDecision.SKIP)
                if normalized in {"a", "abort"}:
                    return ApprovalResponse(decision=ApprovalDecision.ABORT)
                if normalized in {"v", "view", "details"}:
                    self._render_request(request, full=True)
                    continue
                self.console.print(
                    "Choose y, g, n, s, a, or v." if request.can_approve_session else "Choose y, n, s, a, or v.",
                    style="yellow",
                    markup=False,
                    highlight=False,
                )
        finally:
            if self.run_state is not None and previous_phase is not None:
                self.run_state.phase = previous_phase
                self.run_state.activity = previous_activity
                self.run_state.activity_target = previous_target
            if self.transient_status is not None:
                self.transient_status.resume()

    def _render_request(self, request: ApprovalRequest, *, full: bool) -> None:
        preview = request.preview
        body = Text()
        body.append(f"Risk: {request.risk_level}\n")
        if request.tool_name == "run_command":
            body.append(
                f"Command: {preview.get('command') or ' '.join(request.arguments.get('argv', []))}\n"
            )
            body.append(f"Category: {preview.get('policy_category', 'unknown')}\n")
            effects = preview.get("effects") or []
            if effects:
                body.append(
                    "Effects: " + "; ".join(str(effect) for effect in effects) + "\n"
                )
            body.append(f"Workspace: {preview.get('workspace', '')}\n")
            body.append(f"Timeout: {preview.get('timeout_seconds', 120)}s")
        elif request.tool_name in {"edit", "write"}:
            body.append(
                f"File: {preview.get('path') or request.arguments.get('path', '')}\n"
            )
            if request.tool_name == "edit":
                body.append(
                    f"Replacement: {preview.get('old_chars', 0)} -> "
                    f"{preview.get('new_chars', 0)} chars"
                )
            else:
                action = "Create" if preview.get("created") else "Overwrite"
                body.append(f"Action: {action}\n")
                body.append(
                    f"Bytes: {preview.get('old_bytes', 0)} -> "
                    f"{preview.get('new_bytes', 0)}\n"
                )
                body.append(f"Changed lines: {preview.get('changed_lines', 0)}")
        elif request.tool_name == "apply_patch":
            files = preview.get("files") or []
            body.append(
                "Files: " + (", ".join(str(path) for path in files) or "unknown")
            )
        elif request.tool_name.startswith("mcp__"):
            body.append(
                "Tool: " + request.tool_name.removeprefix("mcp__").replace("__", "/")
            )
        else:
            body.append(f"Tool: {request.tool_name}")

        self.console.print(
            Panel(body, title=_approval_title(request.tool_name), expand=False),
        )
        diff = str(preview.get("diff") or preview.get("diff_summary") or "")
        if diff:
            self.diff_renderer.render(diff, full=full)
        if full:
            details = {
                "tool": request.tool_name,
                "risk_level": request.risk_level,
                "arguments": request.arguments,
                "preview": preview,
            }
            self.console.print(
                json.dumps(details, ensure_ascii=False, indent=2, default=str),
                markup=False,
                highlight=False,
            )


def _approval_title(tool_name: str) -> str:
    if tool_name == "run_command":
        return "Run command"
    if tool_name in {"edit", "write", "apply_patch"}:
        return "Modify workspace"
    if tool_name.startswith("mcp__"):
        return "Call MCP tool"
    return "Approval required"
