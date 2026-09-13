"""Pure terminal-layer types that never enter model context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from minicode_harness.output import OutputSink
from minicode_harness.policy import ApprovalPolicy, PermissionMode
from minicode_harness.runtime import CollaborationMode, DEFAULT_COLLABORATION_MODE
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.state import ApprovalClient, RunStore
from minicode_harness.tools import SandboxMode

if TYPE_CHECKING:
    from minicode_harness.state import ReplSessionStore
    from minicode_harness.runtime.run_executor import RunExecutor


class SessionLaunchMode(StrEnum):
    """How an interactive terminal obtains its conversation session."""

    NEW = "new"
    CONTINUE = "continue"
    EXACT = "exact"


@dataclass(frozen=True)
class SlashCommand:
    """Parsed slash command."""

    name: str
    arguments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommandResult:
    """Result returned by the terminal command router."""

    status: Literal["completed", "failed", "exit"]
    content: str | None = None
    session_id: str | None = None


@dataclass
class TerminalRunState:
    """Pure UI state; never persisted or rendered into model context."""

    phase: Literal[
        "idle",
        "preparing",
        "thinking",
        "tool_running",
        "approval",
        "answering",
        "completed",
        "failed",
        "cancelled",
    ] = "idle"
    run_id: str | None = None
    model_name: str | None = None
    activity: str | None = None
    activity_target: str | None = None
    active_tool: str | None = None
    tool_count: int = 0
    started_at_monotonic: float | None = None
    context_token_estimate: int | None = None
    context_window: int | None = None
    prompt_budget: int | None = None
    reserved_output: int | None = None
    context_remaining: int | None = None


@dataclass
class TerminalSessionSettings:
    """Mutable terminal-only defaults applied when the next Run starts."""

    collaboration_mode: CollaborationMode = DEFAULT_COLLABORATION_MODE


@dataclass(frozen=True)
class TerminalContext:
    """Runtime facts available to slash commands."""

    workspace: Path
    provider: str
    model: str | None
    write_enabled: bool
    prompt_cache_enabled: bool
    repository_memory_enabled: bool
    subagents_enabled: bool
    mcp_config: Path | None
    output_sink: OutputSink
    approval_client: ApprovalClient
    run_store: RunStore
    approval_policy: ApprovalPolicy = ApprovalPolicy.ON_REQUEST
    permission_mode: PermissionMode = PermissionMode.READ_ONLY
    sandbox_mode: SandboxMode = SandboxMode.LOCAL
    sandbox_image: str | None = None
    session_settings: TerminalSessionSettings = field(
        default_factory=TerminalSessionSettings
    )
    session_store: ReplSessionStore | None = None
    executor: RunExecutor | None = None
    session_id: str | None = None
    cancellation_token: CancellationToken | None = None
    steering_queue: SteeringQueue | None = None
