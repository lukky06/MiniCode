"""Context and run-state model types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenBudget(BaseModel):
    """MVP context budget settings."""

    context_budget: int = 32000
    reserved_output: int = 6000
    soft_limit: float = 0.80
    hard_limit: float = 0.95

    @property
    def prompt_budget(self) -> int:
        return self.context_budget - self.reserved_output

    @property
    def soft_token_limit(self) -> int:
        return int(self.prompt_budget * self.soft_limit)

    @property
    def hard_token_limit(self) -> int:
        return int(self.prompt_budget * self.hard_limit)


class ContextSkill(BaseModel):
    """Compact skill catalog entry exposed in the system prompt."""

    name: str
    description: str
    source: str


class ContextObservation(BaseModel):
    """One tool result retained for trace, hooks, checkpoint, and resume."""

    tool_call_id: str
    tool_name: str
    content: str
    output_preview: str
    token_estimate: int
    summary: str = ""
    is_important: bool = False
    is_truncated: bool = False
    artifact_path: str | None = None
    lossiness: Literal["none", "excerpted", "summarized"] = "none"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationState(BaseModel):
    """Structured result of the latest verification action for one run."""

    status: Literal["not_run", "passed", "failed", "rolled_back"] = "not_run"
    command: str | None = None
    returncode: int | None = None
    reason: str | None = None


class InspectedFile(BaseModel):
    """Strongest known read coverage for one file in the current run."""

    path: str
    summary: str
    last_tool_call_id: str
    last_step: int
    line_start: int | None = None
    line_end: int | None = None
    total_lines: int | None = None
    artifact_path: str | None = None
    content_status: str = "summary_only"
    content_sha256: str | None = None
    workspace_generation: int = 0


class RunState(BaseModel):
    """Minimal deterministic state required by checkpoint and resume.

    Tool evidence and execution progress live in canonical messages,
    observations, and trace. This model stores only file-read coverage and the
    latest structured verification result.
    """

    inspected_files: list[InspectedFile] = Field(default_factory=list)
    verification: VerificationState = Field(default_factory=VerificationState)


class ContextCompressionEvent(BaseModel):
    """One context compression event."""

    reason: str
    before_tokens: int
    after_tokens: int
    details: dict[str, Any] = Field(default_factory=dict)


class BuiltContext(BaseModel):
    """Dynamic system and call-time context prepared before message budgeting."""

    messages: list[dict[str, Any]]
    token_estimate: int
    compression_events: list[ContextCompressionEvent] = Field(default_factory=list)
    prompt_prefix_hash: str | None = None
    prompt_prefix_tokens: int = 0
