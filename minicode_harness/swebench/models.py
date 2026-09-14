"""Schemas for the external SWE-bench adapter.

Gold fields are deliberately kept out of :class:`SweBenchInstance`, which is
used by the agent-facing runner.  Official evaluation metadata has a separate
schema and is consumed only by the evaluator boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


SweBenchStatus = Literal[
    "pending",
    "running",
    "agent_completed",
    "resolved",
    "unresolved",
    "workspace_setup_failed",
    "provider_error",
    "agent_error",
    "agent_timeout",
    "step_budget_exhausted",
    "tool_budget_exhausted",
    "no_patch",
    "patch_invalid",
    "patch_apply_failed",
    "tests_failed",
    "tests_timeout",
    "evaluation_error",
    "blocked",
]


class SweBenchInstance(BaseModel):
    """Minimal instance data allowed to reach the MiniCode runner."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str | None = None

    model_config = {"extra": "ignore"}

    @field_validator("instance_id", "repo", "base_commit", "problem_statement")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must not be empty")
        return normalized


class SweBenchGoldMetadata(BaseModel):
    """Evaluation-only metadata that never enters the runner or prompt."""

    instance_id: str
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)


class SweBenchPrediction(BaseModel):
    """Official prediction record written to ``predictions.jsonl``."""

    instance_id: str
    model_name_or_path: str
    model_patch: str


class WorkspaceManifest(BaseModel):
    """Deterministic record of one prepared instance worktree."""

    instance_id: str
    repo: str
    base_commit: str
    resolved_head: str
    workspace: str
    mirror_path: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class PatchResult(BaseModel):
    """Patch export and validation result."""

    status: Literal["ok", "no_patch", "invalid"]
    patch: str = ""
    patch_path: str | None = None
    bytes: int = 0
    changed_files: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)
    validation_error: str | None = None


class SweBenchBudget(BaseModel):
    """Bounded per-instance execution budget."""

    max_steps: int = Field(40, ge=1)
    max_tool_calls: int = Field(120, ge=1)
    max_elapsed_seconds: int | None = Field(None, ge=1)
    max_input_tokens: int | None = Field(None, ge=1)
    max_output_tokens: int | None = Field(None, ge=1)
    max_cost_usd: float | None = Field(None, ge=0)


class SweBenchInstanceResult(BaseModel):
    """Persisted outcome for one MiniCode attempt."""

    instance_id: str
    repo: str
    base_commit: str
    status: SweBenchStatus
    model_name_or_path: str
    attempt: int = Field(1, ge=1)
    run_id: str | None = None
    stop_reason: str | None = None
    final_text: str | None = None
    steps: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    modified_files: list[str] = Field(default_factory=list)
    patch_bytes: int = 0
    patch_lines: int = 0
    patch_path: str | None = None
    prediction_path: str | None = None
    workspace: str | None = None
    trace_path: str | None = None
    checkpoint_dir: str | None = None
    official_resolved: bool | None = None
    fail_to_pass_success: bool | None = None
    pass_to_pass_success: bool | None = None
    evaluation_report_path: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SweBenchEvaluationInstanceResult(BaseModel):
    """Normalized official Harness result for one instance."""

    instance_id: str
    status: SweBenchStatus
    resolved: bool | None = None
    fail_to_pass_success: bool | None = None
    pass_to_pass_success: bool | None = None
    report_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SweBenchEvaluationResult(BaseModel):
    """Normalized result of one official Harness invocation."""

    run_id: str
    status: Literal["completed", "error", "timeout"]
    returncode: int | None = None
    elapsed_seconds: float = 0.0
    stdout_path: str | None = None
    stderr_path: str | None = None
    report_paths: list[str] = Field(default_factory=list)
    instances: list[SweBenchEvaluationInstanceResult] = Field(default_factory=list)
    error: str | None = None


class SweBenchSuiteSummary(BaseModel):
    """Suite-level official and Harness efficiency summary."""

    dataset: str
    model_name_or_path: str
    total_instances: int
    total_attempts: int
    submitted: int
    patches_generated: int
    patch_applied: int
    resolved: int
    unresolved: int
    fail_to_pass_success: int
    pass_to_pass_success: int
    no_patch: int
    evaluation_errors: int
    resolve_rate: float
    avg_steps: float = 0.0
    avg_tool_calls: float = 0.0
    avg_elapsed_seconds: float = 0.0
    avg_evaluation_seconds: float = 0.0
    avg_context_tokens: float = 0.0
    repeated_read_calls: int = 0
    verification_commands: int = 0
    command_policy_rejections: int = 0
    context_compactions: int = 0
    model_recoveries: int = 0
    by_repository: dict[str, dict[str, Any]] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    by_attempt: dict[str, dict[str, Any]] = Field(default_factory=dict)
    results: list[SweBenchInstanceResult] = Field(default_factory=list)
