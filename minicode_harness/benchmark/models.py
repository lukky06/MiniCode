"""Benchmark task schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
import yaml


class BenchmarkSetup(BaseModel):
    """Commands to run before the agent starts."""

    commands: list[str] = Field(default_factory=list)


class BenchmarkAsset(BaseModel):
    """One grader-owned file copied into an isolated grading workspace."""

    source: str
    destination: str


class BenchmarkMutation(BaseModel):
    """One predefined faulty implementation used to grade generated tests."""

    id: str
    assets: list[BenchmarkAsset] = Field(default_factory=list)
    command: str | list[str] | None = None


class BenchmarkOracle(BaseModel):
    """Oracle used to score a benchmark task."""

    type: Literal[
        "none",
        "command",
        "final_text_keywords",
        "trace_contains",
        "hidden_command",
        "structured_final",
        "mutation_test",
    ] = "command"
    command: str | list[str] | None = None
    required_keywords: list[str] = Field(default_factory=list)
    event_type: str | None = None
    required_fields: dict[str, Any] = Field(default_factory=dict)
    expected_fields: dict[str, Any] = Field(default_factory=dict)
    allow_extra_fields: bool = False
    assets: list[BenchmarkAsset] = Field(default_factory=list)
    mutations: list[BenchmarkMutation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_oracle_shape(self) -> "BenchmarkOracle":
        if self.type in {"command", "hidden_command", "mutation_test"} and not self.command:
            raise ValueError(f"{self.type} oracle requires oracle.command.")
        if self.type == "hidden_command" and not self.assets:
            raise ValueError("hidden_command oracle requires at least one grader asset.")
        if self.type == "structured_final" and not self.expected_fields:
            raise ValueError("structured_final oracle requires expected_fields.")
        if self.type == "mutation_test" and not self.mutations:
            raise ValueError("mutation_test oracle requires at least one mutation.")
        return self


class BenchmarkConstraints(BaseModel):
    """Task-specific auto-approval constraints."""

    allowed_commands: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    forbidden_files: list[str] = Field(default_factory=list)
    expose_allowed_commands: bool = True


class BenchmarkExpected(BaseModel):
    """Expected and permitted task side effects."""

    modified_files: list[str] = Field(default_factory=list)
    required_modified_files: list[str] = Field(default_factory=list)
    allowed_modified_files: list[str] = Field(default_factory=list)
    max_modified_files: int | None = Field(default=None, ge=0)
    forbidden_file_changed: bool = False


class BenchmarkTask(BaseModel):
    """One benchmark task fixture."""

    id: str
    title: str
    category: str
    difficulty: str = "easy"
    workspace: str
    prompt: str
    skills: list[str] = Field(default_factory=list)
    setup: BenchmarkSetup = Field(default_factory=BenchmarkSetup)
    oracle: BenchmarkOracle = Field(default_factory=BenchmarkOracle)
    constraints: BenchmarkConstraints = Field(default_factory=BenchmarkConstraints)
    expected: BenchmarkExpected = Field(default_factory=BenchmarkExpected)
    tags: list[str] = Field(default_factory=list)
    source_path: Path | None = None

    model_config = {"arbitrary_types_allowed": True}


class BenchmarkTaskResult(BaseModel):
    """Persisted result for one benchmark task."""

    id: str
    title: str
    category: str
    status: Literal["resolved", "failed", "error"]
    resolved: bool
    stop_reason: str | None = None
    steps: int = 0
    tool_calls: int = 0
    context_tokens: int = 0
    compression_count: int = 0
    checkpoint_count: int = 0
    read_tool_calls: int = 0
    unique_read_resources: int = 0
    repeated_read_calls: int = 0
    project_cache_hit_count: int = 0
    model_retry_count: int = 0
    reactive_compaction_count: int = 0
    output_recovery_count: int = 0
    subagent_call_count: int = 0
    mcp_tool_call_count: int = 0
    history_compaction_tokens_removed: int = 0
    history_compaction_tokens_retained: int = 0
    tool_calls_before_first_write: int | None = None
    unique_reads_before_first_write: int | None = None
    verification_calls_after_first_write: int = 0
    remaining_tool_calls_at_first_failed_verification: int | None = None
    final_test_passed: bool | None = None
    elapsed_seconds: float = 0.0
    oracle: dict[str, Any] = Field(default_factory=dict)
    final_text: str | None = None
    expected_files_changed: bool = True
    forbidden_file_changed: bool = False
    changed_files: list[str] = Field(default_factory=list)
    unexpected_modified_files: list[str] = Field(default_factory=list)
    modified_file_count: int = 0
    max_modified_files_exceeded: bool = False
    workspace: str
    output_dir: str
    trace_path: str
    final_diff_path: str
    error: str | None = None


class BenchmarkSummary(BaseModel):
    """Persisted summary for a benchmark suite run."""

    suite: str
    total_tasks: int
    resolved: int
    failed: int
    resolve_rate: float
    by_category: dict[str, dict[str, Any]]
    avg_steps: float
    avg_tool_calls: float
    avg_context_tokens: float
    compression_count: int
    checkpoint_count: int
    read_tool_calls: int = 0
    unique_read_resources: int = 0
    repeated_read_calls: int = 0
    project_cache_hit_count: int = 0
    model_retry_count: int = 0
    reactive_compaction_count: int = 0
    output_recovery_count: int = 0
    subagent_call_count: int = 0
    mcp_tool_call_count: int = 0
    history_compaction_tokens_removed: int = 0
    history_compaction_tokens_retained: int = 0
    avg_tool_calls_before_first_write: float = 0.0
    avg_unique_reads_before_first_write: float = 0.0
    verification_calls_after_first_write: int = 0
    avg_remaining_tool_calls_at_first_failed_verification: float = 0.0
    final_test_pass_rate: float = 0.0
    elapsed_seconds: float
    tasks: list[BenchmarkTaskResult]


def load_benchmark_tasks(suite_path: Path | str) -> list[BenchmarkTask]:
    """Load benchmark tasks from a suite directory."""

    suite = Path(suite_path)
    if not suite.is_dir():
        raise FileNotFoundError(f"Benchmark suite does not exist: {suite}")

    task_files = sorted(
        path
        for path in suite.rglob("*.yaml")
        if path.name not in {"task-template.yaml", "scenario-template.yaml"}
    )
    tasks: list[BenchmarkTask] = []
    for task_file in task_files:
        payload = yaml.safe_load(task_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Benchmark task must be a YAML mapping: {task_file}")
        task = BenchmarkTask.model_validate(payload)
        task.source_path = task_file
        tasks.append(task)

    if not tasks:
        raise ValueError(f"No benchmark tasks found under {suite}")
    return tasks


def write_json(path: Path, payload: BaseModel | dict[str, Any]) -> None:
    """Write formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
