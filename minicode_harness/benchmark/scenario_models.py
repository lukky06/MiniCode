"""Multi-turn benchmark schemas for memory and context-compaction evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
import yaml


from .models import (
    BenchmarkConstraints,
    BenchmarkExpected,
    BenchmarkOracle,
    BenchmarkSetup,
)


MemoryMode = Literal["off", "index_only", "index_topic"]
ContextCompactionMode = Literal["deterministic", "llm_hard"]
SemanticCompactionFault = Literal[
    "none",
    "invalid_json",
    "fabricated_execution",
]
EvaluationGroup = Literal["development", "blind"]
EvaluationTarget = Literal[
    "memory_lifecycle",
    "behavior_compliance",
    "code_correctness",
    "context_rationality",
]
FactLocation = Literal[
    "final_text",
    "semantic_history",
    "compacted_history",
    "historical_canonical_messages",
    "current_user",
    "canonical_messages",
    "trace",
]


class BenchmarkSeedMemory(BaseModel):
    """One stable scenario-local alias for a seeded long-term-memory item."""

    key: str
    content: str
    kind: Literal["preference", "workflow", "coding_style", "architecture"] = "preference"
    topic: Literal[
        "instructions",
        "build-and-test",
        "debugging",
        "decisions",
        "environment",
    ] | None = None
    origin: Literal["explicit", "inferred"] = "explicit"
    status: Literal["active", "inactive"] = "active"


class BenchmarkFactExpectation(BaseModel):
    """A deterministic keyword oracle for one fact that must survive a turn."""

    id: str
    keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    match: Literal["all", "any"] = "all"
    locations: list[FactLocation] = Field(
        default_factory=lambda: ["final_text", "compacted_history"]
    )
    context_compaction_modes: list[ContextCompactionMode] = Field(
        default_factory=list
    )
    exclude_current_user: bool = True

    @model_validator(mode="after")
    def validate_keywords(self) -> "BenchmarkFactExpectation":
        if not self.keywords and not self.forbidden_keywords:
            raise ValueError("fact expectations require required or forbidden keywords")
        return self


class BenchmarkToolCallExpectation(BaseModel):
    """Structured count expectation over normalized ``tool_called`` Trace events."""

    id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    memory_modes: list[MemoryMode] = Field(default_factory=list)
    min_count: int = Field(default=1, ge=0)
    max_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_count_range(self) -> "BenchmarkToolCallExpectation":
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("tool expectation max_count must be >= min_count")
        return self


class BenchmarkFileContentExpectation(BaseModel):
    """Deterministic final-workspace text expectation."""

    id: str
    path: str
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    must_exist: bool = True

    @model_validator(mode="after")
    def validate_keywords(self) -> "BenchmarkFileContentExpectation":
        if not self.required_keywords and not self.forbidden_keywords:
            raise ValueError("file-content expectations require required or forbidden keywords")
        return self

class BenchmarkRepositoryMemoryTopicExpectation(BaseModel):
    """Aggregated content Gold for one active Repository Memory Topic."""

    required_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)


class BenchmarkRepositoryMemoryExpectation(BaseModel):
    """Durable Repository Memory state expected after a scenario."""

    after_turn_id: str | None = None
    pending_reviews: int | None = Field(default=None, ge=0)
    pending_candidates: int | None = Field(default=None, ge=0)
    registered_topics: list[str] | None = None
    topics: dict[str, BenchmarkRepositoryMemoryTopicExpectation] = Field(
        default_factory=dict
    )
    required_events: list[str] = Field(default_factory=list)


class BenchmarkRepositoryMemoryCheck(BaseModel):
    """One deterministic Repository Memory Store-state check."""

    id: str
    passed: bool
    expected: Any = None
    actual: Any = None
    details: list[str] = Field(default_factory=list)

class BenchmarkContextQualityExpectation(BaseModel):
    """Scenario-level gates for judging context-management rationality."""

    gate_resolution: bool = False
    require_compaction: bool = False
    require_protocol_valid: bool = True
    require_active_user_anchor: bool = True
    forbid_compacted_execution_leaks: bool = True
    max_semantic_summaries_per_turn: int = Field(default=1, ge=0)
    max_semantic_summary_attempts: int | None = Field(default=None, ge=0)
    max_semantic_summary_failures: int | None = Field(default=None, ge=0)
    max_semantic_summary_contamination_count: int = Field(default=0, ge=0)
    max_prompt_budget_exceeded_count: int = Field(default=0, ge=0)
    max_reactive_compaction_count: int | None = Field(default=1, ge=0)
    max_hard_fallback_count: int | None = Field(default=None, ge=0)
    max_hard_fallback_turn_count: int | None = Field(default=None, ge=0)
    max_hard_fallbacks_per_turn: int | None = Field(default=None, ge=0)
    max_ineffective_hard_fallback_count: int | None = Field(default=0, ge=0)
    max_semantic_turn_deletion_count: int | None = Field(default=0, ge=0)
    min_placeholder_replacements: int = Field(default=0, ge=0)
    max_placeholder_replacements: int | None = Field(default=None, ge=0)
    min_cold_turn_compaction_count: int = Field(default=0, ge=0)
    min_cold_turns_compacted: int = Field(default=0, ge=0)
    min_removed_protocol_messages: int = Field(default=0, ge=0)
    min_semantic_summary_attempts: int = Field(default=0, ge=0)
    min_semantic_summary_successes: int = Field(default=0, ge=0)
    min_semantic_summary_failures: int = Field(default=0, ge=0)
    min_hard_fallback_count: int = Field(default=0, ge=0)
    required_semantic_summary_attempt_turn_ids: list[str] = Field(default_factory=list)
    required_semantic_summary_success_turn_ids: list[str] = Field(default_factory=list)
    required_semantic_summary_failure_turn_ids: list[str] = Field(default_factory=list)
    max_repeated_read_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_budget_usage_ratio: float | None = Field(default=None, ge=0.0)
    min_context_reduction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    min_semantic_history_fact_retention: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    min_historical_dialogue_fact_retention: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class BenchmarkContextQualityResult(BaseModel):
    """Aggregated context-quality outcome for one scenario run."""

    passed: bool = True
    failures: list[str] = Field(default_factory=list)
    protocol_error_count: int = 0
    active_user_anchor_failures: int = 0
    semantic_summary_count: int = 0
    semantic_summary_attempt_count: int = 0
    semantic_summary_success_count: int = 0
    semantic_summary_failure_count: int = 0
    semantic_summary_contamination_count: int = 0
    semantic_summary_input_tokens: int = 0
    semantic_summary_output_tokens: int = 0
    semantic_summary_duration_ms: int = 0
    cold_turn_compaction_count: int = 0
    cold_turns_compacted: int = 0
    removed_protocol_messages: int = 0
    retained_execution_fact_count: int = 0
    compacted_execution_leak_count: int = 0
    prompt_budget_exceeded_count: int = 0
    hard_fallback_count: int = 0
    hard_fallback_turn_count: int = 0
    max_hard_fallbacks_per_turn: int = 0
    ineffective_hard_fallback_count: int = 0
    semantic_turn_deletion_count: int = 0
    reactive_compaction_count: int = 0
    placeholder_replacements: int = 0
    repeated_read_rate: float = 0.0
    max_budget_usage_ratio: float = 0.0
    context_tokens_reduced: int = 0
    context_reduction_ratio: float = 0.0
    semantic_history_fact_retention: float | None = None
    historical_dialogue_fact_retention: float | None = None


class BenchmarkScenarioTurn(BaseModel):
    """One user turn in a shared workspace/session benchmark scenario."""

    id: str
    title: str | None = None
    prompt: str
    session_key: str = "default"
    execution_mode: Literal["agent", "memory_only"] = "agent"
    assistant_text: str | None = None
    expected_finalization_status: Literal[
        "deferred",
        "completed",
        "failed",
        "captured",
        "idempotent_noop",
    ] | None = None
    skills: list[str] = Field(default_factory=list)
    oracle: BenchmarkOracle = Field(default_factory=BenchmarkOracle)
    constraints: BenchmarkConstraints = Field(default_factory=BenchmarkConstraints)
    expected: BenchmarkExpected = Field(default_factory=BenchmarkExpected)
    expected_memory_keys: list[str] = Field(default_factory=list)
    forbidden_memory_keys: list[str] = Field(default_factory=list)
    expected_tool_calls: list[BenchmarkToolCallExpectation] = Field(default_factory=list)
    expected_file_contents: list[BenchmarkFileContentExpectation] = Field(
        default_factory=list
    )
    required_facts: list[BenchmarkFactExpectation] = Field(default_factory=list)


class BenchmarkLifecycle(BaseModel):
    """Producer-once lifecycle followed by isolated Consumer ablation forks."""

    producer_turn_ids: list[str] = Field(min_length=1)
    fork_after_turn_id: str
    consumer_turn_ids: list[str] = Field(min_length=1)
    consumer_ablation_modes: list[MemoryMode] = Field(min_length=1)
    fresh_consumer_session: bool = True
    expected_memory_entry_count: int | None = Field(default=None, ge=0)
    min_index_topic_reads: int = Field(default=0, ge=0)
    max_index_topic_reads: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_topic_read_bounds(self) -> "BenchmarkLifecycle":
        if (
            self.max_index_topic_reads is not None
            and self.max_index_topic_reads < self.min_index_topic_reads
        ):
            raise ValueError(
                "lifecycle max_index_topic_reads must be at least min_index_topic_reads"
            )
        return self


class BenchmarkScenario(BaseModel):
    """A multi-turn benchmark that shares workspace, session, and memory state."""

    id: str
    title: str
    category: str = "memory_context"
    difficulty: str = "medium"
    workspace: str
    setup: BenchmarkSetup = Field(default_factory=BenchmarkSetup)
    memory_mode: MemoryMode = "index_topic"
    ablation_modes: list[MemoryMode] = Field(default_factory=list)
    context_compaction_mode: ContextCompactionMode = "llm_hard"
    context_compaction_ablation_modes: list[ContextCompactionMode] = Field(
        default_factory=list
    )
    semantic_compaction_fault: SemanticCompactionFault = "none"
    evaluation_group: EvaluationGroup = "development"
    evaluation_target: EvaluationTarget = "behavior_compliance"
    tool_calls_gate_resolve: bool = True
    gold_version: str = "v1"
    gold_locked: bool = False
    recommended_repetitions: int = Field(default=5, ge=1, le=50)
    context_budget: int | None = Field(default=None, ge=2000)
    reserved_output: int = Field(default=1000, ge=256)
    seed_memories: list[BenchmarkSeedMemory] = Field(default_factory=list)
    lifecycle: BenchmarkLifecycle | None = None
    expected_repository_memory: BenchmarkRepositoryMemoryExpectation | None = None
    required_trace_events: list[str] = Field(default_factory=list)
    required_turn_ids: list[str] = Field(default_factory=list)
    context_quality: BenchmarkContextQualityExpectation = Field(
        default_factory=BenchmarkContextQualityExpectation
    )
    turns: list[BenchmarkScenarioTurn] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    source_path: Path | None = None
    suite_root: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_budget(self) -> "BenchmarkScenario":
        if self.context_budget is not None and self.reserved_output >= self.context_budget:
            raise ValueError("reserved_output must be smaller than context_budget")
        keys = [memory.key for memory in self.seed_memories]
        if len(keys) != len(set(keys)):
            raise ValueError("seed memory keys must be unique")
        turn_ids = [turn.id for turn in self.turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("scenario turn IDs must be unique")
        if len(self.required_turn_ids) != len(set(self.required_turn_ids)):
            raise ValueError("required_turn_ids must be unique")
        unknown_required_turns = sorted(set(self.required_turn_ids) - set(turn_ids))
        if unknown_required_turns:
            raise ValueError(
                "required_turn_ids reference unknown turns: "
                + ", ".join(unknown_required_turns)
            )
        if (
            self.expected_repository_memory is not None
            and self.expected_repository_memory.after_turn_id is not None
            and self.expected_repository_memory.after_turn_id not in set(turn_ids)
        ):
            raise ValueError(
                "expected_repository_memory.after_turn_id references an unknown turn"
            )
        if len(self.ablation_modes) != len(set(self.ablation_modes)):
            raise ValueError("ablation_modes must be unique")
        if self.lifecycle is not None:
            lifecycle = self.lifecycle
            producer_ids = lifecycle.producer_turn_ids
            consumer_ids = lifecycle.consumer_turn_ids
            if len(producer_ids) != len(set(producer_ids)):
                raise ValueError("lifecycle producer_turn_ids must be unique")
            if len(consumer_ids) != len(set(consumer_ids)):
                raise ValueError("lifecycle consumer_turn_ids must be unique")
            if set(producer_ids) & set(consumer_ids):
                raise ValueError("lifecycle producer and consumer turns must not overlap")
            unknown_lifecycle_turns = sorted(
                (set(producer_ids) | set(consumer_ids)) - set(turn_ids)
            )
            if unknown_lifecycle_turns:
                raise ValueError(
                    "lifecycle references unknown turns: "
                    + ", ".join(unknown_lifecycle_turns)
                )
            if set(producer_ids) | set(consumer_ids) != set(turn_ids):
                raise ValueError("lifecycle must classify every scenario turn")
            ordered_producers = [item for item in turn_ids if item in set(producer_ids)]
            ordered_consumers = [item for item in turn_ids if item in set(consumer_ids)]
            if producer_ids != ordered_producers or consumer_ids != ordered_consumers:
                raise ValueError("lifecycle turn IDs must follow scenario turn order")
            if turn_ids != [*producer_ids, *consumer_ids]:
                raise ValueError("all lifecycle Producer turns must precede Consumers")
            if lifecycle.fork_after_turn_id != producer_ids[-1]:
                raise ValueError("lifecycle fork_after_turn_id must be the last Producer turn")
            if len(lifecycle.consumer_ablation_modes) != len(
                set(lifecycle.consumer_ablation_modes)
            ):
                raise ValueError("lifecycle consumer_ablation_modes must be unique")
            if self.ablation_modes:
                raise ValueError(
                    "lifecycle consumer_ablation_modes replace scenario ablation_modes"
                )
            if self.seed_memories:
                raise ValueError("lifecycle scenarios must not declare seed_memories")
            if self.memory_mode != "index_topic":
                raise ValueError("lifecycle Producer memory_mode must be index_topic")
            if (
                self.expected_repository_memory is not None
                and self.expected_repository_memory.after_turn_id is not None
                and self.expected_repository_memory.after_turn_id
                != lifecycle.fork_after_turn_id
            ):
                raise ValueError(
                    "Repository Memory Gold must be evaluated at the lifecycle fork Turn"
                )
            if lifecycle.fresh_consumer_session:
                by_id = {turn.id: turn for turn in self.turns}
                producer_sessions = {
                    by_id[turn_id].session_key for turn_id in producer_ids
                }
                consumer_sessions = {
                    by_id[turn_id].session_key for turn_id in consumer_ids
                }
                if producer_sessions & consumer_sessions:
                    raise ValueError(
                        "fresh lifecycle Consumer sessions must not reuse Producer session keys"
                    )
        if len(self.context_compaction_ablation_modes) != len(
            set(self.context_compaction_ablation_modes)
        ):
            raise ValueError("context_compaction_ablation_modes must be unique")
        quality_turn_fields = (
            self.context_quality.required_semantic_summary_attempt_turn_ids,
            self.context_quality.required_semantic_summary_success_turn_ids,
            self.context_quality.required_semantic_summary_failure_turn_ids,
        )
        for required_ids in quality_turn_fields:
            unknown = sorted(set(required_ids) - set(turn_ids))
            if unknown:
                raise ValueError(
                    "context quality references unknown turns: "
                    + ", ".join(unknown)
                )
        if (
            self.semantic_compaction_fault != "none"
            and self.context_compaction_mode != "llm_hard"
            and "llm_hard" not in self.context_compaction_ablation_modes
        ):
            raise ValueError(
                "semantic_compaction_fault requires llm_hard compaction"
            )
        if self.evaluation_group == "blind" and not self.gold_locked:
            raise ValueError("blind scenarios must use gold_locked=true")
        return self


class BenchmarkScenarioVariant(BaseModel):
    """One memory/context ablation configuration."""

    id: str
    memory_mode: MemoryMode
    context_compaction_mode: ContextCompactionMode = "llm_hard"


class BenchmarkRecallResult(BaseModel):
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_memory_keys: list[str] = Field(default_factory=list)
    expected_memory_keys: list[str] = Field(default_factory=list)
    forbidden_memory_keys: list[str] = Field(default_factory=list)
    true_positive_count: int = 0
    selected_count: int = 0
    expected_count: int = 0
    precision: float = 1.0
    recall: float = 1.0
    f1: float = 1.0
    passed: bool = True

class BenchmarkToolCallCheck(BaseModel):
    id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    min_count: int = 1
    max_count: int | None = None
    matched_count: int = 0
    passed: bool = False


class BenchmarkFileContentCheck(BaseModel):
    id: str
    path: str
    matched: bool
    exists: bool
    missing_keywords: list[str] = Field(default_factory=list)
    forbidden_hits: list[str] = Field(default_factory=list)


class BenchmarkFactCheck(BaseModel):
    id: str
    matched: bool
    keywords: list[str]
    forbidden_keywords: list[str] = Field(default_factory=list)
    locations: list[FactLocation]
    matched_locations: dict[FactLocation, bool] = Field(default_factory=dict)
    forbidden_hits_by_location: dict[FactLocation, list[str]] = Field(
        default_factory=dict
    )
    exclude_current_user: bool = True


class BenchmarkScenarioTurnMetrics(BaseModel):
    steps: int = 0
    tool_calls: int = 0
    context_tokens_avg: int = 0
    context_tokens_max: int = 0
    compression_count: int = 0
    soft_compaction_count: int = 0
    cold_turn_compaction_count: int = 0
    cold_turns_compacted: int = 0
    removed_protocol_messages: int = 0
    retained_execution_fact_count: int = 0
    semantic_summary_count: int = 0
    semantic_summary_attempt_count: int = 0
    semantic_summary_success_count: int = 0
    semantic_summary_failure_count: int = 0
    semantic_summary_contamination_count: int = 0
    semantic_summary_input_tokens: int = 0
    semantic_summary_output_tokens: int = 0
    semantic_summary_duration_ms: int = 0
    hard_fallback_count: int = 0
    ineffective_hard_fallback_count: int = 0
    semantic_turn_deletion_count: int = 0
    reactive_compaction_count: int = 0
    placeholder_replacements: int = 0
    command_result_compaction_count: int = 0
    tool_result_preview_compaction_count: int = 0
    semantic_message_compaction_count: int = 0
    protocol_error_count: int = 0
    active_user_anchor_failures: int = 0
    compacted_execution_leak_count: int = 0
    prompt_budget_exceeded_count: int = 0
    context_tokens_before_compaction: int = 0
    context_tokens_reduced: int = 0
    context_reduction_ratio: float = 0.0
    max_budget_usage_ratio: float = 0.0
    consecutive_compaction_count: int = 0
    provider_cached_input_tokens: int = 0
    provider_cache_miss_input_tokens: int = 0
    read_tool_calls: int = 0
    unique_read_resources: int = 0
    repeated_read_calls: int = 0
    project_cache_hit_count: int = 0
    memory_topic_read_count: int = 0
    unique_memory_topics: int = 0
    repeated_memory_topic_reads: int = 0
    history_compaction_tokens_removed: int = 0
    history_compaction_tokens_retained: int = 0


class BenchmarkScenarioTurnResult(BaseModel):
    id: str
    title: str
    status: Literal["resolved", "failed", "error"]
    resolved: bool
    session_key: str = "default"
    session_id: str | None = None
    stop_reason: str | None = None
    final_text: str | None = None
    memory_finalization_status: str | None = None
    memory_pending_user_turns: int = 0
    memory_batch_user_turns: int = 0
    memory_finalization_passed: bool = True
    oracle: dict[str, Any] = Field(default_factory=dict)
    recall: BenchmarkRecallResult = Field(default_factory=BenchmarkRecallResult)
    tool_call_checks: list[BenchmarkToolCallCheck] = Field(default_factory=list)
    tool_calls_passed: bool = True
    file_content_checks: list[BenchmarkFileContentCheck] = Field(default_factory=list)
    file_contents_passed: bool = True
    fact_checks: list[BenchmarkFactCheck] = Field(default_factory=list)
    fact_retention_rate: float = 1.0
    final_text_fact_accuracy: float | None = None
    semantic_history_fact_retention: float | None = None
    compacted_history_fact_retention: float | None = None
    historical_dialogue_fact_retention: float | None = None
    facts_passed: bool = True
    expected_files_changed: bool = True
    forbidden_file_changed: bool = False
    changed_files: list[str] = Field(default_factory=list)
    unexpected_modified_files: list[str] = Field(default_factory=list)
    max_modified_files_exceeded: bool = False
    metrics: BenchmarkScenarioTurnMetrics = Field(default_factory=BenchmarkScenarioTurnMetrics)
    trace_path: str
    diff_path: str
    error: str | None = None


class BenchmarkScenarioResult(BaseModel):
    id: str
    title: str
    category: str
    evaluation_target: EvaluationTarget = "behavior_compliance"
    variant: BenchmarkScenarioVariant
    status: Literal["resolved", "failed", "error"]
    resolved: bool
    turn_count: int
    resolved_turns: int
    turn_resolve_rate: float | None
    required_turn_count: int = 0
    resolved_required_turns: int = 0
    required_turn_resolve_rate: float = 0.0
    required_turn_ids: list[str] = Field(default_factory=list)
    failed_required_turn_ids: list[str] = Field(default_factory=list)
    diagnostic_failed_turn_ids: list[str] = Field(default_factory=list)
    recall_precision: float | None
    recall_recall: float | None
    recall_f1: float | None
    tool_compliance_rate: float | None = 1.0
    file_content_accuracy: float = 1.0
    fact_retention_rate: float
    final_text_fact_accuracy: float | None = None
    compacted_history_fact_retention: float | None = None
    historical_dialogue_fact_retention: float | None = None
    avg_context_tokens: float | None
    history_compression_ratio: float
    repeated_read_rate: float
    memory_topic_read_count: int = 0
    unique_memory_topics: int = 0
    repeated_memory_topic_reads: int = 0
    compaction_count: int = 0
    consecutive_compaction_count: int = 0
    stable_append_turns: int = 0
    provider_cached_input_tokens: int = 0
    provider_cache_miss_input_tokens: int = 0
    execution_fact_retention: float | None = None
    semantic_history_fact_retention: float | None = None
    context_quality: BenchmarkContextQualityResult = Field(
        default_factory=BenchmarkContextQualityResult
    )
    repository_memory_checks: list[BenchmarkRepositoryMemoryCheck] = Field(
        default_factory=list
    )
    repository_memory_passed: bool = True
    producer_succeeded: bool | None = None
    capture_passed: bool | None = None
    consolidation_passed: bool | None = None
    consolidation_protocol_passed: bool | None = None
    cross_session_isolation_passed: bool | None = None
    recall_passed: bool | None = None
    hidden_correctness_passed: bool | None = None
    snapshot_integrity_passed: bool | None = None
    unconditional_e2e_success: bool | None = None
    consumer_correctness_given_producer_succeeded: bool | None = None
    producer_tool_calls: int = 0
    producer_memory_entry_count: int = 0
    expected_producer_memory_entry_count: int = 0
    producer_avg_context_tokens: float = 0.0
    producer_elapsed_seconds: float = 0.0
    producer_workspace_hash: str | None = None
    producer_memory_snapshot_hash: str | None = None
    consumer_start_workspace_hash: str | None = None
    consumer_start_memory_snapshot_hash: str | None = None
    durable_memory_hash_before: str | None = None
    durable_memory_hash_after: str | None = None
    elapsed_seconds: float
    memory_aliases: dict[str, str] = Field(default_factory=dict)
    turns: list[BenchmarkScenarioTurnResult] = Field(default_factory=list)
    workspace: str
    output_dir: str
    final_diff_path: str
    required_trace_events: list[str] = Field(default_factory=list)
    missing_trace_events: list[str] = Field(default_factory=list)
    error: str | None = None


class BenchmarkVariantSummary(BaseModel):
    variant: BenchmarkScenarioVariant
    scenario_count: int
    resolve_rate: float
    turn_resolve_rate: float | None
    recall_precision: float | None
    recall_recall: float | None
    recall_f1: float | None
    tool_compliance_rate: float | None = 1.0
    file_content_accuracy: float = 1.0
    fact_retention_rate: float
    final_text_fact_accuracy: float | None = None
    compacted_history_fact_retention: float | None = None
    historical_dialogue_fact_retention: float | None = None
    avg_context_tokens: float | None
    history_compression_ratio: float
    repeated_read_rate: float
    memory_topic_read_count: int = 0
    unique_memory_topics: int = 0
    repeated_memory_topic_reads: int = 0
    compaction_count: int = 0
    consecutive_compaction_count: int = 0
    stable_append_turns: int = 0
    provider_cached_input_tokens: int = 0
    provider_cache_miss_input_tokens: int = 0
    execution_fact_retention: float | None = None
    semantic_history_fact_retention: float | None = None
    context_quality_pass_rate: float = 1.0
    semantic_summary_attempt_count: int = 0
    semantic_summary_success_count: int = 0
    semantic_summary_failure_count: int = 0
    semantic_summary_contamination_count: int = 0
    semantic_summary_input_tokens: int = 0
    semantic_summary_output_tokens: int = 0
    semantic_summary_duration_ms: int = 0
    cold_turn_compaction_count: int = 0
    cold_turns_compacted: int = 0
    removed_protocol_messages: int = 0
    retained_execution_fact_count: int = 0
    hard_fallback_count: int = 0
    hard_fallback_turn_count: int = 0
    max_hard_fallbacks_per_turn: int = 0
    ineffective_hard_fallback_count: int = 0
    semantic_turn_deletion_count: int = 0
    reactive_compaction_count: int = 0
    placeholder_replacements: int = 0
    protocol_error_count: int = 0
    active_user_anchor_failures: int = 0
    compacted_execution_leak_count: int = 0
    prompt_budget_exceeded_count: int = 0
    max_budget_usage_ratio: float = 0.0
    context_token_delta_vs_baseline: float = 0.0
    resolve_rate_delta_vs_baseline: float = 0.0
    fact_retention_delta_vs_baseline: float = 0.0
    code_correctness_scenario_count: int = 0
    code_correctness_resolve_rate: float | None = None
    code_correctness_delta_vs_baseline: float = 0.0
    paired_code_correctness_scenario_count: int = 0
    positive_transfer_count: int = 0
    negative_transfer_count: int = 0
    negative_transfer_rate: float = 0.0
    unconditional_e2e_success_rate: float | None = None
    consumer_correctness_given_producer_succeeded: float | None = None


class BenchmarkScenarioSummary(BaseModel):
    suite: str
    total_runs: int
    resolved_runs: int
    resolve_rate: float
    elapsed_seconds: float
    variants: list[BenchmarkVariantSummary]
    scenarios: list[BenchmarkScenarioResult]


def load_benchmark_scenarios(suite_path: Path | str) -> list[BenchmarkScenario]:
    """Load multi-turn scenario YAML files from one suite directory."""

    suite = Path(suite_path)
    if not suite.is_dir():
        raise FileNotFoundError(f"Benchmark scenario suite does not exist: {suite}")
    scenario_files = sorted(
        path
        for path in suite.rglob("*.yaml")
        if path.name not in {"scenario-template.yaml", "task-template.yaml"}
    )
    scenarios: list[BenchmarkScenario] = []
    for scenario_file in scenario_files:
        payload = yaml.safe_load(scenario_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "turns" not in payload:
            continue
        payload = _normalize_yaml_modes(payload)
        scenario = BenchmarkScenario.model_validate(payload)
        scenario.source_path = scenario_file
        scenario.suite_root = suite.resolve()
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"No benchmark scenarios found under {suite}")
    return scenarios


def _normalize_yaml_modes(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize YAML 1.1's unquoted ``off`` boolean coercion."""

    normalized = dict(payload)
    if normalized.get("memory_mode") is False:
        normalized["memory_mode"] = "off"
    modes = normalized.get("ablation_modes")
    if isinstance(modes, list):
        normalized["ablation_modes"] = [
            "off" if mode is False else mode
            for mode in modes
        ]
    lifecycle = normalized.get("lifecycle")
    if isinstance(lifecycle, dict):
        normalized_lifecycle = dict(lifecycle)
        consumer_modes = normalized_lifecycle.get("consumer_ablation_modes")
        if isinstance(consumer_modes, list):
            normalized_lifecycle["consumer_ablation_modes"] = [
                "off" if mode is False else mode for mode in consumer_modes
            ]
        normalized["lifecycle"] = normalized_lifecycle
    return normalized
