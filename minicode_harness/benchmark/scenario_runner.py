"""Shared-session benchmark runner for memory and context ablation studies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from minicode_harness.context import (
    ContextBuilder,
    ContextPreparer,
    LLMSemanticHistoryCompactor,
    TokenBudget,
    validate_message_protocol,
)
from minicode_harness.loop import AgentLoop, AgentLoopConfig, AgentRunResult
from minicode_harness.memory import (
    MemorySnapshotStore,
    RepositoryMemorySnapshotSource,
    RepositoryMemoryStore,
)
from minicode_harness.models import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    create_model_client,
)
from minicode_harness.report import collect_context_metrics
from minicode_harness.runtime.request_orchestrator import RequestOrchestrator
from minicode_harness.state import (
    ApprovalStore,
    CheckpointStore,
    ReplSessionMemory,
    ReplSessionStore,
)
from minicode_harness.trace import TraceWriter

from .models import BenchmarkConstraints, write_json
from .runner import (
    BENCHMARK_IGNORED_NAMES,
    BENCHMARK_IGNORED_PATTERNS,
    BenchmarkApprovalClient,
    _changed_paths_from_diff,
    _evaluate_oracle,
    _initialize_benchmark_git_baseline,
    _remove_tree,
    _render_benchmark_task,
    _expected_files_changed,
    _forbidden_file_changed,
    _run_setup_command,
    _text_files,
    _unexpected_modified_files,
    _write_final_diff,
)
from .scenario_models import (
    BenchmarkContextQualityExpectation,
    BenchmarkContextQualityResult,
    BenchmarkFactCheck,
    BenchmarkFactExpectation,
    BenchmarkFileContentCheck,
    BenchmarkFileContentExpectation,
    BenchmarkRecallResult,
    BenchmarkRepositoryMemoryCheck,
    BenchmarkRepositoryMemoryExpectation,
    BenchmarkScenario,
    BenchmarkScenarioResult,
    BenchmarkScenarioSummary,
    BenchmarkScenarioTurn,
    BenchmarkScenarioTurnMetrics,
    BenchmarkScenarioTurnResult,
    BenchmarkScenarioVariant,
    BenchmarkToolCallCheck,
    BenchmarkToolCallExpectation,
    BenchmarkVariantSummary,
    ContextCompactionMode,
    MemoryMode,
    load_benchmark_scenarios,
)


ScenarioModelClientFactory = Callable[
    [BenchmarkScenario, BenchmarkScenarioVariant],
    ModelClient,
]


_ABLATION_VARIANTS = (
    BenchmarkScenarioVariant(id="off", memory_mode="off"),
    BenchmarkScenarioVariant(id="index_only", memory_mode="index_only"),
    BenchmarkScenarioVariant(id="index_topic", memory_mode="index_topic"),
)


class _SemanticCompactionFaultClient(ModelClient):
    """Inject a summary-only fault without changing the production compactor."""

    def __init__(self, delegate: ModelClient, *, fault: str) -> None:
        self.delegate = delegate
        self.fault = fault
        self.capabilities = delegate.capabilities

    def call_request(self, request: ModelRequest) -> ModelResponse:
        if request.metadata.get("purpose") != "semantic_history_compaction":
            return self.delegate.call_request(request)
        if self.fault == "invalid_json":
            return ModelResponse(final_text="{invalid semantic compaction json")
        if self.fault == "fabricated_execution":
            payload = json.loads(str(request.messages[0].get("content") or "{}"))
            source_ids = payload.get("compressible_source_turn_ids") or []
            source_id = str(source_ids[0]) if source_ids else "t0001"
            return ModelResponse(
                final_text=json.dumps(
                    {
                        "items": [
                            {
                                "kind": "fact",
                                "text": (
                                    "已修改 tests/test_fabricated.py，"
                                    "测试已通过。"
                                ),
                                "source_turn_ids": [source_id],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        return self.delegate.call_request(request)


def _semantic_compaction_client(
    model_client: ModelClient,
    *,
    fault: str,
) -> ModelClient:
    if fault == "none":
        return model_client
    return _SemanticCompactionFaultClient(model_client, fault=fault)


@dataclass(frozen=True)
class BenchmarkScenarioRunnerConfig:
    """Runtime and ablation settings for multi-turn benchmark scenarios."""

    provider: str = "openai"
    model: str | None = None
    max_steps: int = 50
    max_tool_calls: int = 60
    ablation_matrix: bool = False
    memory_mode: MemoryMode | None = None
    context_compaction_mode: ContextCompactionMode | None = None
    scenario_id: str | None = None
    context_budget_override: int | None = None
    baseline_mode: bool = False


class BenchmarkScenarioRunner:
    """Run multi-turn scenarios with shared session and convention state."""

    def __init__(
        self,
        config: BenchmarkScenarioRunnerConfig | None = None,
        *,
        model_client_factory: ScenarioModelClientFactory | None = None,
    ) -> None:
        self.config = config or BenchmarkScenarioRunnerConfig()
        self.model_client_factory = model_client_factory or self._create_model_client

    def run_suite(
        self,
        suite: Path | str,
        output: Path | str,
    ) -> BenchmarkScenarioSummary:
        suite_path = Path(suite).resolve()
        output_path = Path(output).resolve()
        scenarios = load_benchmark_scenarios(suite_path)
        if self.config.scenario_id is not None:
            scenarios = [
                scenario
                for scenario in scenarios
                if scenario.id == self.config.scenario_id
            ]
            if not scenarios:
                raise ValueError(
                    f"Unknown scenario id: {self.config.scenario_id}"
                )
        output_path.mkdir(parents=True, exist_ok=True)

        started_at = time.monotonic()
        results: list[BenchmarkScenarioResult] = []
        for scenario in scenarios:
            variants = self._variants_for(scenario)
            if scenario.lifecycle is not None:
                results.extend(
                    self._run_lifecycle_scenario(
                        scenario,
                        variants=variants,
                        suite_path=suite_path,
                        output_path=output_path,
                    )
                )
                continue
            for variant in variants:
                results.append(
                    self._run_scenario(
                        scenario,
                        variant=variant,
                        suite_path=suite_path,
                        output_path=output_path,
                    )
                )
        summary = _build_scenario_summary(
            suite=suite_path.name,
            results=results,
            elapsed_seconds=time.monotonic() - started_at,
        )
        write_json(output_path / "summary.json", summary)
        _write_scenario_report(output_path / "benchmark_report.md", summary)
        return summary

    def _variants_for(self, scenario: BenchmarkScenario) -> list[BenchmarkScenarioVariant]:
        if scenario.lifecycle is not None:
            if self.config.ablation_matrix:
                raise ValueError(
                    "The global ablation matrix is incompatible with lifecycle scenarios."
                )
            memory_modes = (
                [self.config.memory_mode]
                if self.config.memory_mode is not None
                else list(scenario.lifecycle.consumer_ablation_modes)
            )
            compaction_mode = (
                self.config.context_compaction_mode
                or scenario.context_compaction_mode
            )
            return [
                BenchmarkScenarioVariant(
                    id=memory_mode,
                    memory_mode=memory_mode,
                    context_compaction_mode=compaction_mode,
                )
                for memory_mode in memory_modes
            ]
        if self.config.ablation_matrix:
            compaction_mode = (
                self.config.context_compaction_mode
                or scenario.context_compaction_mode
            )
            return [
                variant.model_copy(
                    update={"context_compaction_mode": compaction_mode}
                )
                for variant in _ABLATION_VARIANTS
            ]

        if self.config.memory_mode is not None:
            memory_modes = [self.config.memory_mode]
        elif scenario.ablation_modes:
            memory_modes = list(scenario.ablation_modes)
        else:
            memory_modes = [scenario.memory_mode]

        if self.config.context_compaction_mode is not None:
            compaction_modes = [self.config.context_compaction_mode]
        elif scenario.context_compaction_ablation_modes:
            compaction_modes = list(scenario.context_compaction_ablation_modes)
        else:
            compaction_modes = [scenario.context_compaction_mode]

        variants: list[BenchmarkScenarioVariant] = []
        compaction_varied = (
            len(compaction_modes) > 1
            or self.config.context_compaction_mode is not None
            or bool(scenario.context_compaction_ablation_modes)
        )
        for memory_mode in memory_modes:
            for compaction_mode in compaction_modes:
                if len(memory_modes) == 1 and compaction_varied:
                    variant_id = compaction_mode
                elif len(compaction_modes) == 1:
                    variant_id = memory_mode
                else:
                    variant_id = f"{memory_mode}-{compaction_mode}"
                variants.append(
                    BenchmarkScenarioVariant(
                        id=variant_id,
                        memory_mode=memory_mode,
                        context_compaction_mode=compaction_mode,
                    )
                )
        return variants

    def _run_lifecycle_scenario(
        self,
        scenario: BenchmarkScenario,
        *,
        variants: list[BenchmarkScenarioVariant],
        suite_path: Path,
        output_path: Path,
    ) -> list[BenchmarkScenarioResult]:
        """Run Producers once, then fork identical durable state per Consumer mode."""

        lifecycle = scenario.lifecycle
        assert lifecycle is not None
        scenario_root = output_path / "scenarios" / scenario.id
        if scenario_root.exists():
            _remove_tree(scenario_root)
        producer_output = scenario_root / "producer"
        producer_output.mkdir(parents=True)

        source_workspace = _resolve_scenario_workspace(scenario, suite_path)
        producer_workspace = producer_output / "workspace"
        _copy_scenario_workspace(
            source_workspace,
            producer_workspace,
            include_git=False,
        )
        setup_trace = TraceWriter(producer_output / "setup-trace.jsonl")
        producer_turns = [
            turn
            for turn in scenario.turns
            if turn.id in set(lifecycle.producer_turn_ids)
        ]
        consumer_turns = [
            turn
            for turn in scenario.turns
            if turn.id in set(lifecycle.consumer_turn_ids)
        ]
        for index, command in enumerate(scenario.setup.commands, start=1):
            _run_setup_command(
                command=command,
                constraints=_combined_constraints(scenario.turns),
                workspace=producer_workspace,
                trace_writer=setup_trace,
                step=index,
            )
        _initialize_benchmark_git_baseline(producer_workspace, setup_trace)

        producer_memory_data = producer_output / "memory-v2-data"
        producer_repository_memory = RepositoryMemoryStore(
            producer_workspace,
            data_dir=producer_memory_data,
        )
        producer_session_store = ReplSessionStore(
            producer_output / "session-data"
        )
        producer_sessions: dict[str, ReplSessionMemory] = {}
        producer_variant = BenchmarkScenarioVariant(
            id="producer",
            memory_mode="index_topic",
            context_compaction_mode=(
                self.config.context_compaction_mode
                or scenario.context_compaction_mode
            ),
        )
        producer_model = self.model_client_factory(scenario, producer_variant)
        producer_results: list[BenchmarkScenarioTurnResult] = []
        repository_memory_checks: list[BenchmarkRepositoryMemoryCheck] = []
        producer_error: str | None = None
        producer_started_at = time.monotonic()
        try:
            for turn_index, turn in enumerate(producer_turns, start=1):
                session_memory = producer_sessions.get(turn.session_key)
                if session_memory is None:
                    session_memory = producer_session_store.create(producer_workspace)
                    producer_sessions[turn.session_key] = session_memory
                producer_results.append(
                    self._run_turn(
                        scenario,
                        turn,
                        turn_index=turn_index,
                        variant=producer_variant,
                        workspace=producer_workspace,
                        scenario_output=producer_output,
                        model_client=producer_model,
                        repository_memory=producer_repository_memory,
                        session_memory=session_memory,
                        memory_aliases={},
                    )
                )
            producer_orchestrator = RequestOrchestrator(
                repository_memory=producer_repository_memory,
                review_model_client=producer_model,
            )
            for _ in range(8):
                if not producer_repository_memory.workflow_store.pending_reviews():
                    break
                _, reviewed_turns, _, _ = producer_orchestrator.review_pending(force=True)
                if reviewed_turns == 0:
                    break
        except Exception as exc:
            producer_error = f"{type(exc).__name__}: {exc}"
        producer_elapsed_seconds = time.monotonic() - producer_started_at

        if not repository_memory_checks:
            repository_memory_checks = _score_repository_memory(
                scenario.expected_repository_memory,
                repository_memory=producer_repository_memory,
            )
        repository_memory_passed = all(
            check.passed for check in repository_memory_checks
        )
        producer_memory_entry_count = sum(
            len(producer_repository_memory.topic_store.active_entries(topic))
            for topic in producer_repository_memory.topic_store.registered_topics()
        )
        expected_memory_entry_count = lifecycle.expected_memory_entry_count
        capture_passed = (
            len(producer_results) == len(producer_turns)
            and (
                expected_memory_entry_count is None
                or producer_memory_entry_count == expected_memory_entry_count
            )
            and all(
                turn.tool_calls_passed
                and not turn.changed_files
                and turn.error is None
                for turn in producer_results
            )
        )
        actual_finalization_statuses = [
            turn.memory_finalization_status for turn in producer_results
        ]
        consolidation_protocol_passed = (
            bool(actual_finalization_statuses)
            and all(status == "review_recorded" for status in actual_finalization_statuses)
            and not producer_repository_memory.workflow_store.pending_reviews()
            and not producer_repository_memory.workflow_store.list_candidates(
                status="pending"
            )
        )
        producer_succeeded = (
            producer_error is None
            and capture_passed
            and consolidation_protocol_passed
            and repository_memory_passed
        )
        producer_workspace_hash = _directory_manifest_hash(
            producer_workspace,
            ignored_patterns=BENCHMARK_IGNORED_PATTERNS,
        )
        producer_memory_snapshot_hash = _directory_manifest_hash(
            producer_memory_data
        )
        producer_durable_hash = _directory_manifest_hash(
            producer_memory_data,
            ignored_suffixes=(
                "/events.jsonl",
                "/review-inbox.jsonl",
                "/review-state.json",
            ),
        )
        producer_session_ids = {
            session.session_id for session in producer_sessions.values()
        }
        producer_prompts = [turn.prompt for turn in producer_turns]
        producer_tool_calls = sum(turn.metrics.tool_calls for turn in producer_results)
        producer_avg_context_tokens = _average(
            [turn.metrics.context_tokens_avg for turn in producer_results]
        )

        results: list[BenchmarkScenarioResult] = []
        for variant in variants:
            variant_started_at = time.monotonic()
            variant_output = scenario_root / variant.id
            variant_output.mkdir(parents=True)
            workspace = variant_output / "workspace"
            memory_data = variant_output / "memory-v2-data"
            turn_results: list[BenchmarkScenarioTurnResult] = []
            variant_error: str | None = None
            consumer_start_workspace_hash: str | None = None
            consumer_start_memory_snapshot_hash: str | None = None
            durable_memory_hash_before: str | None = None
            durable_memory_hash_after: str | None = None
            cross_session_isolation_passed: bool | None = None
            snapshot_integrity_passed: bool | None = None

            if producer_succeeded:
                _copy_scenario_workspace(
                    producer_workspace,
                    workspace,
                    include_git=True,
                )
                shutil.copytree(producer_memory_data, memory_data)
                consumer_start_workspace_hash = _directory_manifest_hash(
                    workspace,
                    ignored_patterns=BENCHMARK_IGNORED_PATTERNS,
                )
                consumer_start_memory_snapshot_hash = _directory_manifest_hash(
                    memory_data
                )
                durable_memory_hash_before = _directory_manifest_hash(
                    memory_data,
                    ignored_suffixes=(
                        "/events.jsonl",
                        "/review-inbox.jsonl",
                        "/review-state.json",
                    ),
                )
                snapshot_integrity_passed = (
                    consumer_start_workspace_hash == producer_workspace_hash
                    and consumer_start_memory_snapshot_hash
                    == producer_memory_snapshot_hash
                    and durable_memory_hash_before == producer_durable_hash
                )
                if not snapshot_integrity_passed:
                    variant_error = "Lifecycle snapshot hash mismatch before Consumer run."
                repository_memory = RepositoryMemoryStore(
                    workspace,
                    data_dir=memory_data,
                )
                if (
                    repository_memory.repository_id
                    != producer_repository_memory.repository_id
                ):
                    snapshot_integrity_passed = False
                    variant_error = "Repository identity changed across lifecycle fork."
                session_store = ReplSessionStore(variant_output / "session-data")
                session_memories: dict[str, ReplSessionMemory] = {}
                if variant_error is None:
                    model_client = self.model_client_factory(scenario, variant)
                    try:
                        for turn_index, turn in enumerate(consumer_turns, start=1):
                            session_memory = session_memories.get(turn.session_key)
                            if session_memory is None:
                                session_memory = session_store.create(workspace)
                                session_memories[turn.session_key] = session_memory
                            turn_results.append(
                                self._run_turn(
                                    scenario,
                                    turn,
                                    turn_index=turn_index,
                                    variant=variant,
                                    workspace=workspace,
                                    scenario_output=variant_output,
                                    model_client=model_client,
                                    repository_memory=repository_memory,
                                    session_memory=session_memory,
                                    memory_aliases={},
                                )
                            )
                    except Exception as exc:
                        variant_error = f"{type(exc).__name__}: {exc}"
                durable_memory_hash_after = _directory_manifest_hash(
                    memory_data,
                    ignored_suffixes=(
                        "/events.jsonl",
                        "/review-inbox.jsonl",
                        "/review-state.json",
                    ),
                )
                snapshot_integrity_passed = (
                    snapshot_integrity_passed
                    and durable_memory_hash_before == durable_memory_hash_after
                )
                cross_session_isolation_passed = _consumer_sessions_are_isolated(
                    session_memories.values(),
                    producer_session_ids=producer_session_ids,
                    producer_prompts=producer_prompts,
                )
            else:
                _copy_scenario_workspace(
                    producer_workspace,
                    workspace,
                    include_git=True,
                )
                variant_error = producer_error or "Producer lifecycle gate failed."

            final_diff_path = variant_output / "final.diff"
            _write_final_diff(producer_workspace, workspace, final_diff_path)
            result = _build_scenario_result(
                scenario=scenario,
                variant=variant,
                turn_results=turn_results,
                elapsed_seconds=(
                    producer_elapsed_seconds
                    + (time.monotonic() - variant_started_at)
                ),
                memory_aliases={},
                workspace=workspace,
                output_dir=variant_output,
                final_diff_path=final_diff_path,
                error=variant_error,
                enforce_required_trace_events=not self.config.baseline_mode,
            )
            recall_passed = (
                all(turn.tool_calls_passed for turn in turn_results)
                if turn_results
                else None
            )
            if recall_passed and variant.memory_mode == "index_topic":
                topic_reads = sum(
                    turn.metrics.memory_topic_read_count for turn in turn_results
                )
                recall_passed = (
                    topic_reads >= lifecycle.min_index_topic_reads
                    and (
                        lifecycle.max_index_topic_reads is None
                        or topic_reads <= lifecycle.max_index_topic_reads
                    )
                )
            hidden_correctness_passed = (
                all(
                    turn.error is None
                    and bool(turn.oracle.get("passed"))
                    and turn.expected_files_changed
                    and not turn.forbidden_file_changed
                    and not turn.unexpected_modified_files
                    and not turn.max_modified_files_exceeded
                    for turn in turn_results
                )
                if turn_results
                else None
            )
            if not turn_results:
                cross_session_isolation_passed = None
            unconditional_e2e_success = bool(
                producer_succeeded
                and snapshot_integrity_passed
                and cross_session_isolation_passed
                and recall_passed
                and hidden_correctness_passed
            )
            consumer_correctness = (
                hidden_correctness_passed if producer_succeeded else None
            )
            result = result.model_copy(
                update={
                    "status": "resolved" if unconditional_e2e_success else "failed",
                    "resolved": unconditional_e2e_success,
                    "repository_memory_checks": repository_memory_checks,
                    "repository_memory_passed": repository_memory_passed,
                    "producer_succeeded": producer_succeeded,
                    "capture_passed": capture_passed,
                    "consolidation_passed": consolidation_protocol_passed,
                    "consolidation_protocol_passed": consolidation_protocol_passed,
                    "cross_session_isolation_passed": cross_session_isolation_passed,
                    "recall_passed": recall_passed,
                    "hidden_correctness_passed": hidden_correctness_passed,
                    "snapshot_integrity_passed": snapshot_integrity_passed,
                    "unconditional_e2e_success": unconditional_e2e_success,
                    "consumer_correctness_given_producer_succeeded": consumer_correctness,
                    "producer_tool_calls": producer_tool_calls,
                    "producer_memory_entry_count": producer_memory_entry_count,
                    "expected_producer_memory_entry_count": expected_memory_entry_count,
                    "producer_avg_context_tokens": producer_avg_context_tokens,
                    "producer_elapsed_seconds": producer_elapsed_seconds,
                    "producer_workspace_hash": producer_workspace_hash,
                    "producer_memory_snapshot_hash": producer_memory_snapshot_hash,
                    "consumer_start_workspace_hash": consumer_start_workspace_hash,
                    "consumer_start_memory_snapshot_hash": consumer_start_memory_snapshot_hash,
                    "durable_memory_hash_before": durable_memory_hash_before,
                    "durable_memory_hash_after": durable_memory_hash_after,
                    **(
                        {}
                        if turn_results
                        else {
                            "turn_resolve_rate": None,
                            "recall_precision": None,
                            "recall_recall": None,
                            "recall_f1": None,
                            "tool_compliance_rate": None,
                            "avg_context_tokens": None,
                        }
                    ),
                }
            )
            write_json(variant_output / "result.json", result)
            results.append(result)
        write_json(
            producer_output / "producer-result.json",
            {
                "scenario_id": scenario.id,
                "producer_succeeded": producer_succeeded,
                "capture_passed": capture_passed,
                "producer_memory_entry_count": producer_memory_entry_count,
                "expected_producer_memory_entry_count": expected_memory_entry_count,
                "consolidation_passed": consolidation_protocol_passed,
                "consolidation_protocol_passed": consolidation_protocol_passed,
                "repository_memory_passed": repository_memory_passed,
                "repository_memory_checks": [
                    check.model_dump(mode="json")
                    for check in repository_memory_checks
                ],
                "workspace_hash": producer_workspace_hash,
                "memory_snapshot_hash": producer_memory_snapshot_hash,
                "durable_memory_hash": producer_durable_hash,
                "session_ids": sorted(producer_session_ids),
                "turns": [turn.model_dump(mode="json") for turn in producer_results],
                "error": producer_error,
            },
        )
        return results

    def _run_scenario(
        self,
        scenario: BenchmarkScenario,
        *,
        variant: BenchmarkScenarioVariant,
        suite_path: Path,
        output_path: Path,
    ) -> BenchmarkScenarioResult:
        scenario_output = output_path / "scenarios" / scenario.id / variant.id
        if scenario_output.exists():
            _remove_tree(scenario_output)
        scenario_output.mkdir(parents=True)

        source_workspace = _resolve_scenario_workspace(scenario, suite_path)
        workspace = scenario_output / "workspace"
        shutil.copytree(
            source_workspace,
            workspace,
            ignore=shutil.ignore_patterns(
                "target",
                ".git",
                "__pycache__",
                "runs",
                ".idea",
                ".vscode",
            ),
        )
        model_client = self.model_client_factory(scenario, variant)
        repository_memory = RepositoryMemoryStore(
            workspace,
            data_dir=scenario_output / "memory-v2-data",
        )
        session_store = ReplSessionStore(scenario_output / "session-data")
        session_memories: dict[str, ReplSessionMemory] = {}
        memory_aliases = _seed_memories(
            scenario,
            repository_memory=repository_memory,
        )

        started_at = time.monotonic()
        scenario_error: str | None = None
        turn_results: list[BenchmarkScenarioTurnResult] = []
        repository_memory_checks: list[BenchmarkRepositoryMemoryCheck] = []
        try:
            setup_trace = TraceWriter(scenario_output / "setup-trace.jsonl")
            for index, command in enumerate(scenario.setup.commands, start=1):
                _run_setup_command(
                    command=command,
                    constraints=_combined_constraints(scenario.turns),
                    workspace=workspace,
                    trace_writer=setup_trace,
                    step=index,
                )
            _initialize_benchmark_git_baseline(workspace, setup_trace)
            for turn_index, turn in enumerate(scenario.turns, start=1):
                session_memory = session_memories.get(turn.session_key)
                if session_memory is None:
                    session_memory = session_store.create(workspace)
                    session_memories[turn.session_key] = session_memory
                turn_result = self._run_turn(
                    scenario,
                    turn,
                    turn_index=turn_index,
                    variant=variant,
                    workspace=workspace,
                    scenario_output=scenario_output,
                    model_client=model_client,
                    repository_memory=repository_memory,
                    session_memory=session_memory,
                    memory_aliases=memory_aliases,
                )
                turn_results.append(turn_result)
                expectation = scenario.expected_repository_memory
                if expectation is not None and expectation.after_turn_id == turn.id:
                    repository_memory_checks = _score_repository_memory(
                        expectation,
                        repository_memory=repository_memory,
                    )
        except Exception as exc:
            scenario_error = f"{type(exc).__name__}: {exc}"

        final_diff_path = scenario_output / "final.diff"
        _write_final_diff(source_workspace, workspace, final_diff_path)
        result = _build_scenario_result(
            scenario=scenario,
            variant=variant,
            turn_results=turn_results,
            elapsed_seconds=time.monotonic() - started_at,
            memory_aliases=memory_aliases,
            workspace=workspace,
            output_dir=scenario_output,
            final_diff_path=final_diff_path,
            error=scenario_error,
            enforce_required_trace_events=not self.config.baseline_mode,
        )
        if (
            scenario.expected_repository_memory is not None
            and not repository_memory_checks
        ):
            repository_memory_checks = _score_repository_memory(
                scenario.expected_repository_memory,
                repository_memory=repository_memory,
            )
        repository_memory_passed = all(
            check.passed for check in repository_memory_checks
        )
        result = result.model_copy(
            update={
                "resolved": result.resolved and repository_memory_passed,
                "status": (
                    result.status
                    if repository_memory_passed
                    else "failed"
                ),
                "repository_memory_checks": repository_memory_checks,
                "repository_memory_passed": repository_memory_passed,
            }
        )
        write_json(scenario_output / "result.json", result)
        return result

    def _run_turn(
        self,
        scenario: BenchmarkScenario,
        turn: BenchmarkScenarioTurn,
        *,
        turn_index: int,
        variant: BenchmarkScenarioVariant,
        workspace: Path,
        scenario_output: Path,
        model_client: ModelClient,
        repository_memory: RepositoryMemoryStore,
        session_memory: ReplSessionMemory,
        memory_aliases: dict[str, str],
    ) -> BenchmarkScenarioTurnResult:
        turn_output = scenario_output / "turns" / f"{turn_index:02d}_{turn.id}"
        turn_output.mkdir(parents=True, exist_ok=True)
        trace_writer = TraceWriter(turn_output / "trace.jsonl")
        run_id = f"{scenario.id}_{variant.id}_{turn.id}"
        trace_writer.write_event(
            "benchmark_scenario_turn_started",
            scenario_id=scenario.id,
            variant=variant.id,
            turn_id=turn.id,
            memory_mode=variant.memory_mode,
            session_key=turn.session_key,
            conversation_session_id=session_memory.session_id,
        )
        before_files = _text_files(workspace)
        loop: AgentLoop | None = None
        agent_result: AgentRunResult | None = None
        oracle_payload: dict[str, Any] = {}
        finalized = None
        error: str | None = None
        selected_ids: list[str] = []

        try:
            orchestrator = (
                RequestOrchestrator(
                    repository_memory=repository_memory,
                    trace_writer=trace_writer,
                    review_model_client=(model_client if scenario.lifecycle is not None else None),
                )
                if variant.memory_mode != "off"
                else None
            )
            long_term_context, selected_ids, memory_source = _recall_for_turn(
                variant.memory_mode,
                repository_memory=repository_memory,
                trace_writer=trace_writer,
            )
            memory_snapshot = (
                MemorySnapshotStore(turn_output).save(
                    repository_id=repository_memory.repository_id,
                    rendered_index=memory_source.rendered_index,
                    topic_payloads=memory_source.topic_payloads,
                )
                if memory_source is not None
                else None
            )
            context_builder, context_preparer = _scenario_context_components(
                scenario,
                variant=variant,
                model_client=model_client,
                trace_writer=trace_writer,
                context_budget_override=self.config.context_budget_override,
            )
            if turn.execution_mode == "memory_only":
                final_text = turn.assistant_text or "已记录当前要求。"
                agent_result = AgentRunResult(
                    status="completed",
                    final_text=final_text,
                    steps=0,
                    tool_calls=0,
                    stop_reason="benchmark_memory_only",
                )
                trace_writer.write_event(
                    "benchmark_memory_only_turn_completed",
                    scenario_id=scenario.id,
                    turn_id=turn.id,
                )
            else:
                loop = AgentLoop(
                    task=_render_benchmark_task(turn.prompt, turn.constraints),
                    workspace=workspace,
                    model_client=model_client,
                    trace_writer=trace_writer,
                    config=AgentLoopConfig(
                        max_steps=self.config.max_steps,
                        max_tool_calls=self.config.max_tool_calls,
                        rollback_on_unfinished_stop=False,
                        repository_memory_enabled=variant.memory_mode != "off",
                    ),
                    skill_names=turn.skills,
                    data_dir=repository_memory.data_dir,
                    repository_memory=(
                        repository_memory
                        if variant.memory_mode == "index_topic"
                        else None
                    ),
                    long_term_context=long_term_context,
                    memory_snapshot_hash=(
                        memory_snapshot.index_hash
                        if memory_snapshot is not None
                        else None
                    ),
                    memory_snapshot_path=(
                        MemorySnapshotStore(turn_output).checkpoint_path
                        if memory_snapshot is not None
                        else None
                    ),
                    context_builder=context_builder,
                    context_preparer=context_preparer,
                    enable_write=True,
                    approval_client=BenchmarkApprovalClient(turn.constraints),
                    approval_store=ApprovalStore(turn_output / "approvals"),
                    checkpoint_store=CheckpointStore(turn_output / "checkpoints"),
                    run_id=run_id,
                    provider=self.config.provider,
                    model=self.config.model or getattr(model_client, "model", None),
                    session_memory=session_memory,
                )
                agent_result = loop.run()
                final_text = agent_result.final_text or ""
            session_memory.add_user_turn(turn.prompt, run_id=run_id)
            session_memory.add_assistant_turn(
                final_text
                or (
                    f"Run finished with status {agent_result.status}: "
                    f"{agent_result.stop_reason}"
                ),
                history_length=len(session_memory.load_message_history()),
                run_id=run_id,
            )
            if (
                agent_result.status == "completed"
                and final_text
                and orchestrator is not None
            ):
                finalized = orchestrator.finalize_completed_run(
                    run_id=run_id,
                    user_input=turn.prompt,
                    assistant_text=final_text,
                    observations=(
                        list(loop.observations) if loop is not None else []
                    ),
                    modified_files=(
                        list(loop.modified_files) if loop is not None else []
                    ),
                    verification=(
                        loop.run_state.verification if loop is not None else None
                    ),
                )
            oracle_payload = _evaluate_oracle(
                turn,
                workspace=workspace,
                final_text=final_text,
                trace_path=trace_writer.trace_path,
                task_output=turn_output,
                grader_source_path=(
                    scenario.suite_root / "__scenario_suite__.yaml"
                    if scenario.suite_root is not None
                    else scenario.source_path
                ),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trace_writer.write_event(
                "benchmark_scenario_turn_error",
                scenario_id=scenario.id,
                turn_id=turn.id,
                error=error,
            )

        after_files = _text_files(workspace)
        diff_text = _diff_text(before_files, after_files)
        diff_path = turn_output / "turn.diff"
        diff_path.write_text(diff_text, encoding="utf-8")
        events = _read_trace_events(trace_writer.trace_path)
        recall = _score_recall(
            selected_ids=selected_ids,
            expected_keys=turn.expected_memory_keys,
            forbidden_keys=turn.forbidden_memory_keys,
            aliases=memory_aliases,
        )
        tool_call_checks = _score_tool_calls(
            turn.expected_tool_calls,
            events,
            memory_mode=variant.memory_mode,
        )
        tool_calls_passed = all(check.passed for check in tool_call_checks)
        file_content_checks = _score_file_contents(
            turn.expected_file_contents,
            workspace=workspace,
        )
        file_contents_passed = all(check.matched for check in file_content_checks)
        canonical_messages = loop.user_turn.snapshot_messages() if loop is not None else []
        fact_checks = _score_facts(
            turn.required_facts,
            final_text=agent_result.final_text if agent_result else None,
            canonical_messages=canonical_messages,
            events=events,
            context_compaction_mode=variant.context_compaction_mode,
        )
        facts_passed = all(check.matched for check in fact_checks)
        changed_files = _changed_paths_from_diff(diff_text)
        expected_changed = _expected_files_changed(
            diff_text,
            turn.expected.required_modified_files,
        )
        forbidden_changed = _forbidden_file_changed(diff_text, turn.constraints.forbidden_files)
        unexpected_modified_files = _unexpected_modified_files(
            changed_files,
            turn.expected.allowed_modified_files,
        )
        max_modified_files_exceeded = (
            turn.expected.max_modified_files is not None
            and len(changed_files) > turn.expected.max_modified_files
        )
        expected_finalization_status = (
            "review_recorded" if scenario.lifecycle is not None else turn.expected_finalization_status
        )
        memory_finalization_passed = (
            expected_finalization_status is None
            or (
                finalized is not None
                and finalized.status == expected_finalization_status
            )
        )
        resolved = (
            error is None
            and agent_result is not None
            and agent_result.status == "completed"
            and bool(oracle_payload.get("passed"))
            and recall.passed
            and _tool_calls_satisfy_resolve_gate(scenario, tool_calls_passed)
            and file_contents_passed
            and facts_passed
            and expected_changed
            and not forbidden_changed
            and not unexpected_modified_files
            and not max_modified_files_exceeded
            and memory_finalization_passed
        )
        metrics = _scenario_turn_metrics(
            events,
            steps=agent_result.steps if agent_result else 0,
            tool_calls=agent_result.tool_calls if agent_result else 0,
            canonical_messages=canonical_messages,
            current_prompt=turn.prompt,
        )
        result = BenchmarkScenarioTurnResult(
            id=turn.id,
            title=turn.title or turn.id,
            status="resolved" if resolved else ("error" if error else "failed"),
            resolved=resolved,
            session_key=turn.session_key,
            session_id=session_memory.session_id,
            stop_reason=agent_result.stop_reason if agent_result else None,
            final_text=agent_result.final_text if agent_result else None,
            memory_finalization_status=(finalized.status if finalized is not None else None),
            memory_pending_user_turns=(finalized.pending_user_turns if finalized is not None else 0),
            memory_batch_user_turns=0,
            memory_finalization_passed=memory_finalization_passed,
            oracle=oracle_payload,
            recall=recall,
            tool_call_checks=tool_call_checks,
            tool_calls_passed=tool_calls_passed,
            file_content_checks=file_content_checks,
            file_contents_passed=file_contents_passed,
            fact_checks=fact_checks,
            fact_retention_rate=_rate(
                sum(1 for item in fact_checks if item.matched),
                len(fact_checks),
            ),
            final_text_fact_accuracy=_fact_location_rate(fact_checks, "final_text"),
            semantic_history_fact_retention=_semantic_history_fact_retention(
                fact_checks,
                canonical_messages,
            ),
            compacted_history_fact_retention=_fact_location_rate(
                fact_checks,
                "compacted_history",
            ),
            historical_dialogue_fact_retention=_fact_location_rate(
                fact_checks,
                "historical_canonical_messages",
            ),
            facts_passed=facts_passed,
            expected_files_changed=expected_changed,
            forbidden_file_changed=forbidden_changed,
            changed_files=changed_files,
            unexpected_modified_files=unexpected_modified_files,
            max_modified_files_exceeded=max_modified_files_exceeded,
            metrics=metrics,
            trace_path=str(trace_writer.trace_path),
            diff_path=str(diff_path),
            error=error,
        )
        trace_writer.write_event(
            "benchmark_scenario_turn_finished",
            scenario_id=scenario.id,
            turn_id=turn.id,
            resolved=resolved,
            recall_f1=recall.f1,
            fact_retention_rate=result.fact_retention_rate,
            memory_finalization_status=(finalized.status if finalized is not None else None),
        )
        write_json(turn_output / "result.json", result)
        return result

    def _create_model_client(
        self,
        scenario: BenchmarkScenario,
        variant: BenchmarkScenarioVariant,
    ) -> ModelClient:
        _ = (scenario, variant)
        return create_model_client(provider=self.config.provider, model=self.config.model)


def _resolve_scenario_workspace(
    scenario: BenchmarkScenario,
    suite_path: Path,
) -> Path:
    base = scenario.source_path.parent if scenario.source_path is not None else suite_path
    workspace = Path(scenario.workspace)
    if not workspace.is_absolute():
        workspace = base / workspace
    resolved = workspace.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Scenario workspace does not exist: {resolved}")
    return resolved


def _copy_scenario_workspace(
    source: Path,
    destination: Path,
    *,
    include_git: bool,
) -> None:
    ignored_names = tuple(
        name
        for name in BENCHMARK_IGNORED_NAMES
        if not (include_git and name == ".git")
    )
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*ignored_names),
    )


def _directory_manifest_hash(
    root: Path,
    *,
    ignored_patterns: tuple[str, ...] = (),
    ignored_suffixes: tuple[str, ...] = (),
) -> str:
    manifest: list[dict[str, object]] = []
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            if _path_matches_patterns(relative, ignored_patterns):
                continue
            normalized = f"/{relative}"
            if any(normalized.endswith(suffix) for suffix in ignored_suffixes):
                continue
            content = path.read_bytes()
            manifest.append(
                {
                    "path": relative,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _path_matches_patterns(path: str, patterns: tuple[str, ...]) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(path, pattern) for pattern in patterns)


def _consumer_sessions_are_isolated(
    sessions: Iterable[ReplSessionMemory],
    *,
    producer_session_ids: set[str],
    producer_prompts: list[str],
) -> bool:
    values = list(sessions)
    if not values:
        return False
    for session in values:
        if session.session_id in producer_session_ids:
            return False
        serialized = json.dumps(
            {
                "dialogue": [turn.model_dump(mode="json") for turn in session.dialogue],
                "message_history": session.message_history,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if any(prompt in serialized for prompt in producer_prompts):
            return False
    return True


def _combined_constraints(
    turns: list[BenchmarkScenarioTurn],
) -> BenchmarkConstraints:
    commands: list[str] = []
    allowed_files: list[str] = []
    forbidden_files: list[str] = []
    for turn in turns:
        for value, target in (
            (turn.constraints.allowed_commands, commands),
            (turn.constraints.allowed_files, allowed_files),
            (turn.constraints.forbidden_files, forbidden_files),
        ):
            for item in value:
                if item not in target:
                    target.append(item)
    return BenchmarkConstraints(
        allowed_commands=commands,
        allowed_files=allowed_files,
        forbidden_files=forbidden_files,
    )


def _seed_memories(
    scenario: BenchmarkScenario,
    *,
    repository_memory: RepositoryMemoryStore,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for seed in scenario.seed_memories:
        topic = seed.topic or {
            "preference": "instructions",
            "coding_style": "instructions",
            "workflow": "build-and-test",
            "architecture": "decisions",
        }[seed.kind]
        entry_type = {
            "instructions": "user_instruction",
            "build-and-test": "procedure",
            "debugging": "pitfall",
            "decisions": "decision",
            "environment": "environment",
        }[topic]
        entry = repository_memory.topic_store.add_entry(
            topic=topic,
            entry_type=entry_type,
            summary=seed.content,
            evidence_ids=[f"benchmark_seed:{scenario.id}:{seed.key}"],
            entry_id=f"benchmark_{scenario.id}_{seed.key}",
        )
        aliases[seed.key] = entry.entry_id
        if seed.status == "inactive":
            repository_memory.topic_store.deactivate_entry(
                topic=topic,
                entry_id=entry.entry_id,
            )
    repository_memory.refresh_index()
    return aliases


def _score_repository_memory(
    expectation: BenchmarkRepositoryMemoryExpectation | None,
    *,
    repository_memory: RepositoryMemoryStore,
) -> list[BenchmarkRepositoryMemoryCheck]:
    """Score durable V2 state without constraining consolidation entry counts."""

    if expectation is None:
        return []

    checks: list[BenchmarkRepositoryMemoryCheck] = []
    workflow = repository_memory.workflow_store.snapshot()
    events, event_errors = _read_jsonl_objects_strict(
        repository_memory.event_store.path
    )

    if expectation.pending_reviews is not None:
        actual = len(workflow.pending_reviews)
        checks.append(
            BenchmarkRepositoryMemoryCheck(
                id="pending_reviews",
                passed=actual == expectation.pending_reviews,
                expected=expectation.pending_reviews,
                actual=actual,
            )
        )
    if expectation.pending_candidates is not None:
        actual = sum(1 for item in workflow.candidates if item.status == "pending")
        checks.append(
            BenchmarkRepositoryMemoryCheck(
                id="pending_candidates",
                passed=actual == expectation.pending_candidates,
                expected=expectation.pending_candidates,
                actual=actual,
            )
        )
    if expectation.registered_topics is not None:
        expected_topics = sorted(set(expectation.registered_topics))
        actual_topics = sorted(repository_memory.index_store.registered_topics())
        checks.append(
            BenchmarkRepositoryMemoryCheck(
                id="registered_topics",
                passed=actual_topics == expected_topics,
                expected=expected_topics,
                actual=actual_topics,
            )
        )
        index_path = repository_memory.index_store.path
        index_content = (
            index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
        )
        latest_refresh = next(
            (
                event
                for event in reversed(events)
                if event.get("event") == "memory_index_refreshed"
            ),
            None,
        )
        actual_hash = repository_memory.index_store.content_hash(index_content)
        index_topics = sorted(
            topic
            for topic in actual_topics
            if f"- {topic}:" in index_content
        )
        index_passed = (
            index_topics == actual_topics
            and latest_refresh is not None
            and latest_refresh.get("index_hash") == actual_hash
        )
        checks.append(
            BenchmarkRepositoryMemoryCheck(
                id="memory_index",
                passed=index_passed,
                expected={"registered_topics": expected_topics},
                actual={
                    "registered_topics": index_topics,
                    "index_hash": actual_hash,
                    "event_index_hash": (
                        latest_refresh.get("index_hash")
                        if latest_refresh is not None
                        else None
                    ),
                },
            )
        )

    for topic, topic_expectation in expectation.topics.items():
        document = repository_memory.topic_store.read(topic)
        active_text = "\n".join(
            entry.summary for entry in document.entries if entry.status == "active"
        )
        normalized = active_text.casefold()
        missing = [
            keyword
            for keyword in topic_expectation.required_keywords
            if keyword.casefold() not in normalized
        ]
        forbidden = [
            keyword
            for keyword in topic_expectation.forbidden_keywords
            if keyword.casefold() in normalized
        ]
        checks.append(
            BenchmarkRepositoryMemoryCheck(
                id=f"topic:{topic}",
                passed=not missing and not forbidden,
                expected={
                    "required_keywords": topic_expectation.required_keywords,
                    "forbidden_keywords": topic_expectation.forbidden_keywords,
                },
                actual={
                    "active_entry_count": sum(
                        1 for entry in document.entries if entry.status == "active"
                    ),
                    "missing_keywords": missing,
                    "forbidden_hits": forbidden,
                },
            )
        )

    event_names = [str(event.get("event") or "") for event in events]
    missing_events = [
        event for event in expectation.required_events if event not in event_names
    ]
    checks.append(
        BenchmarkRepositoryMemoryCheck(
            id="event_log",
            passed=not missing_events and not event_errors,
            expected=expectation.required_events,
            actual={
                "event_count": len(events),
                "missing_events": missing_events,
                "parse_errors": event_errors,
            },
        )
    )

    return checks


def _read_jsonl_objects_strict(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {line_number}: expected object")
            continue
        records.append(payload)
    return records, errors


def _recall_for_turn(
    mode: MemoryMode,
    *,
    repository_memory: RepositoryMemoryStore,
    trace_writer: TraceWriter,
) -> tuple[str, list[str], RepositoryMemorySnapshotSource | None]:
    if mode == "off":
        trace_writer.write_event(
            "memory_profile_injected",
            injection_mode="off",
            selected_ids=[],
            registered_topics=[],
        )
        return "", [], None
    read_memory_available = mode == "index_topic"
    memory_source = (
        repository_memory.capture_snapshot_source()
        if read_memory_available
        else None
    )
    rendered = (
        memory_source.rendered_index
        if memory_source is not None
        else repository_memory.render_index(include_read_instructions=False)
    )
    trace_writer.write_event(
        "memory_profile_injected",
        injection_mode=mode,
        selected_ids=[],
        registered_topics=(
            list(memory_source.topic_payloads)
            if memory_source is not None
            else list(repository_memory.index_store.registered_topics())
        ),
        index_chars=len(rendered),
        read_memory_available=read_memory_available,
    )
    return rendered, [], memory_source


def _scenario_context_components(
    scenario: BenchmarkScenario,
    *,
    variant: BenchmarkScenarioVariant,
    model_client: ModelClient,
    trace_writer: TraceWriter,
    context_budget_override: int | None = None,
) -> tuple[ContextBuilder | None, ContextPreparer | None]:
    resolved_budget = (
        context_budget_override
        if context_budget_override is not None
        else scenario.context_budget
    )
    if resolved_budget is None:
        return None, None
    context_budget = int(resolved_budget)
    reserved_output = min(scenario.reserved_output, context_budget - 1)
    budget = TokenBudget(
        context_budget=context_budget,
        reserved_output=reserved_output,
        soft_limit=0.80,
        hard_limit=0.95,
    )
    semantic_compactor = None
    if variant.context_compaction_mode == "llm_hard":
        semantic_compactor = LLMSemanticHistoryCompactor(
            _semantic_compaction_client(
                model_client,
                fault=scenario.semantic_compaction_fault,
            ),
            trace_writer=trace_writer,
        )
    return (
        ContextBuilder(budget=budget),
        ContextPreparer(
            budget,
            semantic_compactor=semantic_compactor,
        ),
    )


def _score_recall(
    *,
    selected_ids: list[str],
    expected_keys: list[str],
    forbidden_keys: list[str],
    aliases: dict[str, str],
) -> BenchmarkRecallResult:
    expected_ids = {
        aliases[key]
        for key in expected_keys
        if key in aliases
    }
    missing_expected_keys = [key for key in expected_keys if key not in aliases]
    forbidden_ids = {
        aliases[key]
        for key in forbidden_keys
        if key in aliases
    }
    selected = set(selected_ids)
    true_positive_count = len(selected & expected_ids)
    precision = _precision(true_positive_count, len(selected), len(expected_keys))
    recall = _rate(true_positive_count, len(expected_keys))
    f1 = _f1(precision, recall)
    reverse_aliases = {value: key for key, value in aliases.items()}
    selected_keys = [reverse_aliases.get(value, f"id:{value}") for value in selected_ids]
    return BenchmarkRecallResult(
        selected_memory_ids=selected_ids,
        selected_memory_keys=selected_keys,
        expected_memory_keys=expected_keys,
        forbidden_memory_keys=forbidden_keys,
        true_positive_count=true_positive_count,
        selected_count=len(selected),
        expected_count=len(expected_keys),
        precision=precision,
        recall=recall,
        f1=f1,
        passed=(
            not missing_expected_keys
            and expected_ids.issubset(selected)
            and not bool(selected & forbidden_ids)
        ),
    )


def _score_tool_calls(
    expectations: list[BenchmarkToolCallExpectation],
    events: list[dict[str, Any]],
    *,
    memory_mode: MemoryMode = "off",
) -> list[BenchmarkToolCallCheck]:
    tool_events = [event for event in events if event.get("type") == "tool_called"]
    checks: list[BenchmarkToolCallCheck] = []
    for expectation in expectations:
        if expectation.memory_modes and memory_mode not in expectation.memory_modes:
            continue
        matched_count = sum(
            1
            for event in tool_events
            if event.get("tool") == expectation.tool
            and _contains_expected_value(
                event.get("args") if isinstance(event.get("args"), dict) else {},
                expectation.arguments,
            )
        )
        passed = matched_count >= expectation.min_count and (
            expectation.max_count is None or matched_count <= expectation.max_count
        )
        checks.append(
            BenchmarkToolCallCheck(
                id=expectation.id,
                tool=expectation.tool,
                arguments=expectation.arguments,
                min_count=expectation.min_count,
                max_count=expectation.max_count,
                matched_count=matched_count,
                passed=passed,
            )
        )
    return checks


def _tool_calls_satisfy_resolve_gate(
    scenario: BenchmarkScenario,
    tool_calls_passed: bool,
) -> bool:
    return tool_calls_passed or not scenario.tool_calls_gate_resolve


def _score_file_contents(
    expectations: list[BenchmarkFileContentExpectation],
    *,
    workspace: Path,
) -> list[BenchmarkFileContentCheck]:
    workspace_root = workspace.resolve()
    checks: list[BenchmarkFileContentCheck] = []
    for expectation in expectations:
        path = (workspace_root / expectation.path).resolve()
        if not path.is_relative_to(workspace_root):
            checks.append(
                BenchmarkFileContentCheck(
                    id=expectation.id,
                    path=expectation.path,
                    matched=False,
                    exists=False,
                    missing_keywords=list(expectation.required_keywords),
                )
            )
            continue
        exists = path.is_file()
        content = path.read_text(encoding="utf-8") if exists else ""
        lowered = content.lower()
        missing_keywords = [
            keyword
            for keyword in expectation.required_keywords
            if keyword.lower() not in lowered
        ]
        forbidden_hits = [
            keyword
            for keyword in expectation.forbidden_keywords
            if keyword.lower() in lowered
        ]
        matched = (
            (exists if expectation.must_exist else not exists)
            and not missing_keywords
            and not forbidden_hits
        )
        checks.append(
            BenchmarkFileContentCheck(
                id=expectation.id,
                path=expectation.path,
                matched=matched,
                exists=exists,
                missing_keywords=missing_keywords,
                forbidden_hits=forbidden_hits,
            )
        )
    return checks


def _contains_expected_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_expected_value(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


def _score_facts(
    expectations: list[BenchmarkFactExpectation],
    *,
    final_text: str | None,
    canonical_messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    context_compaction_mode: ContextCompactionMode = "llm_hard",
) -> list[BenchmarkFactCheck]:
    current_user_index = _latest_user_message_index(canonical_messages)
    historical_messages = (
        canonical_messages[:current_user_index]
        if current_user_index is not None
        else canonical_messages
    )
    current_user_messages = (
        canonical_messages[current_user_index : current_user_index + 1]
        if current_user_index is not None
        else []
    )
    canonical_without_current_user = (
        [
            *canonical_messages[:current_user_index],
            *canonical_messages[current_user_index + 1 :],
        ]
        if current_user_index is not None
        else canonical_messages
    )
    semantic_history = _semantic_history_text(canonical_messages)
    stable_sources = {
        "final_text": final_text or "",
        "semantic_history": semantic_history,
        # Includes semantic-history and deterministic compacted-execution records.
        "compacted_history": _compacted_history_text(canonical_messages),
        "historical_canonical_messages": json.dumps(
            historical_messages,
            ensure_ascii=False,
            default=str,
        ),
        "current_user": json.dumps(
            current_user_messages,
            ensure_ascii=False,
            default=str,
        ),
        "trace": json.dumps(events, ensure_ascii=False, default=str),
    }
    checks: list[BenchmarkFactCheck] = []
    for expectation in expectations:
        if (
            expectation.context_compaction_modes
            and context_compaction_mode not in expectation.context_compaction_modes
        ):
            continue
        sources = {
            **stable_sources,
            "canonical_messages": json.dumps(
                (
                    canonical_without_current_user
                    if expectation.exclude_current_user
                    else canonical_messages
                ),
                ensure_ascii=False,
                default=str,
            ),
        }
        matched_locations: dict[str, bool] = {}
        forbidden_hits_by_location: dict[str, list[str]] = {}
        for location in expectation.locations:
            haystack = sources[location].lower()
            matches = [keyword.lower() in haystack for keyword in expectation.keywords]
            required_matched = (
                all(matches) if expectation.match == "all" else any(matches)
            )
            forbidden_hits = [
                keyword
                for keyword in expectation.forbidden_keywords
                if keyword.lower() in haystack
            ]
            forbidden_hits_by_location[location] = forbidden_hits
            matched_locations[location] = required_matched and not forbidden_hits
        checks.append(
            BenchmarkFactCheck(
                id=expectation.id,
                matched=any(matched_locations.values()),
                keywords=expectation.keywords,
                forbidden_keywords=expectation.forbidden_keywords,
                locations=expectation.locations,
                matched_locations=matched_locations,
                forbidden_hits_by_location=forbidden_hits_by_location,
                exclude_current_user=expectation.exclude_current_user,
            )
        )
    return checks


def _latest_user_message_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def _fact_location_rate(
    checks: list[BenchmarkFactCheck],
    location: str,
) -> float | None:
    relevant = [
        check.matched_locations[location]
        for check in checks
        if location in check.matched_locations
    ]
    if not relevant:
        return None
    return _rate(sum(1 for matched in relevant if matched), len(relevant))


def _semantic_history_fact_retention(
    checks: list[BenchmarkFactCheck],
    messages: list[dict[str, Any]],
) -> float | None:
    """Return N/A when no semantic-history record exists in this Turn."""

    if not _semantic_history_text(messages):
        return None
    return _fact_location_rate(checks, "semantic_history")


def _semantic_history_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if str(message.get("content") or "").lstrip().startswith(
            "[MiniCode semantic history]"
        )
    )


def _compacted_history_text(messages: list[dict[str, Any]]) -> str:
    headings = (
        "[MiniCode semantic history]",
        "[MiniCode compacted execution]",
        "Compacted history:",
        "Reactive compact summary:",
        "Earlier conversation summary:",
    )
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if any(
            str(message.get("content") or "").lstrip().startswith(heading)
            for heading in headings
        )
    )


def _context_quality_metrics(
    events: list[dict[str, Any]],
    *,
    canonical_messages: list[dict[str, Any]],
    current_prompt: str,
) -> BenchmarkScenarioTurnMetrics:
    """Derive context-rationality signals from Trace and persisted messages."""

    compression_events = [
        event for event in events if event.get("type") == "context_compressed"
    ]
    soft_events = [
        event
        for event in compression_events
        if (event.get("details") or {}).get("phase") == "soft"
    ]
    cold_turn_events = [
        event
        for event in compression_events
        if event.get("reason") == "cold_turn_compaction"
    ]
    semantic_events = [
        event
        for event in compression_events
        if event.get("reason") == "semantic_history"
    ]
    semantic_started_events = [
        event
        for event in events
        if event.get("type") == "semantic_compaction_started"
    ]
    semantic_finished_events = [
        event
        for event in events
        if event.get("type") in {
            "semantic_compaction_completed",
            "semantic_compaction_failed",
        }
    ]
    hard_events = [
        event
        for event in compression_events
        if event.get("reason") == "hard_fallback"
        or (event.get("details") or {}).get("phase") == "hard"
    ]
    positive_reductions = [
        max(
            0,
            int(event.get("before_tokens") or 0)
            - int(event.get("after_tokens") or 0),
        )
        for event in compression_events
    ]
    before_tokens = max(
        [int(event.get("before_tokens") or 0) for event in compression_events],
        default=0,
    )
    reduced_tokens = sum(positive_reductions)
    semantic_messages = [
        str(message.get("content") or "")
        for message in canonical_messages
        if str(message.get("content") or "").lstrip().startswith(
            "[MiniCode semantic history]"
        )
    ]
    contamination_markers = (
        '"tool_call_id"',
        '"tool_calls"',
        '"role":"tool"',
        '"role": "tool"',
        "[result removed:",
    )
    semantic_summary_contamination_count = sum(
        1
        for content in semantic_messages
        if any(marker in content.lower() for marker in contamination_markers)
    )
    latest_user = next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(canonical_messages)
            if message.get("role") == "user"
        ),
        "",
    )
    context_events = [
        event for event in events if event.get("type") == "context_built"
    ]
    return BenchmarkScenarioTurnMetrics(
        compression_count=len(compression_events),
        soft_compaction_count=len(soft_events),
        cold_turn_compaction_count=len(cold_turn_events),
        cold_turns_compacted=sum(
            int((event.get("details") or {}).get("compacted_turns") or 0)
            for event in cold_turn_events
        ),
        removed_protocol_messages=sum(
            int(
                (event.get("details") or {}).get("removed_protocol_messages")
                or 0
            )
            for event in cold_turn_events
        ),
        retained_execution_fact_count=(
            sum(
                int(
                    (event.get("details") or {}).get("retained_execution_facts")
                    or 0
                )
                for event in cold_turn_events
            )
            + sum(
                1
                for message in canonical_messages
                for line in str(message.get("content") or "").splitlines()
                if str(message.get("content") or "").lstrip().startswith(
                    "[MiniCode compacted execution]"
                )
                and line.startswith("- ")
            )
        ),
        semantic_summary_count=len(semantic_messages),
        semantic_summary_attempt_count=sum(
            1
            for event in semantic_events
            if bool((event.get("details") or {}).get("attempted"))
        ),
        semantic_summary_success_count=sum(
            1
            for event in semantic_events
            if bool((event.get("details") or {}).get("success"))
        ),
        semantic_summary_failure_count=sum(
            1
            for event in semantic_events
            if bool((event.get("details") or {}).get("attempted"))
            and not bool((event.get("details") or {}).get("success"))
        ),
        semantic_summary_contamination_count=(
            semantic_summary_contamination_count
        ),
        semantic_summary_input_tokens=sum(
            int(event.get("input_tokens") or 0)
            for event in semantic_started_events
        ),
        semantic_summary_output_tokens=sum(
            int(event.get("output_tokens") or 0)
            for event in semantic_finished_events
        ),
        semantic_summary_duration_ms=sum(
            int(event.get("duration_ms") or 0)
            for event in semantic_finished_events
        ),
        hard_fallback_count=len(hard_events),
        ineffective_hard_fallback_count=sum(
            1
            for event in hard_events
            if (
                (event.get("details") or {}).get("changed") is False
                or (
                    int(event.get("before_tokens") or 0) > 0
                    and int(event.get("after_tokens") or 0)
                    >= int(event.get("before_tokens") or 0)
                )
            )
        ),
        semantic_turn_deletion_count=sum(
            int((event.get("details") or {}).get("semantic_turns_removed") or 0)
            for event in hard_events
        ),
        reactive_compaction_count=sum(
            1
            for event in events
            if event.get("type") == "model_recovery"
            and event.get("action") == "reactive_compact"
        ),
        placeholder_replacements=sum(
            int((event.get("details") or {}).get("replaced_results") or 0)
            for event in compression_events
        ),
        command_result_compaction_count=sum(
            int(
                (event.get("details") or {}).get(
                    "command_results_compacted"
                )
                or 0
            )
            for event in hard_events
        ),
        tool_result_preview_compaction_count=sum(
            int(
                (event.get("details") or {}).get(
                    "tool_result_previews_compacted"
                )
                or 0
            )
            for event in hard_events
        ),
        semantic_message_compaction_count=sum(
            int(
                (event.get("details") or {}).get(
                    "semantic_messages_compacted"
                )
                or 0
            )
            for event in hard_events
        ),
        protocol_error_count=len(validate_message_protocol(canonical_messages)),
        active_user_anchor_failures=(
            0
            if current_prompt.strip()
            and current_prompt.strip() in latest_user
            else 1
        ),
        compacted_execution_leak_count=sum(
            1
            for message in canonical_messages
            if str(message.get("content") or "").lstrip().startswith(
                "[MiniCode compacted execution]"
            )
        ),
        prompt_budget_exceeded_count=sum(
            1
            for event in events
            if event.get("type") == "context_budget_exceeded"
        ),
        context_tokens_before_compaction=before_tokens,
        context_tokens_reduced=reduced_tokens,
        context_reduction_ratio=(
            round(reduced_tokens / before_tokens, 6)
            if before_tokens
            else 0.0
        ),
        max_budget_usage_ratio=max(
            [float(event.get("budget_usage_ratio") or 0.0) for event in context_events],
            default=0.0,
        ),
    )


def _scenario_turn_metrics(
    events: list[dict[str, Any]],
    *,
    steps: int,
    tool_calls: int,
    canonical_messages: list[dict[str, Any]],
    current_prompt: str,
) -> BenchmarkScenarioTurnMetrics:
    context = collect_context_metrics(events)
    trace_metrics = _trace_metrics_from_events(events, context)
    quality = _context_quality_metrics(
        events,
        canonical_messages=canonical_messages,
        current_prompt=current_prompt,
    )
    return quality.model_copy(
        update={
            "steps": steps,
            "tool_calls": tool_calls,
            "context_tokens_avg": context.context_token_estimate_avg,
            "context_tokens_max": context.context_token_estimate_max,
            "compression_count": trace_metrics["compression_count"],
            "consecutive_compaction_count": trace_metrics[
                "consecutive_compaction_count"
            ],
            "provider_cached_input_tokens": trace_metrics[
                "provider_cached_input_tokens"
            ],
            "provider_cache_miss_input_tokens": trace_metrics[
                "provider_cache_miss_input_tokens"
            ],
            "read_tool_calls": trace_metrics["read_tool_calls"],
            "unique_read_resources": context.unique_read_resources,
            "repeated_read_calls": context.repeated_read_calls,
            "memory_topic_read_count": trace_metrics["memory_topic_read_count"],
            "unique_memory_topics": trace_metrics["unique_memory_topics"],
            "repeated_memory_topic_reads": trace_metrics[
                "repeated_memory_topic_reads"
            ],
            "history_compaction_tokens_removed": (
                context.history_compaction_tokens_removed
            ),
            "history_compaction_tokens_retained": (
                context.history_compaction_tokens_retained
            ),
        }
    )


def _successful_memory_topics(events: list[dict[str, Any]]) -> list[str]:
    memory_calls = {
        str(event.get("tool_call_id")): event
        for event in events
        if event.get("type") == "tool_called"
        and _event_is_memory_read(event)
        and event.get("tool_call_id") is not None
    }
    successful_call_ids = {
        str(event.get("tool_call_id"))
        for event in events
        if event.get("type") == "tool_result"
        and event.get("status") == "ok"
        and str(event.get("tool_call_id")) in memory_calls
    }
    return [
        topic
        for tool_call_id, event in memory_calls.items()
        if tool_call_id in successful_call_ids
        if (
            topic := str(
                (event.get("args") or {}).get("target")
                or (event.get("args") or {}).get("topic")
                or ""
            )
        )
    ]


def _event_is_memory_read(event: dict[str, Any]) -> bool:
    tool_name = str(event.get("tool") or "")
    args = event.get("args")
    return tool_name == "read" and isinstance(args, dict) and args.get("source") == "memory"


def _trace_metrics_from_events(
    events: list[dict[str, Any]],
    context: Any,
) -> dict[str, int | float]:
    read_tools = {
        "read",
        "search",
        "task",
        "runtime_task_status",
        "delegate_task",
    }
    context_builds = [
        event for event in events if event.get("type") == "context_built"
    ]
    compressed_builds = [
        int(event.get("compression_count") or 0) > 0
        for event in context_builds
    ]
    model_responses = [
        event for event in events if event.get("type") == "model_response"
    ]
    memory_topics = _successful_memory_topics(events)
    unique_memory_topics = len(set(memory_topics))
    return {
        "compression_count": sum(
            1 for event in events if event.get("type") == "context_compressed"
        ),
        "consecutive_compaction_count": sum(
            1
            for previous, current in zip(
                compressed_builds,
                compressed_builds[1:],
            )
            if previous and current
        ),
        "provider_cached_input_tokens": sum(
            int((event.get("usage") or {}).get("cached_input_tokens") or 0)
            for event in model_responses
        ),
        "provider_cache_miss_input_tokens": sum(
            int(
                (event.get("usage") or {}).get("cache_miss_input_tokens")
                or 0
            )
            for event in model_responses
        ),
        "read_tool_calls": sum(
            1
            for event in events
            if event.get("type") == "tool_called" and event.get("tool") in read_tools
        ),
        "memory_topic_read_count": len(memory_topics),
        "unique_memory_topics": unique_memory_topics,
        "repeated_memory_topic_reads": len(memory_topics) - unique_memory_topics,
        "unique_read_resources": context.unique_read_resources,
        "repeated_read_calls": context.repeated_read_calls,
    }


def _score_context_quality(
    expectation: BenchmarkContextQualityExpectation,
    turn_results: list[Any],
    *,
    context_compaction_mode: ContextCompactionMode = "llm_hard",
    semantic_history_fact_retention: float | None,
    historical_dialogue_fact_retention: float | None,
) -> BenchmarkContextQualityResult:
    """Evaluate explicit context-quality gates without an opaque weighted score."""

    metrics = [turn.metrics for turn in turn_results]
    compression_count = sum(item.compression_count for item in metrics)
    protocol_error_count = sum(item.protocol_error_count for item in metrics)
    active_user_anchor_failures = sum(
        item.active_user_anchor_failures for item in metrics
    )
    semantic_summary_count = sum(item.semantic_summary_count for item in metrics)
    semantic_summary_attempt_count = sum(
        item.semantic_summary_attempt_count for item in metrics
    )
    semantic_summary_success_count = sum(
        item.semantic_summary_success_count for item in metrics
    )
    semantic_summary_failure_count = sum(
        item.semantic_summary_failure_count for item in metrics
    )
    semantic_summary_contamination_count = sum(
        item.semantic_summary_contamination_count for item in metrics
    )
    semantic_summary_input_tokens = sum(
        item.semantic_summary_input_tokens for item in metrics
    )
    semantic_summary_output_tokens = sum(
        item.semantic_summary_output_tokens for item in metrics
    )
    semantic_summary_duration_ms = sum(
        item.semantic_summary_duration_ms for item in metrics
    )
    metrics_by_turn_id = {
        str(getattr(turn, "id", f"turn_{index}")): turn.metrics
        for index, turn in enumerate(turn_results, start=1)
    }
    cold_turn_compaction_count = sum(
        item.cold_turn_compaction_count for item in metrics
    )
    cold_turns_compacted = sum(item.cold_turns_compacted for item in metrics)
    removed_protocol_messages = sum(
        item.removed_protocol_messages for item in metrics
    )
    retained_execution_fact_count = sum(
        item.retained_execution_fact_count for item in metrics
    )
    compacted_execution_leak_count = sum(
        item.compacted_execution_leak_count for item in metrics
    )
    prompt_budget_exceeded_count = sum(
        item.prompt_budget_exceeded_count for item in metrics
    )
    hard_fallback_count = sum(item.hard_fallback_count for item in metrics)
    hard_fallback_turn_count = sum(
        1 for item in metrics if item.hard_fallback_count > 0
    )
    max_hard_fallbacks_per_turn = max(
        [item.hard_fallback_count for item in metrics],
        default=0,
    )
    ineffective_hard_fallback_count = sum(
        item.ineffective_hard_fallback_count for item in metrics
    )
    semantic_turn_deletion_count = sum(
        item.semantic_turn_deletion_count for item in metrics
    )
    reactive_compaction_count = sum(
        item.reactive_compaction_count for item in metrics
    )
    placeholder_replacements = sum(
        item.placeholder_replacements for item in metrics
    )
    read_calls = sum(item.read_tool_calls for item in metrics)
    repeated_reads = sum(item.repeated_read_calls for item in metrics)
    repeated_read_rate = (
        repeated_reads / read_calls if read_calls else 0.0
    )
    max_budget_usage_ratio = max(
        [item.max_budget_usage_ratio for item in metrics],
        default=0.0,
    )
    context_tokens_before = sum(
        item.context_tokens_before_compaction for item in metrics
    )
    context_tokens_reduced = sum(item.context_tokens_reduced for item in metrics)
    context_reduction_ratio = (
        round(context_tokens_reduced / context_tokens_before, 6)
        if context_tokens_before
        else _average([item.context_reduction_ratio for item in metrics])
    )
    max_semantic_summaries_in_turn = max(
        [item.semantic_summary_count for item in metrics],
        default=0,
    )

    failures: list[str] = []
    if expectation.require_compaction and compression_count == 0:
        failures.append("compaction_count=0")
    if expectation.require_protocol_valid and protocol_error_count:
        failures.append(f"protocol_errors={protocol_error_count}")
    if expectation.require_active_user_anchor and active_user_anchor_failures:
        failures.append(
            f"active_user_anchor_failures={active_user_anchor_failures}"
        )
    if (
        semantic_summary_contamination_count
        > expectation.max_semantic_summary_contamination_count
    ):
        failures.append(
            "semantic_summary_contamination="
            f"{semantic_summary_contamination_count}"
        )
    if expectation.forbid_compacted_execution_leaks and compacted_execution_leak_count:
        failures.append(
            f"compacted_execution_leaks={compacted_execution_leak_count}"
        )
    if (
        prompt_budget_exceeded_count
        > expectation.max_prompt_budget_exceeded_count
    ):
        failures.append(
            f"prompt_budget_exceeded={prompt_budget_exceeded_count}"
        )
    if (
        max_semantic_summaries_in_turn
        > expectation.max_semantic_summaries_per_turn
    ):
        failures.append(
            "semantic_summaries_per_turn="
            f"{max_semantic_summaries_in_turn}"
        )
    if (
        expectation.max_semantic_summary_attempts is not None
        and semantic_summary_attempt_count
        > expectation.max_semantic_summary_attempts
    ):
        failures.append(
            f"semantic_summary_attempts={semantic_summary_attempt_count}"
        )
    if (
        expectation.max_semantic_summary_failures is not None
        and semantic_summary_failure_count
        > expectation.max_semantic_summary_failures
    ):
        failures.append(
            f"semantic_summary_failures={semantic_summary_failure_count}"
        )
    if (
        expectation.max_hard_fallback_count is not None
        and hard_fallback_count > expectation.max_hard_fallback_count
    ):
        failures.append(f"hard_fallback_count={hard_fallback_count}")
    if (
        expectation.max_hard_fallback_turn_count is not None
        and hard_fallback_turn_count
        > expectation.max_hard_fallback_turn_count
    ):
        failures.append(
            f"hard_fallback_turn_count={hard_fallback_turn_count}"
        )
    if (
        expectation.max_hard_fallbacks_per_turn is not None
        and max_hard_fallbacks_per_turn
        > expectation.max_hard_fallbacks_per_turn
    ):
        failures.append(
            "max_hard_fallbacks_per_turn="
            f"{max_hard_fallbacks_per_turn}"
        )
    if (
        expectation.max_ineffective_hard_fallback_count is not None
        and ineffective_hard_fallback_count
        > expectation.max_ineffective_hard_fallback_count
    ):
        failures.append(
            "ineffective_hard_fallback_count="
            f"{ineffective_hard_fallback_count}"
        )
    if (
        expectation.max_semantic_turn_deletion_count is not None
        and semantic_turn_deletion_count
        > expectation.max_semantic_turn_deletion_count
    ):
        failures.append(
            f"semantic_turn_deletion_count={semantic_turn_deletion_count}"
        )
    if (
        expectation.max_reactive_compaction_count is not None
        and reactive_compaction_count > expectation.max_reactive_compaction_count
    ):
        failures.append(
            f"reactive_compaction_count={reactive_compaction_count}"
        )
    if placeholder_replacements < expectation.min_placeholder_replacements:
        failures.append(
            f"placeholder_replacements={placeholder_replacements}"
        )
    if (
        expectation.max_placeholder_replacements is not None
        and placeholder_replacements > expectation.max_placeholder_replacements
    ):
        failures.append(
            f"placeholder_replacements={placeholder_replacements}"
        )
    if cold_turn_compaction_count < expectation.min_cold_turn_compaction_count:
        failures.append(
            f"cold_turn_compaction_count={cold_turn_compaction_count}"
        )
    if cold_turns_compacted < expectation.min_cold_turns_compacted:
        failures.append(f"cold_turns_compacted={cold_turns_compacted}")
    if removed_protocol_messages < expectation.min_removed_protocol_messages:
        failures.append(
            f"removed_protocol_messages={removed_protocol_messages}"
        )
    if context_compaction_mode == "deterministic":
        if semantic_summary_attempt_count:
            failures.append(
                "deterministic_semantic_summary_attempts="
                f"{semantic_summary_attempt_count}"
            )
    else:
        if (
            semantic_summary_attempt_count
            < expectation.min_semantic_summary_attempts
        ):
            failures.append(
                f"semantic_summary_attempts={semantic_summary_attempt_count}"
            )
        if (
            semantic_summary_success_count
            < expectation.min_semantic_summary_successes
        ):
            failures.append(
                f"semantic_summary_successes={semantic_summary_success_count}"
            )
        if (
            semantic_summary_failure_count
            < expectation.min_semantic_summary_failures
        ):
            failures.append(
                f"semantic_summary_failures={semantic_summary_failure_count}"
            )
        required_summary_turns = (
            (
                "attempt",
                expectation.required_semantic_summary_attempt_turn_ids,
                "semantic_summary_attempt_count",
            ),
            (
                "success",
                expectation.required_semantic_summary_success_turn_ids,
                "semantic_summary_success_count",
            ),
            (
                "failure",
                expectation.required_semantic_summary_failure_turn_ids,
                "semantic_summary_failure_count",
            ),
        )
        for label, turn_ids, field_name in required_summary_turns:
            missing_turns = [
                turn_id
                for turn_id in turn_ids
                if (
                    turn_id not in metrics_by_turn_id
                    or int(
                        getattr(metrics_by_turn_id[turn_id], field_name)
                    )
                    <= 0
                )
            ]
            if missing_turns:
                failures.append(
                    f"semantic_summary_{label}_turns_missing="
                    + ",".join(missing_turns)
                )
    if hard_fallback_count < expectation.min_hard_fallback_count:
        failures.append(f"hard_fallback_count={hard_fallback_count}")
    if (
        expectation.max_repeated_read_rate is not None
        and repeated_read_rate > expectation.max_repeated_read_rate
    ):
        failures.append(f"repeated_read_rate={repeated_read_rate:.6f}")
    if (
        expectation.max_budget_usage_ratio is not None
        and max_budget_usage_ratio > expectation.max_budget_usage_ratio
    ):
        failures.append(
            f"max_budget_usage_ratio={max_budget_usage_ratio:.6f}"
        )
    if context_reduction_ratio < expectation.min_context_reduction_ratio:
        failures.append(
            f"context_reduction_ratio={context_reduction_ratio:.6f}"
        )
    if (
        context_compaction_mode == "llm_hard"
        and expectation.min_semantic_history_fact_retention is not None
    ):
        if (
            semantic_history_fact_retention is None
            or semantic_history_fact_retention
            < expectation.min_semantic_history_fact_retention
        ):
            if semantic_history_fact_retention is None:
                failures.append("semantic_history_fact_retention=N/A")
            else:
                failures.append(
                    "semantic_history_fact_retention="
                    f"{semantic_history_fact_retention:.6f}"
                )
    if expectation.min_historical_dialogue_fact_retention is not None:
        if (
            historical_dialogue_fact_retention is None
            or historical_dialogue_fact_retention
            < expectation.min_historical_dialogue_fact_retention
        ):
            value = historical_dialogue_fact_retention or 0.0
            failures.append(
                f"historical_dialogue_fact_retention={value:.6f}"
            )

    return BenchmarkContextQualityResult(
        passed=not failures,
        failures=failures,
        protocol_error_count=protocol_error_count,
        active_user_anchor_failures=active_user_anchor_failures,
        semantic_summary_count=semantic_summary_count,
        semantic_summary_attempt_count=semantic_summary_attempt_count,
        semantic_summary_success_count=semantic_summary_success_count,
        semantic_summary_failure_count=semantic_summary_failure_count,
        semantic_summary_contamination_count=(
            semantic_summary_contamination_count
        ),
        semantic_summary_input_tokens=semantic_summary_input_tokens,
        semantic_summary_output_tokens=semantic_summary_output_tokens,
        semantic_summary_duration_ms=semantic_summary_duration_ms,
        cold_turn_compaction_count=cold_turn_compaction_count,
        cold_turns_compacted=cold_turns_compacted,
        removed_protocol_messages=removed_protocol_messages,
        retained_execution_fact_count=retained_execution_fact_count,
        compacted_execution_leak_count=compacted_execution_leak_count,
        prompt_budget_exceeded_count=prompt_budget_exceeded_count,
        hard_fallback_count=hard_fallback_count,
        hard_fallback_turn_count=hard_fallback_turn_count,
        max_hard_fallbacks_per_turn=max_hard_fallbacks_per_turn,
        ineffective_hard_fallback_count=ineffective_hard_fallback_count,
        semantic_turn_deletion_count=semantic_turn_deletion_count,
        reactive_compaction_count=reactive_compaction_count,
        placeholder_replacements=placeholder_replacements,
        repeated_read_rate=repeated_read_rate,
        max_budget_usage_ratio=max_budget_usage_ratio,
        context_tokens_reduced=context_tokens_reduced,
        context_reduction_ratio=context_reduction_ratio,
        semantic_history_fact_retention=semantic_history_fact_retention,
        historical_dialogue_fact_retention=(
            historical_dialogue_fact_retention
        ),
    )


def _build_scenario_result(
    *,
    scenario: BenchmarkScenario,
    variant: BenchmarkScenarioVariant,
    turn_results: list[BenchmarkScenarioTurnResult],
    elapsed_seconds: float,
    memory_aliases: dict[str, str],
    workspace: Path,
    output_dir: Path,
    final_diff_path: Path,
    error: str | None,
    enforce_required_trace_events: bool = True,
) -> BenchmarkScenarioResult:
    resolved_turns = sum(1 for turn in turn_results if turn.resolved)
    required_turn_ids = list(scenario.required_turn_ids) or [
        turn.id for turn in scenario.turns
    ]
    required_turn_id_set = set(required_turn_ids)
    required_turn_results = [
        turn for turn in turn_results if turn.id in required_turn_id_set
    ]
    resolved_required_turns = sum(
        1 for turn in required_turn_results if turn.resolved
    )
    failed_required_turn_ids = [
        turn.id for turn in required_turn_results if not turn.resolved
    ]
    diagnostic_failed_turn_ids = [
        turn.id
        for turn in turn_results
        if turn.id not in required_turn_id_set and not turn.resolved
    ]
    recall_tp = sum(turn.recall.true_positive_count for turn in turn_results)
    recall_selected = sum(turn.recall.selected_count for turn in turn_results)
    recall_expected = sum(turn.recall.expected_count for turn in turn_results)
    recall_precision = _precision(recall_tp, recall_selected, recall_expected)
    recall_recall = _rate(recall_tp, recall_expected)
    all_tool_call_checks = [
        check
        for turn in turn_results
        for check in turn.tool_call_checks
    ]
    all_file_content_checks = [
        check
        for turn in turn_results
        for check in turn.file_content_checks
    ]
    all_fact_checks = [
        check
        for turn in turn_results
        for check in turn.fact_checks
    ]
    fact_total = len(all_fact_checks)
    fact_matched = sum(1 for check in all_fact_checks if check.matched)
    semantic_history_fact_retention = _optional_average(
        [turn.semantic_history_fact_retention for turn in turn_results]
    )
    historical_dialogue_fact_retention = _fact_location_rate(
        all_fact_checks,
        "historical_canonical_messages",
    )
    context_expectation = scenario.context_quality
    if not enforce_required_trace_events:
        # A low-pressure baseline still enforces protocol, anchoring, and
        # summary isolation, but does not manufacture compression activity.
        context_expectation = context_expectation.model_copy(
            update={
                "require_compaction": False,
                "min_placeholder_replacements": 0,
                "min_cold_turn_compaction_count": 0,
                "min_cold_turns_compacted": 0,
                "min_removed_protocol_messages": 0,
                "min_semantic_summary_attempts": 0,
                "min_semantic_summary_successes": 0,
                "min_semantic_summary_failures": 0,
                "min_hard_fallback_count": 0,
                "required_semantic_summary_attempt_turn_ids": [],
                "required_semantic_summary_success_turn_ids": [],
                "required_semantic_summary_failure_turn_ids": [],
                "min_context_reduction_ratio": 0.0,
                "min_semantic_history_fact_retention": None,
                "min_historical_dialogue_fact_retention": None,
            }
        )
    context_quality = _score_context_quality(
        context_expectation,
        turn_results,
        context_compaction_mode=variant.context_compaction_mode,
        semantic_history_fact_retention=semantic_history_fact_retention,
        historical_dialogue_fact_retention=(
            historical_dialogue_fact_retention
        ),
    )
    read_calls = sum(turn.metrics.read_tool_calls for turn in turn_results)
    repeated_reads = sum(turn.metrics.repeated_read_calls for turn in turn_results)
    compaction_count = sum(
        turn.metrics.compression_count for turn in turn_results
    )
    consecutive_compaction_count = sum(
        turn.metrics.consecutive_compaction_count for turn in turn_results
    ) + sum(
        1
        for previous, current in zip(turn_results, turn_results[1:])
        if previous.metrics.compression_count > 0
        and current.metrics.compression_count > 0
    )
    provider_cached_input_tokens = sum(
        turn.metrics.provider_cached_input_tokens for turn in turn_results
    )
    provider_cache_miss_input_tokens = sum(
        turn.metrics.provider_cache_miss_input_tokens for turn in turn_results
    )
    turn_event_groups = [
        _read_trace_events(Path(turn.trace_path)) for turn in turn_results
    ]
    memory_topics = [
        topic
        for events in turn_event_groups
        for topic in _successful_memory_topics(events)
    ]
    unique_memory_topics = len(set(memory_topics))
    observed_trace_events = {
        str(event.get("type"))
        for events in turn_event_groups
        for event in events
        if event.get("type") is not None
    }
    required_trace_events = (
        list(scenario.required_trace_events)
        if enforce_required_trace_events
        else []
    )
    missing_trace_events = [
        event_type
        for event_type in required_trace_events
        if event_type not in observed_trace_events
    ]
    resolved = (
        error is None
        and len(turn_results) == len(scenario.turns)
        and len(required_turn_results) == len(required_turn_ids)
        and all(turn.resolved for turn in required_turn_results)
        and not missing_trace_events
        and (
            not scenario.context_quality.gate_resolution
            or context_quality.passed
        )
    )
    return BenchmarkScenarioResult(
        id=scenario.id,
        title=scenario.title,
        category=scenario.category,
        evaluation_target=scenario.evaluation_target,
        variant=variant,
        status="resolved" if resolved else ("error" if error else "failed"),
        resolved=resolved,
        turn_count=len(turn_results),
        resolved_turns=resolved_turns,
        turn_resolve_rate=_rate(resolved_turns, len(turn_results)),
        required_turn_count=len(required_turn_ids),
        resolved_required_turns=resolved_required_turns,
        required_turn_resolve_rate=_rate(
            resolved_required_turns,
            len(required_turn_ids),
        ),
        required_turn_ids=required_turn_ids,
        failed_required_turn_ids=failed_required_turn_ids,
        diagnostic_failed_turn_ids=diagnostic_failed_turn_ids,
        recall_precision=recall_precision,
        recall_recall=recall_recall,
        recall_f1=_f1(recall_precision, recall_recall),
        tool_compliance_rate=_rate(
            sum(1 for check in all_tool_call_checks if check.passed),
            len(all_tool_call_checks),
        ),
        file_content_accuracy=_rate(
            sum(1 for check in all_file_content_checks if check.matched),
            len(all_file_content_checks),
        ),
        fact_retention_rate=_rate(fact_matched, fact_total),
        final_text_fact_accuracy=_fact_location_rate(all_fact_checks, "final_text"),
        compacted_history_fact_retention=_fact_location_rate(
            all_fact_checks,
            "compacted_history",
        ),
        historical_dialogue_fact_retention=(
            historical_dialogue_fact_retention
        ),
        avg_context_tokens=_average(
            [turn.metrics.context_tokens_avg for turn in turn_results]
        ),
        history_compression_ratio=context_quality.context_reduction_ratio,
        repeated_read_rate=(
            repeated_reads / read_calls if read_calls else 0.0
        ),
        memory_topic_read_count=len(memory_topics),
        unique_memory_topics=unique_memory_topics,
        repeated_memory_topic_reads=len(memory_topics) - unique_memory_topics,
        compaction_count=compaction_count,
        consecutive_compaction_count=consecutive_compaction_count,
        stable_append_turns=_stable_append_turns(turn_results),
        provider_cached_input_tokens=provider_cached_input_tokens,
        provider_cache_miss_input_tokens=provider_cache_miss_input_tokens,
        execution_fact_retention=_fact_location_rate(
            all_fact_checks,
            "compacted_history",
        ),
        semantic_history_fact_retention=semantic_history_fact_retention,
        context_quality=context_quality,
        elapsed_seconds=elapsed_seconds,
        memory_aliases=dict(memory_aliases),
        turns=turn_results,
        workspace=str(workspace),
        output_dir=str(output_dir),
        final_diff_path=str(final_diff_path),
        required_trace_events=required_trace_events,
        missing_trace_events=missing_trace_events,
        error=error,
    )


def _stable_append_turns(
    turn_results: list[BenchmarkScenarioTurnResult],
) -> int:
    """Count post-compaction Turns that append without another compaction."""

    seen_compaction = False
    stable_turns = 0
    for turn in turn_results:
        if turn.metrics.compression_count > 0:
            seen_compaction = True
            continue
        if seen_compaction:
            stable_turns += 1
    return stable_turns


def _build_scenario_summary(
    *,
    suite: str,
    results: list[BenchmarkScenarioResult],
    elapsed_seconds: float,
) -> BenchmarkScenarioSummary:
    by_variant: dict[str, list[BenchmarkScenarioResult]] = {}
    for result in results:
        by_variant.setdefault(result.variant.id, []).append(result)
    deterministic_ids = [
        variant_id
        for variant_id, group in by_variant.items()
        if group
        and group[0].variant.context_compaction_mode == "deterministic"
    ]
    if deterministic_ids:
        baseline_id = sorted(deterministic_ids)[0]
    elif "off" in by_variant:
        baseline_id = "off"
    elif "A" in by_variant:
        baseline_id = "A"
    else:
        baseline_id = next(iter(by_variant), "")
    raw: dict[str, BenchmarkVariantSummary] = {}
    for variant_id, group in by_variant.items():
        correctness_group = [
            item for item in group if item.evaluation_target == "code_correctness"
        ]
        raw[variant_id] = BenchmarkVariantSummary(
            variant=group[0].variant,
            scenario_count=len(group),
            resolve_rate=_average([1.0 if item.resolved else 0.0 for item in group]),
            turn_resolve_rate=_optional_average(
                [item.turn_resolve_rate for item in group]
            ),
            recall_precision=_optional_average(
                [item.recall_precision for item in group]
            ),
            recall_recall=_optional_average(
                [item.recall_recall for item in group]
            ),
            recall_f1=_optional_average([item.recall_f1 for item in group]),
            tool_compliance_rate=_optional_average(
                [item.tool_compliance_rate for item in group]
            ),
            file_content_accuracy=_average(
                [item.file_content_accuracy for item in group]
            ),
            fact_retention_rate=_average([item.fact_retention_rate for item in group]),
            final_text_fact_accuracy=_optional_average(
                [item.final_text_fact_accuracy for item in group]
            ),
            compacted_history_fact_retention=_optional_average(
                [item.compacted_history_fact_retention for item in group]
            ),
            historical_dialogue_fact_retention=_optional_average(
                [item.historical_dialogue_fact_retention for item in group]
            ),
            avg_context_tokens=_optional_average(
                [item.avg_context_tokens for item in group]
            ),
            history_compression_ratio=_average(
                [item.history_compression_ratio for item in group]
            ),
            repeated_read_rate=_average([item.repeated_read_rate for item in group]),
            memory_topic_read_count=sum(
                item.memory_topic_read_count for item in group
            ),
            unique_memory_topics=sum(item.unique_memory_topics for item in group),
            repeated_memory_topic_reads=sum(
                item.repeated_memory_topic_reads for item in group
            ),
            compaction_count=sum(item.compaction_count for item in group),
            consecutive_compaction_count=sum(
                item.consecutive_compaction_count for item in group
            ),
            stable_append_turns=sum(item.stable_append_turns for item in group),
            provider_cached_input_tokens=sum(
                item.provider_cached_input_tokens for item in group
            ),
            provider_cache_miss_input_tokens=sum(
                item.provider_cache_miss_input_tokens for item in group
            ),
            execution_fact_retention=_optional_average(
                [item.execution_fact_retention for item in group]
            ),
            semantic_history_fact_retention=_optional_average(
                [item.semantic_history_fact_retention for item in group]
            ),
            context_quality_pass_rate=_average(
                [1.0 if item.context_quality.passed else 0.0 for item in group]
            ),
            semantic_summary_attempt_count=sum(
                item.context_quality.semantic_summary_attempt_count
                for item in group
            ),
            semantic_summary_success_count=sum(
                item.context_quality.semantic_summary_success_count
                for item in group
            ),
            semantic_summary_failure_count=sum(
                item.context_quality.semantic_summary_failure_count
                for item in group
            ),
            semantic_summary_contamination_count=sum(
                item.context_quality.semantic_summary_contamination_count
                for item in group
            ),
            semantic_summary_input_tokens=sum(
                item.context_quality.semantic_summary_input_tokens
                for item in group
            ),
            semantic_summary_output_tokens=sum(
                item.context_quality.semantic_summary_output_tokens
                for item in group
            ),
            semantic_summary_duration_ms=sum(
                item.context_quality.semantic_summary_duration_ms
                for item in group
            ),
            cold_turn_compaction_count=sum(
                item.context_quality.cold_turn_compaction_count
                for item in group
            ),
            cold_turns_compacted=sum(
                item.context_quality.cold_turns_compacted for item in group
            ),
            removed_protocol_messages=sum(
                item.context_quality.removed_protocol_messages
                for item in group
            ),
            retained_execution_fact_count=sum(
                item.context_quality.retained_execution_fact_count
                for item in group
            ),
            hard_fallback_count=sum(
                item.context_quality.hard_fallback_count for item in group
            ),
            hard_fallback_turn_count=sum(
                item.context_quality.hard_fallback_turn_count for item in group
            ),
            max_hard_fallbacks_per_turn=max(
                [
                    item.context_quality.max_hard_fallbacks_per_turn
                    for item in group
                ],
                default=0,
            ),
            ineffective_hard_fallback_count=sum(
                item.context_quality.ineffective_hard_fallback_count
                for item in group
            ),
            semantic_turn_deletion_count=sum(
                item.context_quality.semantic_turn_deletion_count
                for item in group
            ),
            reactive_compaction_count=sum(
                item.context_quality.reactive_compaction_count for item in group
            ),
            placeholder_replacements=sum(
                item.context_quality.placeholder_replacements for item in group
            ),
            protocol_error_count=sum(
                item.context_quality.protocol_error_count for item in group
            ),
            active_user_anchor_failures=sum(
                item.context_quality.active_user_anchor_failures for item in group
            ),
            compacted_execution_leak_count=sum(
                item.context_quality.compacted_execution_leak_count for item in group
            ),
            prompt_budget_exceeded_count=sum(
                item.context_quality.prompt_budget_exceeded_count for item in group
            ),
            max_budget_usage_ratio=max(
                [item.context_quality.max_budget_usage_ratio for item in group],
                default=0.0,
            ),
            code_correctness_scenario_count=len(correctness_group),
            code_correctness_resolve_rate=(
                _average([1.0 if item.resolved else 0.0 for item in correctness_group])
                if correctness_group
                else None
            ),
            unconditional_e2e_success_rate=(
                _average(
                    [
                        1.0 if item.unconditional_e2e_success else 0.0
                        for item in group
                        if item.unconditional_e2e_success is not None
                    ]
                )
                if any(
                    item.unconditional_e2e_success is not None for item in group
                )
                else None
            ),
            consumer_correctness_given_producer_succeeded=(
                _average(
                    [
                        1.0
                        if item.consumer_correctness_given_producer_succeeded
                        else 0.0
                        for item in group
                        if item.consumer_correctness_given_producer_succeeded
                        is not None
                    ]
                )
                if any(
                    item.consumer_correctness_given_producer_succeeded is not None
                    for item in group
                )
                else None
            ),
        )
    baseline = raw.get(baseline_id)
    baseline_by_scenario = {
        result.id: result
        for result in by_variant.get(baseline_id, [])
    }
    variants: list[BenchmarkVariantSummary] = []
    for variant_id in sorted(raw):
        item = raw[variant_id]
        paired = [
            (result, baseline_by_scenario[result.id])
            for result in by_variant.get(variant_id, [])
            if result.id in baseline_by_scenario
        ]
        if baseline is not None and paired:
            token_pairs = [
                (current, reference)
                for current, reference in paired
                if current.avg_context_tokens is not None
                and reference.avg_context_tokens is not None
            ]
            item.context_token_delta_vs_baseline = _average(
                [
                    current.avg_context_tokens - reference.avg_context_tokens
                    for current, reference in token_pairs
                    if current.avg_context_tokens is not None
                    and reference.avg_context_tokens is not None
                ]
            )
            item.resolve_rate_delta_vs_baseline = _average(
                [
                    (1.0 if current.resolved else 0.0)
                    - (1.0 if reference.resolved else 0.0)
                    for current, reference in paired
                ]
            )
            item.fact_retention_delta_vs_baseline = _average(
                [
                    current.fact_retention_rate - reference.fact_retention_rate
                    for current, reference in paired
                ]
            )
            correctness_pairs = [
                (current, reference)
                for current, reference in paired
                if current.evaluation_target == "code_correctness"
                and reference.evaluation_target == "code_correctness"
            ]
            if correctness_pairs:
                item.code_correctness_delta_vs_baseline = _average(
                    [
                        (1.0 if current.resolved else 0.0)
                        - (1.0 if reference.resolved else 0.0)
                        for current, reference in correctness_pairs
                    ]
                )
                item.paired_code_correctness_scenario_count = len(correctness_pairs)
                item.positive_transfer_count = sum(
                    1
                    for current, reference in correctness_pairs
                    if current.resolved and not reference.resolved
                )
                item.negative_transfer_count = sum(
                    1
                    for current, reference in correctness_pairs
                    if reference.resolved and not current.resolved
                )
                baseline_pass_count = sum(
                    1 for _, reference in correctness_pairs if reference.resolved
                )
                item.negative_transfer_rate = (
                    item.negative_transfer_count / baseline_pass_count
                    if baseline_pass_count
                    else 0.0
                )
        variants.append(item)
    return BenchmarkScenarioSummary(
        suite=suite,
        total_runs=len(results),
        resolved_runs=sum(1 for result in results if result.resolved),
        resolve_rate=_rate(
            sum(1 for result in results if result.resolved),
            len(results),
        ),
        elapsed_seconds=elapsed_seconds,
        variants=variants,
        scenarios=results,
    )


def _write_scenario_report(
    path: Path,
    summary: BenchmarkScenarioSummary,
) -> None:
    lines = [
        "# Memory and Context Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Suite: `{summary.suite}`",
        f"- Scenario Runs: {summary.total_runs}",
        f"- Resolved Runs: {summary.resolved_runs}",
        f"- Resolve Rate: {summary.resolve_rate:.0%}",
        f"- Elapsed Seconds: {summary.elapsed_seconds:.2f}",
        "",
        "## Ablation Comparison",
        "",
        "| Variant | Memory | Compaction | Resolve | Code Correctness | Correctness Delta | Turn Resolve | Recall F1 | Tool Gold | File Gold | Fact Retention | Final Text | Semantic History | Historical | Context Gate | Avg Tokens | Token Delta | Reduction | Repeat Reads | Topic Reads/U/R |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.variants:
        lines.append(
            f"| {item.variant.id} | {item.variant.memory_mode} | "
            f"{item.variant.context_compaction_mode} | {item.resolve_rate:.0%} | "
            f"{_format_optional_rate(item.code_correctness_resolve_rate)} | "
            f"{item.code_correctness_delta_vs_baseline:+.0%} | "
            f"{_format_optional_rate(item.turn_resolve_rate)} | "
            f"{_format_optional_decimal(item.recall_f1)} | "
            f"{_format_optional_rate(item.tool_compliance_rate)} | "
            f"{item.file_content_accuracy:.0%} | {item.fact_retention_rate:.0%} | "
            f"{_format_optional_rate(item.final_text_fact_accuracy)} | "
            f"{_format_optional_rate(item.semantic_history_fact_retention)} | "
            f"{_format_optional_rate(item.historical_dialogue_fact_retention)} | "
            f"{item.context_quality_pass_rate:.0%} | "
            f"{_format_optional_number(item.avg_context_tokens)} | "
            f"{item.context_token_delta_vs_baseline:+.0f} | "
            f"{item.history_compression_ratio:.0%} | {item.repeated_read_rate:.0%} | "
            f"{item.memory_topic_read_count}/{item.unique_memory_topics}/"
            f"{item.repeated_memory_topic_reads} |"
        )
    if any(item.code_correctness_scenario_count for item in summary.variants):
        lines.extend(
            [
                "",
                "## Paired Code Correctness Transfer",
                "",
                "| Variant | Paired Scenarios | Positive Transfer | Negative Transfer | Negative Transfer Rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in summary.variants:
            lines.append(
                f"| {item.variant.id} | "
                f"{item.paired_code_correctness_scenario_count} | "
                f"{item.positive_transfer_count} | "
                f"{item.negative_transfer_count} | "
                f"{item.negative_transfer_rate:.0%} |"
            )
    lifecycle_results = [
        result
        for result in summary.scenarios
        if result.unconditional_e2e_success is not None
    ]
    if lifecycle_results:
        lines.extend(
            [
                "",
                "## Repository Memory Lifecycle E2E",
                "",
                "| Variant | Unconditional E2E Success | Consumer Correctness Given Producer Succeeded |",
                "|---|---:|---:|",
            ]
        )
        for item in summary.variants:
            lines.append(
                f"| {item.variant.id} | "
                f"{_format_optional_rate(item.unconditional_e2e_success_rate)} | "
                f"{_format_optional_rate(item.consumer_correctness_given_producer_succeeded)} |"
            )
        lines.extend(
            [
                "",
                "| Scenario | Variant | Producer | Capture | Notes A/E | Consolidation Protocol | Store Contract | Snapshot | Session Isolation | Recall | Hidden Correctness | Producer Tools | Producer Avg Tokens |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in lifecycle_results:
            lines.append(
                f"| `{result.id}` | {result.variant.id} | "
                f"{_format_optional_bool(result.producer_succeeded)} | "
                f"{_format_optional_bool(result.capture_passed)} | "
                f"{result.producer_memory_entry_count}/{result.expected_producer_memory_entry_count} | "
                f"{_format_optional_bool(result.consolidation_protocol_passed)} | "
                f"{_format_optional_bool(result.repository_memory_passed)} | "
                f"{_format_optional_bool(result.snapshot_integrity_passed)} | "
                f"{_format_optional_bool(result.cross_session_isolation_passed)} | "
                f"{_format_optional_bool(result.recall_passed)} | "
                f"{_format_optional_bool(result.hidden_correctness_passed)} | "
                f"{result.producer_tool_calls} | "
                f"{result.producer_avg_context_tokens:.0f} |"
            )
    lines.extend(
        [
            "",
            "## Cache and Compaction Metrics",
            "",
            "| Variant | Compactions | Consecutive | Stable Append Turns | Provider Cached Tokens | Provider Miss Tokens | Summary Input | Summary Output | Summary ms | Execution Fact Retention |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary.variants:
        lines.append(
            f"| {item.variant.id} | {item.compaction_count} | "
            f"{item.consecutive_compaction_count} | {item.stable_append_turns} | "
            f"{item.provider_cached_input_tokens} | "
            f"{item.provider_cache_miss_input_tokens} | "
            f"{item.semantic_summary_input_tokens} | "
            f"{item.semantic_summary_output_tokens} | "
            f"{item.semantic_summary_duration_ms} | "
            f"{_format_optional_rate(item.execution_fact_retention)} |"
        )
    lines.extend(
        [
            "",
            "## Context Quality Gates",
            "",
            "| Variant | Gate Pass | Cold Events | Cold Turns | Protocol Removed | Retained Facts | Summary A/S/F | Summary Contamination | Placeholders | Hard Raw | Hard Turns | Hard Max/Turn | Ineffective Hard | Semantic Turns Deleted | Reactive | Protocol Errors | Anchor Failures | Compacted Execution Leaks | Budget Exceeded | Max Budget Use |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary.variants:
        lines.append(
            f"| {item.variant.id} | {item.context_quality_pass_rate:.0%} | "
            f"{item.cold_turn_compaction_count} | "
            f"{item.cold_turns_compacted} | "
            f"{item.removed_protocol_messages} | "
            f"{item.retained_execution_fact_count} | "
            f"{item.semantic_summary_attempt_count}/"
            f"{item.semantic_summary_success_count}/"
            f"{item.semantic_summary_failure_count} | "
            f"{item.semantic_summary_contamination_count} | "
            f"{item.placeholder_replacements} | {item.hard_fallback_count} | "
            f"{item.hard_fallback_turn_count} | "
            f"{item.max_hard_fallbacks_per_turn} | "
            f"{item.ineffective_hard_fallback_count} | "
            f"{item.semantic_turn_deletion_count} | "
            f"{item.reactive_compaction_count} | {item.protocol_error_count} | "
            f"{item.active_user_anchor_failures} | "
            f"{item.compacted_execution_leak_count} | "
            f"{item.prompt_budget_exceeded_count} | "
            f"{item.max_budget_usage_ratio:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Scenario Runs",
            "",
            "| Scenario | Target | Variant | Resolved | Context Gate | Turns | Required Turns | Recall F1 | Tool Gold | File Gold | Fact Retention | Final Text | Semantic History | Historical | Avg Tokens | Repeated Reads | Topic Reads/U/R |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in summary.scenarios:
        lines.append(
            f"| `{result.id}` | {result.evaluation_target} | {result.variant.id} | "
            f"{'yes' if result.resolved else 'no'} | "
            f"{'pass' if result.context_quality.passed else 'fail'} | "
            f"{result.resolved_turns}/{result.turn_count} | "
            f"{result.resolved_required_turns}/{result.required_turn_count} | "
            f"{_format_optional_decimal(result.recall_f1)} | "
            f"{_format_optional_rate(result.tool_compliance_rate)} | "
            f"{result.file_content_accuracy:.0%} | {result.fact_retention_rate:.0%} | "
            f"{_format_optional_rate(result.final_text_fact_accuracy)} | "
            f"{_format_optional_rate(result.semantic_history_fact_retention)} | "
            f"{_format_optional_rate(result.historical_dialogue_fact_retention)} | "
            f"{_format_optional_number(result.avg_context_tokens)} | "
            f"{result.repeated_read_rate:.0%} | "
            f"{result.memory_topic_read_count}/{result.unique_memory_topics}/"
            f"{result.repeated_memory_topic_reads} |"
        )
    failed = [result for result in summary.scenarios if not result.resolved]
    lines.extend(["", "## Failed Runs", ""])
    if not failed:
        lines.append("No failed scenario runs.")
    else:
        for result in failed:
            reasons = [
                *result.failed_required_turn_ids,
                *(
                    ["missing trace events: " + ", ".join(result.missing_trace_events)]
                    if result.missing_trace_events
                    else []
                ),
                *(
                    ["context quality: " + ", ".join(result.context_quality.failures)]
                    if not result.context_quality.passed
                    else []
                ),
            ]
            lines.append(
                f"- `{result.id}` / `{result.variant.id}`: "
                f"{result.error or '; '.join(reasons) or result.status}"
            )
    diagnostic_failures = [
        result for result in summary.scenarios if result.diagnostic_failed_turn_ids
    ]
    lines.extend(["", "## Diagnostic Turn Failures", ""])
    if not diagnostic_failures:
        lines.append("No diagnostic-only Turn failures.")
    else:
        for result in diagnostic_failures:
            lines.append(
                f"- `{result.id}` / `{result.variant.id}`: "
                + ", ".join(result.diagnostic_failed_turn_ids)
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "For `code_correctness` scenarios, the causal metric is hidden-test resolve rate and its paired memory-on minus memory-off delta. Tool-command compliance or final wording alone does not count as correctness.",
            "For `context_rationality` scenarios, resolve requires the declared context gates: protocol integrity, current-user anchoring, semantic-summary isolation, budget safety, bounded fallback/recovery, and acceptable evidence-reconstruction cost. Compression count or token reduction alone is not sufficient.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _diff_text(before_files: dict[str, str], after_files: dict[str, str]) -> str:
    parts: list[str] = []
    for relative_path in sorted(set(before_files) | set(after_files)):
        before = before_files.get(relative_path, "").splitlines(keepends=True)
        after = after_files.get(relative_path, "").splitlines(keepends=True)
        if before == after:
            continue
        parts.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
    return "".join(parts)


def _precision(true_positive: int, selected: int, expected: int) -> float:
    if selected:
        return true_positive / selected
    return 1.0 if expected == 0 else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _average(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return _average(present) if present else None


def _format_optional_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0%}"


def _format_optional_decimal(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _format_optional_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0f}"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "pass" if value else "fail"
