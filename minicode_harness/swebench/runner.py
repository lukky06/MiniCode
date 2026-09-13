"""Single-instance and batch SWE-bench execution through the normal AgentLoop."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from minicode_harness.context.message_groups import is_final_assistant_message
from minicode_harness.loop import AgentLoop, AgentLoopConfig, AgentRunResult
from minicode_harness.models import ModelClient, create_model_client
from minicode_harness.output import NullOutputSink
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.state import (
    ApprovalClient,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStore,
    CheckpointStore,
    detect_workspace_conflicts,
)
from minicode_harness.tools import CommandExecutor, LocalCommandExecutor
from minicode_harness.trace import TraceWriter

from .dataset import load_instances
from .evaluator import SweBenchEvaluator
from .metrics import collect_swebench_metrics
from .models import (
    SweBenchBudget,
    SweBenchEvaluationResult,
    SweBenchInstance,
    SweBenchInstanceResult,
    SweBenchPrediction,
    SweBenchSuiteSummary,
    WorkspaceManifest,
)
from .patch import PatchExporter
from .prediction import (
    atomic_write_json,
    load_prediction,
    write_prediction,
    write_predictions_jsonl,
)
from .prompt import build_task_prompt
from .reporting import build_suite_summary, write_suite_reports
from .repository import RepositoryCache
from .workspace import SweBenchWorkspaceManager


ModelClientFactory = Callable[[SweBenchInstance], ModelClient]
CommandExecutorFactory = Callable[[SweBenchInstance, WorkspaceManifest], CommandExecutor]


@dataclass(frozen=True)
class SweBenchRunnerConfig:
    """Runtime settings for external SWE-bench execution."""

    provider: str = "qwen"
    model: str | None = None
    budget: SweBenchBudget = field(default_factory=SweBenchBudget)
    prompt_cache_enabled: bool = True
    agent_verification_enabled: bool = True
    enable_subagents: bool = False
    enable_skills: bool = True
    attempts: int = 1
    resume: bool = True
    skip_completed: bool = True
    retry_failed: bool = False
    repository_source_urls: dict[str, str] = field(default_factory=dict)


class SweBenchApprovalClient(ApprovalClient):
    """Auto-approve bounded benchmark actions without enabling external effects."""

    _LOCAL_COMMAND_CATEGORIES: set[str] = set()
    _SANDBOX_COMMAND_CATEGORIES = {
        "diagnostic",
        "repository_script",
        "filesystem",
        "unknown",
    }

    def __init__(
        self,
        *,
        allow_commands: bool = True,
        sandboxed_commands: bool = False,
    ) -> None:
        self.allow_commands = allow_commands
        self.sandboxed_commands = sandboxed_commands
        self.requests: list[ApprovalRequest] = []

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        if request.tool_name in {"edit", "write", "apply_patch"}:
            return ApprovalResponse(
                decision=ApprovalDecision.APPROVE,
                reason="swebench_workspace_write",
            )
        if request.tool_name == "run_command" and self.allow_commands:
            category = str(request.preview.get("policy_category") or "unknown")
            allowed_categories = (
                self._SANDBOX_COMMAND_CATEGORIES
                if self.sandboxed_commands
                else self._LOCAL_COMMAND_CATEGORIES
            )
            if category in allowed_categories:
                return ApprovalResponse(
                    decision=ApprovalDecision.APPROVE,
                    reason=f"swebench_{category}_command",
                )
            return ApprovalResponse(
                decision=ApprovalDecision.REJECT,
                reason=f"SWE-bench does not auto-approve {category} commands.",
            )
        return ApprovalResponse(
            decision=ApprovalDecision.REJECT,
            reason=f"Tool is not auto-approved in SWE-bench mode: {request.tool_name}",
        )


class SweBenchRunner:
    """Run MiniCode against isolated SWE-bench instances and export predictions."""

    def __init__(
        self,
        config: SweBenchRunnerConfig | None = None,
        *,
        model_client_factory: ModelClientFactory | None = None,
        command_executor_factory: CommandExecutorFactory | None = None,
        patch_exporter: PatchExporter | None = None,
    ) -> None:
        self.config = config or SweBenchRunnerConfig()
        if self.config.attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.model_client_factory = model_client_factory or self._create_model_client
        self.command_executor_factory = (
            command_executor_factory
            or (lambda instance, manifest: LocalCommandExecutor())
        )
        self.patch_exporter = patch_exporter or PatchExporter()

    def run_dataset(
        self,
        dataset: Path | str,
        output: Path | str,
        *,
        instance_ids: Iterable[str] | None = None,
        offset: int = 0,
        limit: int | None = None,
        shard_id: int | None = None,
        num_shards: int | None = None,
        dataset_split: str = "test",
        dataset_revision: str | None = None,
        dataset_cache: Path | str | None = None,
        repository_cache: Path | str | None = None,
        evaluator: SweBenchEvaluator | None = None,
        evaluator_dataset_name: str | None = None,
        evaluation_run_id: str | None = None,
    ) -> SweBenchSuiteSummary:
        """Load, execute, optionally evaluate, and report one local dataset."""

        instances = load_instances(
            dataset,
            instance_ids=instance_ids,
            offset=offset,
            limit=limit,
            shard_id=shard_id,
            num_shards=num_shards,
            split=dataset_split,
            revision=dataset_revision,
            cache_dir=dataset_cache,
        )
        return self.run_instances(
            instances,
            output,
            dataset_label=str(dataset),
            repository_cache=repository_cache,
            evaluator=evaluator,
            evaluator_dataset_name=evaluator_dataset_name,
            evaluation_run_id=evaluation_run_id,
        )

    def run_instances(
        self,
        instances: list[SweBenchInstance],
        output: Path | str,
        *,
        dataset_label: str,
        repository_cache: Path | str | None = None,
        evaluator: SweBenchEvaluator | None = None,
        evaluator_dataset_name: str | None = None,
        evaluation_run_id: str | None = None,
    ) -> SweBenchSuiteSummary:
        """Run a stable sequence of instances; one failure never aborts the suite."""

        output_root = Path(output).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        cache_root = (
            Path(repository_cache).expanduser().resolve()
            if repository_cache is not None
            else output_root / "cache"
        )
        workspace_manager = SweBenchWorkspaceManager(
            output_root=output_root,
            repository_cache=RepositoryCache(cache_root),
        )
        ordered = sorted(instances, key=lambda item: item.instance_id)
        atomic_write_json(
            output_root / "manifest.json",
            {
                "dataset": dataset_label,
                "provider": self.config.provider,
                "model": self.config.model,
                "instance_ids": [item.instance_id for item in ordered],
                "attempts": self.config.attempts,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        results: list[SweBenchInstanceResult] = []
        for instance in ordered:
            for attempt in range(1, self.config.attempts + 1):
                results.append(
                    self._run_instance(
                        instance,
                        attempt=attempt,
                        workspace_manager=workspace_manager,
                    )
                )

        self.rebuild_predictions(output_root, results=results)
        if evaluator is not None:
            if not evaluator_dataset_name:
                raise ValueError(
                    "evaluator_dataset_name is required when an evaluator is configured"
                )
            base_run_id = evaluation_run_id or _suite_run_id()
            evaluations: list[SweBenchEvaluationResult] = []
            evaluated_results: list[SweBenchInstanceResult] = []
            for attempt in range(1, self.config.attempts + 1):
                attempt_results = [
                    result for result in results if result.attempt == attempt
                ]
                attempt_predictions = self.rebuild_predictions(
                    output_root,
                    results=attempt_results,
                    filename=f"predictions.attempt_{attempt}.jsonl",
                )
                attempt_evaluator = evaluator.with_output_dir(
                    evaluator.output_dir / f"attempt_{attempt}"
                )
                evaluation = attempt_evaluator.evaluate(
                    attempt_predictions,
                    evaluator_dataset_name,
                    sorted({result.instance_id for result in attempt_results}),
                    f"{base_run_id}_attempt_{attempt}",
                )
                evaluations.append(evaluation)
                evaluated_results.extend(
                    self._merge_evaluation(attempt_results, evaluation)
                )
            results = sorted(
                evaluated_results,
                key=lambda item: (item.instance_id, item.attempt),
            )
            atomic_write_json(
                output_root / "evaluation.json",
                {"attempts": [item.model_dump(mode="json") for item in evaluations]},
            )

        summary = build_suite_summary(
            dataset=dataset_label,
            model_name_or_path=self.model_name_or_path,
            results=results,
        )
        write_suite_reports(output_root, summary)
        return summary

    def rebuild_predictions(
        self,
        output_root: Path | str,
        *,
        results: list[SweBenchInstanceResult] | None = None,
        filename: str = "predictions.jsonl",
    ) -> Path:
        """Rebuild suite JSONL atomically from standalone prediction files."""

        root = Path(output_root).expanduser().resolve()
        if results is None:
            results = []
            for result_path in sorted((root / "instances").glob("*/result.json")):
                try:
                    results.append(
                        SweBenchInstanceResult.model_validate_json(
                            result_path.read_text(encoding="utf-8")
                        )
                    )
                except (OSError, ValueError):
                    continue
        latest: dict[str, tuple[int, SweBenchPrediction]] = {}
        for result in results:
            if not result.prediction_path:
                continue
            path = Path(result.prediction_path)
            if not path.is_file():
                continue
            prediction = load_prediction(path)
            current = latest.get(result.instance_id)
            if current is None or result.attempt >= current[0]:
                latest[result.instance_id] = (result.attempt, prediction)
        return write_predictions_jsonl(
            root / filename,
            [latest[key][1] for key in sorted(latest)],
        )

    @property
    def model_name_or_path(self) -> str:
        return self.config.model or self.config.provider

    def _run_instance(
        self,
        instance: SweBenchInstance,
        *,
        attempt: int,
        workspace_manager: SweBenchWorkspaceManager,
    ) -> SweBenchInstanceResult:
        instance_dir = workspace_manager.instance_dir(
            instance.instance_id,
            attempt=attempt,
        )
        result_path = instance_dir / "result.json"
        existing = _load_result(result_path)
        if existing is not None and self._should_skip(existing):
            return existing
        if existing is not None:
            _reset_instance_artifacts(instance_dir)

        started_at = time.monotonic()
        manifest: WorkspaceManifest | None = None
        trace_path = instance_dir / "trace.jsonl"
        checkpoint_store = CheckpointStore(instance_dir / "checkpoints")
        checkpoint = (
            checkpoint_store.load_latest()
            if self.config.resume and existing is None
            else None
        )
        try:
            if checkpoint is not None and not result_path.exists():
                manifest = _load_manifest(instance_dir / "workspace-manifest.json")
                if manifest is None or not Path(manifest.workspace).is_dir():
                    checkpoint = None
                else:
                    conflicts = detect_workspace_conflicts(
                        manifest.workspace,
                        checkpoint,
                    )
                    if conflicts:
                        result = self._base_result(
                            instance,
                            attempt=attempt,
                            status="blocked",
                            elapsed_seconds=time.monotonic() - started_at,
                            manifest=manifest,
                            error=(
                                "Checkpoint workspace conflict: "
                                + ", ".join(conflict.path for conflict in conflicts)
                            ),
                        )
                        atomic_write_json(result_path, result)
                        return result
            if manifest is None:
                manifest = workspace_manager.prepare(
                    instance,
                    attempt=attempt,
                    source_url=self.config.repository_source_urls.get(instance.repo),
                    reset=True,
                )
        except Exception as exc:
            result = self._base_result(
                instance,
                attempt=attempt,
                status="workspace_setup_failed",
                elapsed_seconds=time.monotonic() - started_at,
                manifest=manifest,
                error=f"{type(exc).__name__}: {exc}",
            )
            atomic_write_json(result_path, result)
            return result

        trace_writer = TraceWriter(trace_path)
        run_id = (
            checkpoint.run_id
            if checkpoint is not None
            else f"swebench_{_safe_id(instance.instance_id)}_{attempt}"
        )
        trace_writer.write_event(
            "swebench_instance_started",
            instance_id=instance.instance_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            attempt=attempt,
            resumed=checkpoint is not None,
        )
        cancellation_token = CancellationToken()
        timed_out = threading.Event()
        timer: threading.Timer | None = None
        if self.config.budget.max_elapsed_seconds is not None:
            timer = threading.Timer(
                self.config.budget.max_elapsed_seconds,
                _cancel_for_timeout,
                args=(cancellation_token, timed_out),
            )
            timer.daemon = True
            timer.start()

        agent_result: AgentRunResult | None = None
        loop: AgentLoop | None = None
        error: str | None = None
        try:
            if checkpoint is not None and checkpoint.status == "completed":
                agent_result = AgentRunResult(
                    status="completed",
                    final_text=_last_assistant_text(checkpoint.message_history),
                    steps=checkpoint.step,
                    tool_calls=checkpoint.tool_calls,
                    stop_reason=checkpoint.reason or "final_text",
                )
            else:
                model_client = self.model_client_factory(instance)
                command_executor = self.command_executor_factory(instance, manifest)
                loop = AgentLoop(
                    task=build_task_prompt(
                        instance,
                        verification_enabled=self.config.agent_verification_enabled,
                    ),
                    workspace=manifest.workspace,
                    model_client=model_client,
                    trace_writer=trace_writer,
                    config=AgentLoopConfig(
                        max_steps=self.config.budget.max_steps,
                        max_tool_calls=self.config.budget.max_tool_calls,
                        start_step=checkpoint.step if checkpoint else 0,
                        prompt_cache_enabled=self.config.prompt_cache_enabled,
                        rollback_on_unfinished_stop=False,
                        repository_memory_enabled=False,
                        enable_subagents=self.config.enable_subagents,
                        enable_progress_guidance=True,
                        hidden_tool_names=tuple(
                            name
                            for name, hidden in (
                                ("delegate_task", not self.config.enable_subagents),
                                ("delegate_worktree", True),
                                (
                                    "runtime_task_status",
                                    not self.config.agent_verification_enabled,
                                ),
                                (
                                    "runtime_task_stop",
                                    not self.config.agent_verification_enabled,
                                ),
                            )
                            if hidden
                        ),
                        reserve_final_step=True,
                        reasoning_effort="high",
                    ),
                    no_skills=not self.config.enable_skills,
                    data_dir=instance_dir / "harness-data",
                    enable_long_term_context=False,
                    long_term_context="",
                    enable_write=True,
                    enable_command=self.config.agent_verification_enabled,
                    approval_client=SweBenchApprovalClient(
                        allow_commands=self.config.agent_verification_enabled,
                        sandboxed_commands=bool(
                            getattr(command_executor, "sandboxed", False)
                        ),
                    ),
                    approval_store=ApprovalStore(instance_dir / "approvals"),
                    checkpoint_store=checkpoint_store,
                    run_id=run_id,
                    initial_observations=(
                        list(checkpoint.recent_observations) if checkpoint else None
                    ),
                    initial_modified_files=(
                        list(checkpoint.modified_files) if checkpoint else None
                    ),
                    initial_run_state=checkpoint.run_state if checkpoint else None,
                    initial_message_history=(
                        list(checkpoint.message_history) if checkpoint else None
                    ),
                    initial_tool_calls=checkpoint.tool_calls if checkpoint else 0,
                    provider=self.config.provider,
                    model=self.config.model or getattr(model_client, "model", None),
                    output_sink=NullOutputSink(),
                    stream_model=False,
                    cancellation_token=cancellation_token,
                    command_executor=command_executor,
                )
                agent_result = loop.run()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trace_writer.write_event(
                "swebench_agent_error",
                instance_id=instance.instance_id,
                error=error,
            )
        finally:
            if timer is not None:
                timer.cancel()

        final_patch_path = instance_dir / "final.patch"
        try:
            patch = self.patch_exporter.export(
                manifest.workspace,
                output_path=final_patch_path,
                expected_head=instance.base_commit,
            )
        except Exception as exc:
            patch = None
            error = error or f"{type(exc).__name__}: {exc}"

        metrics = collect_swebench_metrics(
            trace_path,
            run_path=instance_dir,
        )
        status = _classify_status(
            agent_result=agent_result,
            timed_out=timed_out.is_set(),
            error=error,
            patch_status=patch.status if patch is not None else "invalid",
        )
        prediction_path = instance_dir / "prediction.json"
        prediction = SweBenchPrediction(
            instance_id=instance.instance_id,
            model_name_or_path=self.model_name_or_path,
            model_patch=patch.patch if patch is not None else "",
        )
        write_prediction(prediction_path, prediction)
        modified_files = (
            list(loop.modified_files)
            if loop is not None
            else list(checkpoint.modified_files) if checkpoint is not None else []
        )
        result = SweBenchInstanceResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            status=status,
            model_name_or_path=self.model_name_or_path,
            attempt=attempt,
            run_id=run_id,
            stop_reason=agent_result.stop_reason if agent_result else None,
            final_text=agent_result.final_text if agent_result else None,
            steps=agent_result.steps if agent_result else 0,
            tool_calls=agent_result.tool_calls if agent_result else 0,
            elapsed_seconds=time.monotonic() - started_at,
            input_tokens=int(metrics.get("input_tokens") or 0),
            output_tokens=int(metrics.get("output_tokens") or 0),
            modified_files=modified_files,
            patch_bytes=patch.bytes if patch is not None else 0,
            patch_lines=(patch.patch.count("\n") if patch is not None else 0),
            patch_path=patch.patch_path if patch is not None else None,
            prediction_path=str(prediction_path),
            workspace=manifest.workspace,
            trace_path=str(trace_path),
            checkpoint_dir=str(instance_dir / "checkpoints"),
            metrics=metrics,
            error=(
                error
                or (patch.validation_error if patch is not None else "Patch export failed")
            ),
        )
        trace_writer.write_event(
            "swebench_instance_finished",
            instance_id=instance.instance_id,
            status=status,
            patch_bytes=result.patch_bytes,
            error=result.error,
        )
        atomic_write_json(result_path, result)
        return result

    def _merge_evaluation(
        self,
        results: list[SweBenchInstanceResult],
        evaluation: SweBenchEvaluationResult,
    ) -> list[SweBenchInstanceResult]:
        by_id = {item.instance_id: item for item in evaluation.instances}
        per_instance_seconds = (
            evaluation.elapsed_seconds / len(evaluation.instances)
            if evaluation.instances
            else 0.0
        )
        merged: list[SweBenchInstanceResult] = []
        for result in results:
            official = by_id.get(result.instance_id)
            if evaluation.status != "completed":
                if result.status not in {"no_patch", "patch_invalid"}:
                    result.status = "evaluation_error"
                result.error = evaluation.error or result.error
            elif official is None:
                if result.status not in {"no_patch", "patch_invalid"}:
                    result.status = "evaluation_error"
                result.error = "Official evaluation report missing for instance"
            else:
                result.official_resolved = official.resolved
                result.fail_to_pass_success = official.fail_to_pass_success
                result.pass_to_pass_success = official.pass_to_pass_success
                result.evaluation_report_path = official.report_path
                result.evaluation_seconds = per_instance_seconds
                if result.status not in {"no_patch", "patch_invalid"}:
                    if official.resolved is True:
                        result.status = "resolved"
                    elif official.resolved is False:
                        result.status = "unresolved"
                    else:
                        result.status = "evaluation_error"
                        result.error = "Official report did not contain a resolved result"
            if result.prediction_path:
                atomic_write_json(
                    Path(result.prediction_path).with_name("result.json"),
                    result,
                )
            merged.append(result)
        return merged

    def _should_skip(self, result: SweBenchInstanceResult) -> bool:
        if not self.config.skip_completed:
            return False
        failed = result.status in {
            "workspace_setup_failed",
            "provider_error",
            "agent_error",
            "agent_timeout",
            "evaluation_error",
            "blocked",
        }
        return not (failed and self.config.retry_failed)

    def _base_result(
        self,
        instance: SweBenchInstance,
        *,
        attempt: int,
        status: str,
        elapsed_seconds: float,
        manifest: WorkspaceManifest | None,
        error: str,
    ) -> SweBenchInstanceResult:
        return SweBenchInstanceResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            status=status,
            model_name_or_path=self.model_name_or_path,
            attempt=attempt,
            elapsed_seconds=elapsed_seconds,
            workspace=manifest.workspace if manifest is not None else None,
            error=error,
        )

    def _create_model_client(self, instance: SweBenchInstance) -> ModelClient:
        _ = instance
        return create_model_client(
            provider=self.config.provider,
            model=self.config.model,
        )


def _classify_status(
    *,
    agent_result: AgentRunResult | None,
    timed_out: bool,
    error: str | None,
    patch_status: str,
) -> str:
    if timed_out:
        return "agent_timeout"
    if error is not None and agent_result is None:
        lowered = error.lower()
        if "provider" in lowered or "model client" in lowered:
            return "provider_error"
        return "agent_error"
    if patch_status == "invalid":
        return "patch_invalid"
    if patch_status == "no_patch":
        return "no_patch"
    if agent_result is None:
        return "agent_error"
    if agent_result.stop_reason in {"max_steps", "step_budget_exhausted"}:
        return "step_budget_exhausted"
    if agent_result.stop_reason in {"max_tool_calls", "tool_budget_exhausted"}:
        return "tool_budget_exhausted"
    if agent_result.status != "completed":
        return "agent_error"
    return "agent_completed"


def _cancel_for_timeout(token: CancellationToken, timed_out: threading.Event) -> None:
    timed_out.set()
    token.cancel()


def _load_result(path: Path) -> SweBenchInstanceResult | None:
    if not path.is_file():
        return None
    try:
        return SweBenchInstanceResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def _load_manifest(path: Path) -> WorkspaceManifest | None:
    if not path.is_file():
        return None
    try:
        return WorkspaceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _last_assistant_text(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if not is_final_assistant_message(message):
            continue
        return str(message.get("content") or "").strip()
    return None


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _suite_run_id() -> str:
    return datetime.now(timezone.utc).strftime("minicode_%Y%m%d_%H%M%S")


def _reset_instance_artifacts(instance_dir: Path) -> None:
    """Clear one prior attempt while leaving its registered worktree removable."""

    for name in (
        "trace.jsonl",
        "checkpoints",
        "approvals",
        "artifacts",
        "memory",
        "result.json",
        "prediction.json",
        "final.patch",
        "workspace-manifest.json",
    ):
        path = instance_dir / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
