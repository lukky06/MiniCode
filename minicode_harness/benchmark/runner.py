"""Benchmark suite runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import difflib
import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.models import ModelClient, create_model_client
from minicode_harness.policy import check_command_allowed, render_argv
from minicode_harness.report import collect_context_metrics
from minicode_harness.state import (
    ApprovalClient,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStore,
    CheckpointStore,
)
from minicode_harness.tools import run_command
from minicode_harness.trace import TraceWriter

from .models import (
    BenchmarkConstraints,
    BenchmarkOracle,
    BenchmarkSummary,
    BenchmarkTask,
    BenchmarkTaskResult,
    load_benchmark_tasks,
    write_json,
)


ModelClientFactory = Callable[[BenchmarkTask], ModelClient]
BENCHMARK_IGNORED_PATTERNS = (
    "target/**",
    ".git/**",
    "__pycache__/**",
    ".pytest_cache/**",
    "runs/**",
    ".idea/**",
    ".vscode/**",
)
BENCHMARK_IGNORED_NAMES = (
    "target",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "runs",
    ".idea",
    ".vscode",
)


@dataclass(frozen=True)
class BenchmarkRunnerConfig:
    """Runtime settings for benchmark suite execution."""

    provider: str = "openai"
    model: str | None = None
    max_steps: int = 20
    max_tool_calls: int = 30
    repository_memory_enabled: bool = False
    enable_subagents: bool = False


class BenchmarkRunner:
    """Run benchmark task suites in isolated workspaces."""

    def __init__(
        self,
        config: BenchmarkRunnerConfig | None = None,
        *,
        model_client_factory: ModelClientFactory | None = None,
    ) -> None:
        self.config = config or BenchmarkRunnerConfig()
        self.model_client_factory = model_client_factory or self._create_model_client

    def run_suite(self, suite: Path | str, output: Path | str) -> BenchmarkSummary:
        """Run every task in ``suite`` and write benchmark artifacts under ``output``."""

        suite_path = Path(suite).resolve()
        output_path = Path(output).resolve()
        tasks = load_benchmark_tasks(suite_path)
        output_path.mkdir(parents=True, exist_ok=True)

        started_at = time.monotonic()
        results = [
            self._run_task(task, suite_path=suite_path, output_path=output_path)
            for task in tasks
        ]
        summary = _build_summary(
            suite=suite_path.name,
            results=results,
            elapsed_seconds=time.monotonic() - started_at,
        )
        write_json(output_path / "summary.json", summary)
        _write_report(output_path / "benchmark_report.md", summary)
        return summary

    def _run_task(
        self,
        task: BenchmarkTask,
        *,
        suite_path: Path,
        output_path: Path,
    ) -> BenchmarkTaskResult:
        task_output = output_path / "tasks" / task.id
        _remove_tree(task_output)
        task_output.mkdir(parents=True)

        source_workspace = _resolve_workspace(task, suite_path)
        workspace = task_output / "workspace"
        shutil.copytree(
            source_workspace,
            workspace,
            ignore=shutil.ignore_patterns(*BENCHMARK_IGNORED_NAMES),
        )
        trace_writer = TraceWriter(task_output / "trace.jsonl")
        trace_writer.write_event(
            "benchmark_task_started",
            task_id=task.id,
            title=task.title,
            category=task.category,
            workspace=str(workspace),
        )

        started_at = time.monotonic()
        agent_result = None
        oracle_payload: dict[str, Any] = {}
        error: str | None = None
        try:
            self._run_setup(task, workspace, trace_writer)
            _initialize_benchmark_git_baseline(workspace, trace_writer)
            model_client = self.model_client_factory(task)
            agent_result = AgentLoop(
                task=_render_benchmark_task(task.prompt, task.constraints),
                workspace=workspace,
                model_client=model_client,
                trace_writer=trace_writer,
                config=AgentLoopConfig(
                    max_steps=self.config.max_steps,
                    max_tool_calls=self.config.max_tool_calls,
                    rollback_on_unfinished_stop=False,
                    repository_memory_enabled=self.config.repository_memory_enabled,
                    enable_subagents=self.config.enable_subagents,
                ),
                skill_names=task.skills,
                enable_write=True,
                approval_client=BenchmarkApprovalClient(task.constraints),
                approval_store=ApprovalStore(task_output / "approvals"),
                checkpoint_store=CheckpointStore(task_output / "checkpoints"),
                data_dir=task_output / "harness-data",
                enable_long_term_context=False,
                run_id=task.id,
                provider=self.config.provider,
                model=self.config.model or getattr(model_client, "model", None),
            ).run()
            oracle_payload = _evaluate_oracle(
                task,
                workspace=workspace,
                final_text=agent_result.final_text,
                trace_path=trace_writer.trace_path,
                task_output=task_output,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trace_writer.write_event(
                "benchmark_task_error",
                task_id=task.id,
                error=error,
            )

        final_diff_path = task_output / "final.diff"
        diff_text = _write_final_diff(
            source_workspace=source_workspace,
            final_workspace=workspace,
            diff_path=final_diff_path,
        )
        metrics = _trace_metrics(trace_writer.trace_path)
        changed_files = _changed_paths_from_diff(diff_text)
        expected_changed = _expected_files_changed(
            diff_text,
            task.expected.required_modified_files,
        )
        forbidden_changed = _forbidden_file_changed(
            diff_text,
            task.constraints.forbidden_files,
        )
        unexpected_modified_files = _unexpected_modified_files(
            changed_files,
            task.expected.allowed_modified_files,
        )
        max_modified_files_exceeded = (
            task.expected.max_modified_files is not None
            and len(changed_files) > task.expected.max_modified_files
        )
        resolved = (
            bool(oracle_payload.get("passed"))
            and expected_changed
            and not forbidden_changed
            and not unexpected_modified_files
            and not max_modified_files_exceeded
            and error is None
        )
        trace_writer.write_event(
            "benchmark_task_finished",
            task_id=task.id,
            resolved=resolved,
            error=error,
            changed_files=changed_files,
            unexpected_modified_files=unexpected_modified_files,
            max_modified_files_exceeded=max_modified_files_exceeded,
            final_diff_path=str(final_diff_path),
        )
        result = BenchmarkTaskResult(
            id=task.id,
            title=task.title,
            category=task.category,
            status="resolved" if resolved else ("error" if error else "failed"),
            resolved=resolved,
            stop_reason=agent_result.stop_reason if agent_result else None,
            steps=agent_result.steps if agent_result else 0,
            tool_calls=agent_result.tool_calls if agent_result else 0,
            context_tokens=metrics["context_tokens"],
            compression_count=metrics["compression_count"],
            checkpoint_count=metrics["checkpoint_count"],
            read_tool_calls=metrics["read_tool_calls"],
            unique_read_resources=metrics["unique_read_resources"],
            repeated_read_calls=metrics["repeated_read_calls"],
            model_retry_count=metrics["model_retry_count"],
            reactive_compaction_count=metrics["reactive_compaction_count"],
            output_recovery_count=metrics["output_recovery_count"],
            subagent_call_count=metrics["subagent_call_count"],
            mcp_tool_call_count=metrics["mcp_tool_call_count"],
            history_compaction_tokens_removed=metrics[
                "history_compaction_tokens_removed"
            ],
            history_compaction_tokens_retained=metrics[
                "history_compaction_tokens_retained"
            ],
            tool_calls_before_first_write=metrics[
                "tool_calls_before_first_write"
            ],
            unique_reads_before_first_write=metrics[
                "unique_reads_before_first_write"
            ],
            verification_calls_after_first_write=metrics[
                "verification_calls_after_first_write"
            ],
            remaining_tool_calls_at_first_failed_verification=metrics[
                "remaining_tool_calls_at_first_failed_verification"
            ],
            final_test_passed=_final_test_passed(task.oracle, oracle_payload),
            elapsed_seconds=time.monotonic() - started_at,
            oracle=oracle_payload,
            final_text=agent_result.final_text if agent_result else None,
            expected_files_changed=expected_changed,
            forbidden_file_changed=forbidden_changed,
            changed_files=changed_files,
            unexpected_modified_files=unexpected_modified_files,
            modified_file_count=len(changed_files),
            max_modified_files_exceeded=max_modified_files_exceeded,
            workspace=str(workspace),
            output_dir=str(task_output),
            trace_path=str(trace_writer.trace_path),
            final_diff_path=str(final_diff_path),
            error=error,
        )
        write_json(task_output / "result.json", result)
        return result

    def _run_setup(
        self,
        task: BenchmarkTask,
        workspace: Path,
        trace_writer: TraceWriter,
    ) -> None:
        for index, command in enumerate(task.setup.commands, start=1):
            _run_setup_command(
                command=command,
                constraints=task.constraints,
                workspace=workspace,
                trace_writer=trace_writer,
                step=index,
            )

    def _create_model_client(self, task: BenchmarkTask) -> ModelClient:
        _ = task
        return create_model_client(provider=self.config.provider, model=self.config.model)


def _initialize_benchmark_git_baseline(
    workspace: Path,
    trace_writer: TraceWriter,
) -> None:
    """Create an isolated task-local Git baseline for diff inspection."""

    commands = [
        ["git", "init", "--quiet"],
    ]
    for argv in commands:
        try:
            completed = subprocess.run(
                argv,
                cwd=workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("git executable is required for benchmark isolation") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"Unable to create benchmark Git baseline with {render_argv(argv)}: {detail}"
            )

    exclude_path = workspace / ".git" / "info" / "exclude"
    exclude_path.write_text(
        "__pycache__/\n.pytest_cache/\n*.pyc\ntarget/\n",
        encoding="utf-8",
    )

    commands = [
        ["git", "add", "--all"],
        [
            "git",
            "-c",
            "user.name=MiniCode Benchmark",
            "-c",
            "user.email=minicode-benchmark@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "benchmark baseline",
        ],
    ]
    for argv in commands:
        try:
            completed = subprocess.run(
                argv,
                cwd=workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("git executable is required for benchmark isolation") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"Unable to create benchmark Git baseline with {render_argv(argv)}: {detail}"
            )

    trace_writer.write_event(
        "benchmark_git_baseline_created",
        workspace=str(workspace),
    )


def _render_benchmark_task(
    prompt: str,
    constraints: BenchmarkConstraints,
) -> str:
    """Append task-specific execution boundaries to the model-visible benchmark task."""

    sections = [prompt.strip()]
    boundaries: list[str] = []

    if constraints.allowed_commands and constraints.expose_allowed_commands:
        boundaries.append(
            "Allowed validation command patterns:\n"
            + "\n".join(f"- {command}" for command in constraints.allowed_commands)
        )
    elif constraints.allowed_commands:
        boundaries.append(
            "Validation commands are constrained by the benchmark approval policy. "
            "Choose the command from the task, repository, and applicable long-term memory; "
            "the command whitelist is intentionally hidden to avoid leaking evaluation Gold."
        )
    else:
        boundaries.append("Command execution is not allowed for this task.")

    if constraints.allowed_files:
        boundaries.append(
            "Allowed files to modify:\n"
            + "\n".join(f"- {path}" for path in constraints.allowed_files)
        )

    if constraints.forbidden_files:
        boundaries.append(
            "Forbidden files:\n"
            + "\n".join(f"- {path}" for path in constraints.forbidden_files)
        )

    boundaries.append(
        "Follow these execution boundaries exactly. Use the narrowest matching validation "
        "command, and do not retry rejected commands or paths with equivalent variants."
    )
    sections.append("[Benchmark execution constraints]\n" + "\n\n".join(boundaries))
    return "\n\n".join(section for section in sections if section)


def _run_setup_command(
    *,
    command: str,
    constraints: BenchmarkConstraints,
    workspace: Path,
    trace_writer: TraceWriter,
    step: int,
) -> None:
    """Run one benchmark setup command under the declared task constraints."""

    if not _command_allowed_by_constraints(command, constraints.allowed_commands):
        raise PermissionError(
            f"Setup command is not allowed by task constraints: {command}"
        )
    result = run_command(workspace, command)
    trace_writer.write_event(
        "benchmark_setup_command",
        step=step,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Setup command failed: {command}")


class BenchmarkApprovalClient(ApprovalClient):
    """Auto-approve only tool calls that satisfy benchmark task constraints."""

    def __init__(self, constraints: BenchmarkConstraints) -> None:
        self.constraints = constraints
        self.requests: list[ApprovalRequest] = []

    def decide(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        allowed, reason = _approval_allowed(request, self.constraints)
        if allowed:
            return ApprovalResponse(decision=ApprovalDecision.APPROVE, reason="benchmark_auto_approved")
        return ApprovalResponse(decision=ApprovalDecision.REJECT, reason=reason)


def _approval_allowed(
    request: ApprovalRequest,
    constraints: BenchmarkConstraints,
) -> tuple[bool, str | None]:
    if request.tool_name == "run_command":
        argv = list(request.arguments.get("argv") or [])
        command = render_argv(argv)
        if _command_allowed_by_constraints(argv, constraints.allowed_commands):
            return True, None
        allowed = _render_allowed_values(
            constraints.allowed_commands,
            empty_message="no command execution is allowed",
        )
        return (
            False,
            f"Command is outside task constraints: {command}. "
            f"Allowed command patterns: {allowed}. "
            "Choose one matching pattern; do not retry equivalent command variants.",
        )

    if request.tool_name in {"edit", "write"}:
        path = str(request.arguments.get("path", ""))
        return _path_allowed(path, constraints)

    if request.tool_name == "apply_patch":
        files = [str(path) for path in request.preview.get("files", [])]
        for path in files:
            allowed, reason = _path_allowed(path, constraints)
            if not allowed:
                return False, reason
        return True, None

    return False, f"Tool is not auto-approved in benchmark mode: {request.tool_name}"


def _command_allowed_by_constraints(argv: list[str], allowed_commands: list[str]) -> bool:
    policy_result = check_command_allowed(argv)
    if not policy_result.allowed:
        return False
    command = render_argv(policy_result.argv)
    return command in allowed_commands or bool(policy_result.rule and policy_result.rule in allowed_commands)


def _path_allowed(path: str, constraints: BenchmarkConstraints) -> tuple[bool, str | None]:
    normalized = path.replace("\\", "/")
    allowed = _render_allowed_values(
        constraints.allowed_files,
        empty_message="no file allowlist is declared",
    )
    if _matches_any(normalized, constraints.forbidden_files):
        forbidden = _render_allowed_values(
            constraints.forbidden_files,
            empty_message="none",
        )
        return (
            False,
            f"Path is forbidden by task constraints: {path}. "
            f"Forbidden patterns: {forbidden}. Allowed files: {allowed}.",
        )
    if constraints.allowed_files and not _matches_any(normalized, constraints.allowed_files):
        return (
            False,
            f"Path is outside task allowed files: {path}. Allowed files: {allowed}.",
        )
    return True, None


def _render_allowed_values(values: list[str], *, empty_message: str) -> str:
    return ", ".join(values) if values else empty_message


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _resolve_workspace(task: BenchmarkTask, suite_path: Path) -> Path:
    if task.source_path is not None:
        base = task.source_path.parent
    else:
        base = suite_path
    workspace = Path(task.workspace)
    if not workspace.is_absolute():
        workspace = base / workspace
    resolved = workspace.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Task workspace does not exist: {resolved}")
    return resolved


def _evaluate_oracle(
    task: BenchmarkTask,
    *,
    workspace: Path,
    final_text: str | None,
    trace_path: Path,
    task_output: Path,
    grader_source_path: Path | None = None,
) -> dict[str, Any]:
    oracle = task.oracle
    if oracle.type == "none":
        return {"type": oracle.type, "passed": True}

    if oracle.type == "command":
        result = _run_grader_command(workspace, oracle.command)
        return _command_oracle_payload(oracle.type, result)

    if oracle.type == "hidden_command":
        grading_workspace = _fresh_grading_workspace(
            workspace,
            task_output / "grading" / "hidden",
        )
        try:
            _copy_grader_assets(
                grader_source_path or task.source_path,
                oracle.assets,
                grading_workspace,
            )
            result = _run_grader_command(grading_workspace, oracle.command)
            payload = _command_oracle_payload(oracle.type, result)
            payload["assets"] = [asset.model_dump(mode="json") for asset in oracle.assets]
            return payload
        finally:
            _remove_tree(task_output / "grading", ignore_errors=True)

    if oracle.type == "structured_final":
        parsed, parse_error = _parse_structured_final(final_text or "")
        mismatches = _structured_mismatches(
            parsed,
            oracle.expected_fields,
            allow_extra_fields=oracle.allow_extra_fields,
        )
        return {
            "type": oracle.type,
            "passed": parse_error is None and not mismatches,
            "parsed": parsed,
            "expected_fields": oracle.expected_fields,
            "allow_extra_fields": oracle.allow_extra_fields,
            "parse_error": parse_error,
            "mismatches": mismatches,
        }

    if oracle.type == "mutation_test":
        baseline_workspace = _fresh_grading_workspace(
            workspace,
            task_output / "grading" / "baseline",
        )
        baseline = _run_grader_command(baseline_workspace, oracle.command)
        baseline_payload = _baseline_test_payload(baseline_workspace, baseline)
        mutation_results: list[dict[str, Any]] = []
        try:
            for mutation in oracle.mutations:
                mutation_workspace = _fresh_grading_workspace(
                    workspace,
                    task_output / "grading" / "mutations" / mutation.id,
                )
                _copy_grader_assets(
                    grader_source_path or task.source_path,
                    mutation.assets,
                    mutation_workspace,
                )
                result = _run_grader_command(
                    mutation_workspace,
                    mutation.command or oracle.command,
                )
                mutation_results.append(
                    _mutation_test_payload(mutation.id, mutation_workspace, result)
                )
            return {
                "type": oracle.type,
                "passed": baseline_payload["passed"]
                and all(result["outcome"] == "killed" for result in mutation_results),
                "baseline": baseline_payload,
                "mutations": mutation_results,
                "killed": sum(
                    1 for result in mutation_results if result["outcome"] == "killed"
                ),
                "survived": sum(
                    1 for result in mutation_results if result["outcome"] == "survived"
                ),
                "inconclusive": sum(
                    1
                    for result in mutation_results
                    if result["outcome"] == "inconclusive"
                ),
                "total_mutations": len(mutation_results),
            }
        finally:
            _remove_tree(task_output / "grading", ignore_errors=True)

    if oracle.type == "final_text_keywords":
        text = final_text or ""
        missing = [
            keyword
            for keyword in oracle.required_keywords
            if keyword.lower() not in text.lower()
        ]
        return {
            "type": oracle.type,
            "passed": not missing,
            "required_keywords": oracle.required_keywords,
            "missing_keywords": missing,
        }

    events = _read_trace_events(trace_path)
    matched = []
    for event in events:
        if oracle.event_type and event.get("type") != oracle.event_type:
            continue
        if all(event.get(key) == value for key, value in oracle.required_fields.items()):
            matched.append(event)
    return {
        "type": oracle.type,
        "passed": bool(matched),
        "event_type": oracle.event_type,
        "required_fields": oracle.required_fields,
        "match_count": len(matched),
    }


def _fresh_grading_workspace(source: Path, destination: Path) -> Path:
    _remove_tree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*BENCHMARK_IGNORED_NAMES),
    )
    return destination


def _remove_tree(
    path: Path,
    *,
    ignore_errors: bool = False,
    attempts: int = 5,
    base_delay_seconds: float = 0.05,
) -> None:
    """Remove a benchmark directory with bounded Windows handle-release retries."""

    if not path.exists():
        return

    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path, onerror=_make_writable_and_retry)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(base_delay_seconds * (attempt + 1))

    if ignore_errors:
        return
    assert last_error is not None
    raise last_error


def _make_writable_and_retry(function: Callable[..., Any], path: str, exc_info: Any) -> None:
    """Clear read-only attributes before one ``shutil.rmtree`` retry."""

    _ = exc_info
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _copy_grader_assets(
    source_path: Path | None,
    assets: list[Any],
    workspace: Path,
) -> None:
    if source_path is None:
        raise ValueError("Grader assets require a benchmark source_path.")
    suite_root = source_path.parent.resolve()
    workspace_root = workspace.resolve()
    for asset in assets:
        source = (suite_root / asset.source).resolve()
        if not source.is_relative_to(suite_root):
            raise PermissionError(f"Grader asset escapes the suite: {asset.source}")
        if not source.is_file():
            raise FileNotFoundError(f"Grader asset does not exist: {source}")
        destination = (workspace_root / asset.destination).resolve()
        if not destination.is_relative_to(workspace_root):
            raise PermissionError(
                f"Grader asset destination escapes the grading workspace: {asset.destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_grader_command(workspace: Path, command: str | list[str] | None):
    if not command:
        raise ValueError("Command-based oracle requires oracle.command.")
    return run_command(workspace, command)


def _command_oracle_payload(oracle_type: str, result: Any) -> dict[str, Any]:
    return {
        "type": oracle_type,
        "command": result.command,
        "argv": result.argv,
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": _truncate(result.stdout, 4000),
        "stderr": _truncate(result.stderr, 4000),
    }


def _baseline_test_payload(workspace: Path, result: Any) -> dict[str, Any]:
    evidence = _collect_test_execution_evidence(workspace, result)
    passed = (
        result.returncode == 0
        and evidence["test_execution_confirmed"]
        and evidence["failures"] == 0
        and evidence["errors"] == 0
    )
    if passed:
        reason = "baseline_tests_passed"
    elif not evidence["test_execution_confirmed"]:
        reason = "baseline_test_execution_not_confirmed"
    else:
        reason = "baseline_tests_failed"
    return {
        "passed": passed,
        "command": result.command,
        "argv": result.argv,
        "returncode": result.returncode,
        "stdout": _truncate(result.stdout, 4000),
        "stderr": _truncate(result.stderr, 4000),
        "reason": reason,
        **evidence,
    }


def _mutation_test_payload(mutation_id: str, workspace: Path, result: Any) -> dict[str, Any]:
    evidence = _collect_test_execution_evidence(workspace, result)
    failures = int(evidence["failures"])
    errors = int(evidence["errors"])
    if not evidence["test_execution_confirmed"]:
        outcome = "inconclusive"
        reason = "test_execution_not_confirmed"
    elif result.returncode == 0 and failures == 0 and errors == 0:
        outcome = "survived"
        reason = "mutated_implementation_passed_tests"
    elif result.returncode != 0 and failures + errors > 0:
        outcome = "killed"
        reason = "executed_tests_failed_on_mutation"
    else:
        outcome = "inconclusive"
        reason = "non_test_failure"
    return {
        "id": mutation_id,
        "outcome": outcome,
        "killed": outcome == "killed",
        "command": result.command,
        "argv": result.argv,
        "returncode": result.returncode,
        "stdout": _truncate(result.stdout, 4000),
        "stderr": _truncate(result.stderr, 4000),
        "reason": reason,
        **evidence,
    }


def _collect_test_execution_evidence(workspace: Path, result: Any) -> dict[str, Any]:
    junit = _read_junit_test_counts(workspace)
    if junit is not None:
        return junit

    output = f"{result.stdout}\n{result.stderr}"
    pytest_counts = _parse_pytest_counts(output)
    if pytest_counts is not None:
        return pytest_counts

    unittest_counts = _parse_unittest_counts(output)
    if unittest_counts is not None:
        return unittest_counts

    maven_counts = _parse_maven_console_counts(output)
    if maven_counts is not None:
        return maven_counts

    return {
        "framework": "unknown",
        "test_execution_confirmed": False,
        "tests_run": 0,
        "tests_executed": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


def _read_junit_test_counts(workspace: Path) -> dict[str, Any] | None:
    report_roots = (
        workspace / "target" / "surefire-reports",
        workspace / "target" / "failsafe-reports",
        workspace / "build" / "test-results",
    )
    reports: list[Path] = []
    for root in report_roots:
        if root.is_dir():
            reports.extend(sorted(root.rglob("TEST-*.xml")))
    if not reports:
        return None

    tests = failures = errors = skipped = 0
    parsed_reports = 0
    for report in reports:
        try:
            root = ET.parse(report).getroot()
        except (ET.ParseError, OSError):
            continue
        suites = [root]
        if _xml_local_name(root.tag) == "testsuites" and not root.attrib.get("tests"):
            suites = [
                child
                for child in root
                if _xml_local_name(child.tag) == "testsuite"
            ]
        for suite in suites:
            if _xml_local_name(suite.tag) not in {"testsuite", "testsuites"}:
                continue
            tests += _int_attribute(suite, "tests")
            failures += _int_attribute(suite, "failures")
            errors += _int_attribute(suite, "errors")
            skipped += _int_attribute(suite, "skipped")
        parsed_reports += 1
    if parsed_reports == 0:
        return None

    executed = max(0, tests - skipped)
    return {
        "framework": "junit_xml",
        "test_execution_confirmed": executed > 0,
        "tests_run": tests,
        "tests_executed": executed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "report_files": parsed_reports,
    }


def _parse_pytest_counts(output: str) -> dict[str, Any] | None:
    passed = _last_regex_count(output, r"(\d+)\s+passed\b")
    failed = _last_regex_count(output, r"(\d+)\s+failed\b")
    errors = _last_regex_count(output, r"(\d+)\s+errors?\b")
    skipped = _last_regex_count(output, r"(\d+)\s+skipped\b")
    if passed + failed + errors + skipped == 0:
        return None
    executed = passed + failed
    return {
        "framework": "pytest",
        "test_execution_confirmed": executed > 0,
        "tests_run": executed + errors + skipped,
        "tests_executed": executed,
        "failures": failed,
        "errors": errors,
        "skipped": skipped,
    }


def _parse_unittest_counts(output: str) -> dict[str, Any] | None:
    ran_matches = re.findall(r"Ran\s+(\d+)\s+tests?", output)
    if not ran_matches:
        return None
    tests = int(ran_matches[-1])
    failures = _last_named_failure_count(output, "failures")
    errors = _last_named_failure_count(output, "errors")
    skipped = _last_named_failure_count(output, "skipped")
    return {
        "framework": "unittest",
        "test_execution_confirmed": tests - skipped > 0,
        "tests_run": tests,
        "tests_executed": max(0, tests - skipped),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _parse_maven_console_counts(output: str) -> dict[str, Any] | None:
    matches = re.findall(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),"
        r"\s*Skipped:\s*(\d+)",
        output,
    )
    if not matches:
        return None
    tests, failures, errors, skipped = (int(value) for value in matches[-1])
    executed = max(0, tests - skipped)
    return {
        "framework": "maven_console",
        "test_execution_confirmed": executed > 0,
        "tests_run": tests,
        "tests_executed": executed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _last_regex_count(output: str, pattern: str) -> int:
    matches = re.findall(pattern, output, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else 0


def _last_named_failure_count(output: str, name: str) -> int:
    matches = re.findall(rf"\b{name}=(\d+)\b", output, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else 0


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _int_attribute(element: ET.Element, name: str) -> int:
    try:
        return int(element.attrib.get(name, "0"))
    except ValueError:
        return 0


def _parse_structured_final(text: str) -> tuple[dict[str, Any] | None, str | None]:
    blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if len(blocks) != 1:
        return None, f"Expected exactly one JSON code block, found {len(blocks)}."
    try:
        parsed = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}."
    if not isinstance(parsed, dict):
        return None, "Structured final JSON must be an object."
    return parsed, None


def _structured_mismatches(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    allow_extra_fields: bool,
) -> list[dict[str, Any]]:
    if actual is None:
        return [{"path": "$", "reason": "missing_object"}]
    mismatches: list[dict[str, Any]] = []
    _compare_structured_value(actual, expected, "$", mismatches)
    if not allow_extra_fields:
        extra = sorted(set(actual) - set(expected))
        for key in extra:
            mismatches.append({"path": f"$.{key}", "reason": "unexpected_field"})
    return mismatches


def _compare_structured_value(
    actual: Any,
    expected: Any,
    path: str,
    mismatches: list[dict[str, Any]],
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append(
                {"path": path, "reason": "type_mismatch", "expected": expected, "actual": actual}
            )
            return
        for key, expected_value in expected.items():
            if key not in actual:
                mismatches.append({"path": f"{path}.{key}", "reason": "missing_field"})
                continue
            _compare_structured_value(actual[key], expected_value, f"{path}.{key}", mismatches)
        return
    if actual != expected:
        mismatches.append(
            {"path": path, "reason": "value_mismatch", "expected": expected, "actual": actual}
        )


def _write_final_diff(source_workspace: Path, final_workspace: Path, diff_path: Path) -> str:
    source_files = _text_files(source_workspace)
    final_files = _text_files(final_workspace)
    all_paths = sorted(set(source_files) | set(final_files))
    diff_parts: list[str] = []
    for relative_path in all_paths:
        before = source_files.get(relative_path, "").splitlines(keepends=True)
        after = final_files.get(relative_path, "").splitlines(keepends=True)
        if before == after:
            continue
        diff_parts.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
    diff_text = "".join(diff_parts)
    diff_path.write_text(diff_text, encoding="utf-8")
    return diff_text


def _text_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _matches_any(relative, BENCHMARK_IGNORED_PATTERNS):
            continue
        try:
            files[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def _forbidden_file_changed(diff_text: str, forbidden_patterns: list[str]) -> bool:
    return any(
        _matches_any(path, forbidden_patterns)
        for path in _changed_paths_from_diff(diff_text)
    )


def _expected_files_changed(diff_text: str, expected_patterns: list[str]) -> bool:
    if not expected_patterns:
        return True
    changed_paths = _changed_paths_from_diff(diff_text)
    return all(
        any(fnmatch.fnmatch(path, pattern) for path in changed_paths)
        for pattern in expected_patterns
    )


def _unexpected_modified_files(
    changed_paths: list[str],
    allowed_patterns: list[str],
) -> list[str]:
    if not allowed_patterns:
        return []
    return [
        path
        for path in changed_paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in allowed_patterns)
    ]


def _changed_paths_from_diff(diff_text: str) -> list[str]:
    changed_paths: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        path = line[4:].strip()
        if path == "/dev/null":
            continue
        normalized = path.removeprefix("a/").removeprefix("b/")
        changed_paths.append(normalized)
    return sorted(set(changed_paths))


def _trace_metrics(trace_path: Path) -> dict[str, int | None]:
    events = _read_trace_events(trace_path)
    context_metrics = collect_context_metrics(events)
    convergence_metrics = _convergence_metrics(events)
    read_tool_names = {
        "read",
        "search",
        "task",
        "runtime_task_status",
        "delegate_task",
    }
    return {
        "context_tokens": context_metrics.context_token_estimate_avg,
        "compression_count": sum(1 for event in events if event.get("type") == "context_compressed"),
        "checkpoint_count": sum(1 for event in events if event.get("type") == "checkpoint_saved"),
        "read_tool_calls": sum(
            1
            for event in events
            if event.get("type") == "tool_called" and event.get("tool") in read_tool_names
        ),
        "unique_read_resources": context_metrics.unique_read_resources,
        "repeated_read_calls": context_metrics.repeated_read_calls,
        "model_retry_count": sum(
            1 for event in events if event.get("type") == "model_retry_scheduled"
        ),
        "reactive_compaction_count": sum(
            1
            for event in events
            if event.get("type") == "model_recovery"
            and event.get("action") == "reactive_compact"
        ),
        "output_recovery_count": sum(
            1
            for event in events
            if event.get("type") == "model_recovery"
            and event.get("action")
            in {
                "increase_max_output_tokens",
                "continue_truncated_output",
                "continue_visible_truncated_output",
            }
        ),
        "subagent_call_count": sum(
            1 for event in events if event.get("type") == "subagent_started"
        ),
        "mcp_tool_call_count": sum(
            1
            for event in events
            if event.get("type") == "tool_called"
            and str(event.get("tool") or "").startswith("mcp__")
        ),
        "history_compaction_tokens_removed": (
            context_metrics.history_compaction_tokens_removed
        ),
        "history_compaction_tokens_retained": (
            context_metrics.history_compaction_tokens_retained
        ),
        **convergence_metrics,
    }


def _convergence_metrics(events: list[dict[str, Any]]) -> dict[str, int | None]:
    """Measure budgeted exploration and verification around the first write."""

    meta_tools = {"task"}
    write_tools = {"apply_patch", "edit", "write"}
    first_write_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.get("type") == "tool_called"
            and event.get("tool") in write_tools
        ),
        None,
    )
    if first_write_index is None:
        return {
            "tool_calls_before_first_write": None,
            "unique_reads_before_first_write": None,
            "verification_calls_after_first_write": 0,
            "remaining_tool_calls_at_first_failed_verification": None,
        }

    before_first_write = events[:first_write_index]
    prefix_metrics = collect_context_metrics(before_first_write)
    tool_calls_before_first_write = sum(
        1
        for event in before_first_write
        if event.get("type") == "tool_called"
        and event.get("tool") not in meta_tools
    )
    after_first_write = events[first_write_index + 1 :]
    verification_calls = [
        event
        for event in after_first_write
        if event.get("type") == "tool_called" and event.get("tool") == "run_command"
    ]

    tool_call_ordinals: dict[str, int] = {}
    ordinal = 0
    for event in events:
        if event.get("type") != "tool_called" or event.get("tool") in meta_tools:
            continue
        ordinal += 1
        tool_call_ordinals[str(event.get("tool_call_id") or "")] = ordinal

    first_context = next(
        (event for event in events if event.get("type") == "context_built"),
        {},
    )
    max_tool_calls = int(first_context.get("tool_calls_used") or 0) + int(
        first_context.get("remaining_tool_calls") or 0
    )
    remaining_at_failure: int | None = None
    for event in after_first_write:
        if (
            event.get("type") == "tool_result"
            and event.get("tool") == "run_command"
            and event.get("status") in {
                "command_failed",
                "command_timed_out",
                "command_cancelled",
                "error",
            }
        ):
            call_ordinal = tool_call_ordinals.get(str(event.get("tool_call_id") or ""))
            if call_ordinal is not None and max_tool_calls > 0:
                remaining_at_failure = max(0, max_tool_calls - call_ordinal)
            break

    return {
        "tool_calls_before_first_write": tool_calls_before_first_write,
        "unique_reads_before_first_write": prefix_metrics.unique_read_resources,
        "verification_calls_after_first_write": len(verification_calls),
        "remaining_tool_calls_at_first_failed_verification": remaining_at_failure,
    }


def _read_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _build_summary(
    *,
    suite: str,
    results: list[BenchmarkTaskResult],
    elapsed_seconds: float,
) -> BenchmarkSummary:
    total = len(results)
    resolved = sum(1 for result in results if result.resolved)
    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = by_category.setdefault(
            result.category,
            {"total": 0, "resolved": 0, "resolve_rate": 0.0},
        )
        bucket["total"] += 1
        if result.resolved:
            bucket["resolved"] += 1
    for bucket in by_category.values():
        bucket["resolve_rate"] = bucket["resolved"] / bucket["total"] if bucket["total"] else 0.0

    return BenchmarkSummary(
        suite=suite,
        total_tasks=total,
        resolved=resolved,
        failed=total - resolved,
        resolve_rate=resolved / total if total else 0.0,
        by_category=by_category,
        avg_steps=_average([result.steps for result in results]),
        avg_tool_calls=_average([result.tool_calls for result in results]),
        avg_context_tokens=_average([result.context_tokens for result in results]),
        compression_count=sum(result.compression_count for result in results),
        checkpoint_count=sum(result.checkpoint_count for result in results),
        read_tool_calls=sum(result.read_tool_calls for result in results),
        unique_read_resources=sum(result.unique_read_resources for result in results),
        repeated_read_calls=sum(result.repeated_read_calls for result in results),
        model_retry_count=sum(result.model_retry_count for result in results),
        reactive_compaction_count=sum(
            result.reactive_compaction_count for result in results
        ),
        output_recovery_count=sum(result.output_recovery_count for result in results),
        subagent_call_count=sum(result.subagent_call_count for result in results),
        mcp_tool_call_count=sum(result.mcp_tool_call_count for result in results),
        history_compaction_tokens_removed=sum(
            result.history_compaction_tokens_removed for result in results
        ),
        history_compaction_tokens_retained=sum(
            result.history_compaction_tokens_retained for result in results
        ),
        avg_tool_calls_before_first_write=_average_optional(
            [result.tool_calls_before_first_write for result in results]
        ),
        avg_unique_reads_before_first_write=_average_optional(
            [result.unique_reads_before_first_write for result in results]
        ),
        verification_calls_after_first_write=sum(
            result.verification_calls_after_first_write for result in results
        ),
        avg_remaining_tool_calls_at_first_failed_verification=_average_optional(
            [
                result.remaining_tool_calls_at_first_failed_verification
                for result in results
            ]
        ),
        final_test_pass_rate=_final_test_pass_rate(results),
        elapsed_seconds=elapsed_seconds,
        tasks=results,
    )


def _write_report(path: Path, summary: BenchmarkSummary) -> None:
    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Suite: `{summary.suite}`",
        f"- Total Tasks: {summary.total_tasks}",
        f"- Resolved: {summary.resolved}",
        f"- Failed: {summary.failed}",
        f"- Resolve Rate: {summary.resolve_rate:.0%}",
        f"- Avg Steps: {summary.avg_steps:.2f}",
        f"- Avg Tool Calls: {summary.avg_tool_calls:.2f}",
        f"- Avg Context Tokens: {summary.avg_context_tokens:.2f}",
        f"- Compression Count: {summary.compression_count}",
        f"- Checkpoint Count: {summary.checkpoint_count}",
        f"- Elapsed Seconds: {summary.elapsed_seconds:.2f}",
        "",
        "## Context Management Metrics",
        "",
        f"- Read Tool Calls: {summary.read_tool_calls}",
        f"- Unique Read Resources: {summary.unique_read_resources}",
        f"- Repeated Read Calls: {summary.repeated_read_calls}",
        f"- Model Retries: {summary.model_retry_count}",
        f"- Reactive Compactions: {summary.reactive_compaction_count}",
        f"- Output Recoveries: {summary.output_recovery_count}",
        f"- Subagent Calls: {summary.subagent_call_count}",
        f"- MCP Tool Calls: {summary.mcp_tool_call_count}",
        f"- History Compaction Tokens Removed: {summary.history_compaction_tokens_removed}",
        f"- History Compaction Tokens Retained: {summary.history_compaction_tokens_retained}",
        f"- Final Test Pass Rate: {summary.final_test_pass_rate:.0%}",
        "",
        "## Convergence Metrics",
        "",
        f"- Avg Tool Calls Before First Write: {summary.avg_tool_calls_before_first_write:.2f}",
        f"- Avg Unique Reads Before First Write: {summary.avg_unique_reads_before_first_write:.2f}",
        f"- Verification Calls After First Write: {summary.verification_calls_after_first_write}",
        "- Avg Remaining Tool Calls at First Failed Verification: "
        f"{summary.avg_remaining_tool_calls_at_first_failed_verification:.2f}",
        "",
        "These fields are stable comparison points for baseline and optimized context-management runs.",
        "",
        "## Context Metrics by Task",
        "",
        "| Task | Reads | Unique Reads | Repeated Reads | Before Write | Unique Before Write | Verify After Write | Remaining at First Verify Failure | Final Test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in summary.tasks:
        final_test = (
            "n/a" if result.final_test_passed is None else ("pass" if result.final_test_passed else "fail")
        )
        before_write = (
            "n/a"
            if result.tool_calls_before_first_write is None
            else str(result.tool_calls_before_first_write)
        )
        unique_before_write = (
            "n/a"
            if result.unique_reads_before_first_write is None
            else str(result.unique_reads_before_first_write)
        )
        remaining_at_failure = (
            "n/a"
            if result.remaining_tool_calls_at_first_failed_verification is None
            else str(result.remaining_tool_calls_at_first_failed_verification)
        )
        lines.append(
            f"| `{result.id}` | {result.read_tool_calls} | "
            f"{result.unique_read_resources} | {result.repeated_read_calls} | "
            f"{before_write} | {unique_before_write} | "
            f"{result.verification_calls_after_first_write} | "
            f"{remaining_at_failure} | {final_test} |"
        )
    lines.extend(
        [
            "",
            "## Breakdown by Category",
            "",
            "| Category | Resolved | Total | Resolve Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for category, bucket in sorted(summary.by_category.items()):
        lines.append(
            f"| {category} | {bucket['resolved']} | {bucket['total']} | {bucket['resolve_rate']:.0%} |"
        )
    lines.extend(["", "## Failed Cases", ""])
    failed = [result for result in summary.tasks if not result.resolved]
    if not failed:
        lines.append("No failed cases.")
    else:
        for result in failed:
            lines.append(f"- `{result.id}`: {result.status} ({result.error or result.stop_reason})")
    lines.extend(["", "## Configuration", ""])
    lines.append("Benchmark tasks ran in isolated copied workspaces with task-constrained auto-approval.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _final_test_passed(
    oracle: BenchmarkOracle,
    oracle_payload: dict[str, Any],
) -> bool | None:
    if oracle.type not in {"command", "hidden_command", "mutation_test"}:
        return None
    if not oracle.command:
        return None
    command_text = (
        oracle.command
        if isinstance(oracle.command, str)
        else render_argv(oracle.command)
    )
    if "test" not in command_text.lower() and "pytest" not in command_text.lower():
        return None
    return bool(oracle_payload.get("passed"))


def _final_test_pass_rate(results: list[BenchmarkTaskResult]) -> float:
    test_results = [result for result in results if result.final_test_passed is not None]
    if not test_results:
        return 0.0
    return sum(1 for result in test_results if result.final_test_passed) / len(test_results)


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _average_optional(values: list[int | None]) -> float:
    present = [value for value in values if value is not None]
    return _average(present)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"
