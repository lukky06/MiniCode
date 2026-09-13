"""External-process adapter for the official SWE-bench Harness."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from .models import (
    SweBenchEvaluationInstanceResult,
    SweBenchEvaluationResult,
)
from .prediction import atomic_write_text


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class SweBenchEvaluator:
    """Invoke official SWE-bench scoring without copying its oracle logic."""

    def __init__(
        self,
        *,
        evaluator_python: Path | str,
        output_dir: Path | str,
        dataset_split: str = "test",
        evaluation_workers: int = 1,
        evaluation_timeout: int = 3600,
        docker_cache_level: str | None = None,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        if evaluation_workers < 1:
            raise ValueError("evaluation_workers must be at least 1")
        if evaluation_timeout < 1:
            raise ValueError("evaluation_timeout must be at least 1")
        self.evaluator_python = str(evaluator_python)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.dataset_split = dataset_split
        self.evaluation_workers = evaluation_workers
        self.evaluation_timeout = evaluation_timeout
        self.docker_cache_level = docker_cache_level
        self.process_runner = process_runner or subprocess.run

    def with_output_dir(self, output_dir: Path | str) -> "SweBenchEvaluator":
        """Return an equivalent evaluator isolated to another output directory."""

        return SweBenchEvaluator(
            evaluator_python=self.evaluator_python,
            output_dir=output_dir,
            dataset_split=self.dataset_split,
            evaluation_workers=self.evaluation_workers,
            evaluation_timeout=self.evaluation_timeout,
            docker_cache_level=self.docker_cache_level,
            process_runner=self.process_runner,
        )

    def evaluate(
        self,
        predictions_path: Path | str,
        dataset_name: str,
        instance_ids: list[str],
        run_id: str,
    ) -> SweBenchEvaluationResult:
        """Run official evaluation and normalize its persisted reports."""

        predictions = Path(predictions_path).expanduser().resolve()
        if not predictions.is_file():
            raise FileNotFoundError(f"Predictions file does not exist: {predictions}")
        if not dataset_name.strip():
            raise ValueError("dataset_name must not be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.evaluator_python,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            self.dataset_split,
            "--predictions_path",
            str(predictions),
            "--run_id",
            run_id,
            "--max_workers",
            str(self.evaluation_workers),
        ]
        if instance_ids:
            command.extend(["--instance_ids", *instance_ids])
        if self.docker_cache_level:
            command.extend(["--cache_level", self.docker_cache_level])

        started_at = time.monotonic()
        try:
            process = self.process_runner(
                command,
                cwd=self.output_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self.evaluation_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path = atomic_write_text(
                self.output_dir / "evaluation.stdout.log",
                _timeout_stream(exc.stdout),
            )
            stderr_path = atomic_write_text(
                self.output_dir / "evaluation.stderr.log",
                _timeout_stream(exc.stderr),
            )
            return SweBenchEvaluationResult(
                run_id=run_id,
                status="timeout",
                elapsed_seconds=time.monotonic() - started_at,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error=f"Official SWE-bench evaluation timed out after {self.evaluation_timeout}s",
            )

        stdout_path = atomic_write_text(
            self.output_dir / "evaluation.stdout.log",
            process.stdout or "",
        )
        stderr_path = atomic_write_text(
            self.output_dir / "evaluation.stderr.log",
            process.stderr or "",
        )
        report_paths, normalized = self._load_reports(instance_ids)
        if process.returncode != 0:
            return SweBenchEvaluationResult(
                run_id=run_id,
                status="error",
                returncode=process.returncode,
                elapsed_seconds=time.monotonic() - started_at,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                report_paths=[str(path) for path in report_paths],
                instances=normalized,
                error=(process.stderr or process.stdout or "Official evaluator failed").strip(),
            )
        if instance_ids and not normalized:
            return SweBenchEvaluationResult(
                run_id=run_id,
                status="error",
                returncode=process.returncode,
                elapsed_seconds=time.monotonic() - started_at,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                report_paths=[str(path) for path in report_paths],
                error="Official evaluator exited successfully but no instance report was found",
            )
        return SweBenchEvaluationResult(
            run_id=run_id,
            status="completed",
            returncode=process.returncode,
            elapsed_seconds=time.monotonic() - started_at,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            report_paths=[str(path) for path in report_paths],
            instances=normalized,
        )

    def _load_reports(
        self,
        expected_ids: list[str],
    ) -> tuple[list[Path], list[SweBenchEvaluationInstanceResult]]:
        expected = set(expected_ids)
        report_paths: list[Path] = []
        reports: dict[str, SweBenchEvaluationInstanceResult] = {}
        for path in sorted(self.output_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            extracted = _extract_instance_payloads(payload, expected)
            if not extracted:
                continue
            report_paths.append(path)
            for instance_id, details in extracted.items():
                reports[instance_id] = _normalize_instance_report(
                    instance_id,
                    details,
                    path,
                )
        return report_paths, [reports[key] for key in sorted(reports)]


def _extract_instance_payloads(
    payload: Any,
    expected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            instance_id = str(item.get("instance_id") or "")
            if instance_id and (not expected_ids or instance_id in expected_ids):
                extracted[instance_id] = item
        return extracted
    if not isinstance(payload, dict):
        return extracted
    direct_id = str(payload.get("instance_id") or "")
    if direct_id and (not expected_ids or direct_id in expected_ids):
        extracted[direct_id] = payload
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        if expected_ids and key not in expected_ids:
            continue
        if key in expected_ids or "__" in key:
            extracted[key] = value
    return extracted


def _normalize_instance_report(
    instance_id: str,
    details: dict[str, Any],
    report_path: Path,
) -> SweBenchEvaluationInstanceResult:
    resolved = _optional_bool(
        details.get("resolved"),
        details.get("is_resolved"),
    )
    fail_to_pass = _optional_bool(
        details.get("fail_to_pass_success"),
        details.get("FAIL_TO_PASS"),
        _nested(details, "tests_status", "FAIL_TO_PASS"),
    )
    pass_to_pass = _optional_bool(
        details.get("pass_to_pass_success"),
        details.get("PASS_TO_PASS"),
        _nested(details, "tests_status", "PASS_TO_PASS"),
    )
    if resolved is None and fail_to_pass is not None and pass_to_pass is not None:
        resolved = fail_to_pass and pass_to_pass
    status = "resolved" if resolved is True else "unresolved"
    return SweBenchEvaluationInstanceResult(
        instance_id=instance_id,
        status=status,
        resolved=resolved,
        fail_to_pass_success=fail_to_pass,
        pass_to_pass_success=pass_to_pass,
        report_path=str(report_path),
        details=details,
    )


def _optional_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "passed", "pass", "success", "resolved"}:
                return True
            if normalized in {"false", "failed", "fail", "unresolved"}:
                return False
        if isinstance(value, dict):
            nested = _optional_bool(value.get("success"), value.get("passed"))
            if nested is not None:
                return nested
    return None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
