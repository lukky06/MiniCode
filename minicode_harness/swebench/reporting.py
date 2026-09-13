"""Markdown and JSON suite reporting for SWE-bench runs."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from .models import SweBenchInstanceResult, SweBenchSuiteSummary
from .prediction import atomic_write_json, atomic_write_text


def build_suite_summary(
    *,
    dataset: str,
    model_name_or_path: str,
    results: list[SweBenchInstanceResult],
) -> SweBenchSuiteSummary:
    """Aggregate official outcomes separately from Harness efficiency facts."""

    total = len({result.instance_id for result in results})
    total_attempts = len(results)
    resolved = sum(result.official_resolved is True for result in results)
    unresolved = sum(result.official_resolved is False for result in results)
    submitted = sum(result.prediction_path is not None for result in results)
    patches = sum(result.patch_bytes > 0 for result in results)
    no_patch = sum(result.status == "no_patch" for result in results)
    evaluation_errors = sum(result.status == "evaluation_error" for result in results)
    evaluated = resolved + unresolved
    by_repository = _group_results(results, key=lambda result: result.repo)
    by_attempt = _group_results(results, key=lambda result: str(result.attempt))
    by_status: dict[str, int] = {}
    for result in results:
        by_status[result.status] = by_status.get(result.status, 0) + 1
    return SweBenchSuiteSummary(
        dataset=dataset,
        model_name_or_path=model_name_or_path,
        total_instances=total,
        total_attempts=total_attempts,
        submitted=submitted,
        patches_generated=patches,
        patch_applied=sum(
            result.status not in {"no_patch", "patch_invalid", "patch_apply_failed"}
            and result.official_resolved is not None
            for result in results
        ),
        resolved=resolved,
        unresolved=unresolved,
        fail_to_pass_success=sum(
            result.fail_to_pass_success is True for result in results
        ),
        pass_to_pass_success=sum(
            result.pass_to_pass_success is True for result in results
        ),
        no_patch=no_patch,
        evaluation_errors=evaluation_errors,
        resolve_rate=resolved / evaluated if evaluated else 0.0,
        avg_steps=_average([result.steps for result in results]),
        avg_tool_calls=_average([result.tool_calls for result in results]),
        avg_elapsed_seconds=_average([result.elapsed_seconds for result in results]),
        avg_evaluation_seconds=_average(
            [result.evaluation_seconds for result in results]
        ),
        avg_context_tokens=_average(
            [int(result.metrics.get("avg_context_tokens") or 0) for result in results]
        ),
        repeated_read_calls=sum(
            int(result.metrics.get("repeated_read_calls") or 0) for result in results
        ),
        project_cache_hits=sum(
            int(result.metrics.get("project_cache_hits") or 0) for result in results
        ),
        verification_commands=sum(
            int(result.metrics.get("verification_commands") or 0) for result in results
        ),
        command_policy_rejections=sum(
            int(result.metrics.get("command_policy_rejections") or 0)
            for result in results
        ),
        context_compactions=sum(
            int(result.metrics.get("context_compactions") or 0) for result in results
        ),
        model_recoveries=sum(
            int(result.metrics.get("model_recoveries") or 0) for result in results
        ),
        by_repository=by_repository,
        by_status=dict(sorted(by_status.items())),
        by_attempt=by_attempt,
        results=results,
    )


def write_suite_reports(
    output_dir: Path | str,
    summary: SweBenchSuiteSummary,
) -> tuple[Path, Path]:
    """Write machine-readable summary and a compact comparison-ready report."""

    root = Path(output_dir)
    summary_path = atomic_write_json(root / "summary.json", summary)
    report_path = atomic_write_text(
        root / "swebench_report.md",
        render_suite_report(summary),
    )
    return summary_path, report_path


def render_suite_report(summary: SweBenchSuiteSummary) -> str:
    """Render official outcomes and Harness metrics without conflating them."""

    lines = [
        f"# SWE-bench Report: {summary.dataset}",
        "",
        f"Model: `{summary.model_name_or_path}`",
        "",
        "## Official Evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total Instances | {summary.total_instances} |",
        f"| Total Attempts | {summary.total_attempts} |",
        f"| Submitted Attempts | {summary.submitted} |",
        f"| Patches Generated | {summary.patches_generated} |",
        f"| Patch Applied | {summary.patch_applied} |",
        f"| Resolved | {summary.resolved} |",
        f"| Unresolved | {summary.unresolved} |",
        f"| Resolve Rate | {summary.resolve_rate:.2%} |",
        f"| FAIL_TO_PASS Success | {summary.fail_to_pass_success} |",
        f"| PASS_TO_PASS Success | {summary.pass_to_pass_success} |",
        f"| No Patch | {summary.no_patch} |",
        f"| Evaluation Errors | {summary.evaluation_errors} |",
        "",
        "## MiniCode Harness Efficiency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Average Steps | {summary.avg_steps:.2f} |",
        f"| Average Tool Calls | {summary.avg_tool_calls:.2f} |",
        f"| Average Context Tokens | {summary.avg_context_tokens:.0f} |",
        f"| Average Agent Time | {summary.avg_elapsed_seconds:.2f}s |",
        f"| Average Evaluation Time | {summary.avg_evaluation_seconds:.2f}s |",
        f"| Repeated Read Calls | {summary.repeated_read_calls} |",
        f"| Project Cache Hits | {summary.project_cache_hits} |",
        f"| Verification Commands | {summary.verification_commands} |",
        f"| Command Policy Rejections | {summary.command_policy_rejections} |",
        f"| Context Compactions | {summary.context_compactions} |",
        f"| Model Recoveries | {summary.model_recoveries} |",
        "",
        "## Instances",
        "",
        "| Instance | Attempt | Repository | Status | Official | Steps | Tools | Patch |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for result in summary.results:
        official = (
            "resolved"
            if result.official_resolved is True
            else "unresolved"
            if result.official_resolved is False
            else "not evaluated"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    result.instance_id,
                    str(result.attempt),
                    result.repo,
                    result.status,
                    official,
                    str(result.steps),
                    str(result.tool_calls),
                    f"{result.patch_bytes} B",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Aggregates by Repository",
            "",
            "| Repository | Attempts | Resolved | Resolve Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for repository, aggregate in summary.by_repository.items():
        lines.append(
            f"| {repository} | {aggregate['attempts']} | {aggregate['resolved']} | "
            f"{aggregate['resolve_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Aggregates by Attempt",
            "",
            "| Attempt | Instances | Resolved | Resolve Rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for attempt, aggregate in summary.by_attempt.items():
        lines.append(
            f"| {attempt} | {aggregate['attempts']} | {aggregate['resolved']} | "
            f"{aggregate['resolve_rate']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def _group_results(
    results: list[SweBenchInstanceResult],
    *,
    key,
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[SweBenchInstanceResult]] = {}
    for result in results:
        buckets.setdefault(str(key(result)), []).append(result)
    aggregates: dict[str, dict[str, float | int]] = {}
    for name in sorted(buckets):
        bucket = buckets[name]
        resolved = sum(result.official_resolved is True for result in bucket)
        evaluated = sum(result.official_resolved is not None for result in bucket)
        aggregates[name] = {
            "attempts": len(bucket),
            "resolved": resolved,
            "resolve_rate": resolved / evaluated if evaluated else 0.0,
        }
    return aggregates


def _average(values: list[int | float]) -> float:
    return float(mean(values)) if values else 0.0
