from __future__ import annotations

import json
from pathlib import Path
import subprocess

from minicode_harness.swebench import SweBenchEvaluator


def test_evaluator_invokes_official_entry_and_parses_report(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "instance_id": "owner__repo-1",
                "model_name_or_path": "model",
                "model_patch": "diff --git ...",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured: list[str] = []

    def fake_run(command, **kwargs):
        captured.extend(command)
        output = Path(kwargs["cwd"])
        (output / "official-report.json").write_text(
            json.dumps(
                {
                    "owner__repo-1": {
                        "resolved": True,
                        "tests_status": {
                            "FAIL_TO_PASS": {"success": True},
                            "PASS_TO_PASS": {"success": True},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    evaluator = SweBenchEvaluator(
        evaluator_python="python",
        output_dir=tmp_path / "evaluation",
        process_runner=fake_run,
    )

    result = evaluator.evaluate(
        predictions,
        "SWE-bench/SWE-bench_Verified",
        ["owner__repo-1"],
        "run-1",
    )

    assert result.status == "completed"
    assert result.instances[0].resolved is True
    assert result.instances[0].fail_to_pass_success is True
    assert result.instances[0].pass_to_pass_success is True
    assert "swebench.harness.run_evaluation" in captured
    assert "--predictions_path" in captured
    assert Path(result.stdout_path or "").read_text(encoding="utf-8") == "ok"


def test_evaluator_preserves_failed_process_logs(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")

    def fake_run(command, **kwargs):
        _ = kwargs
        return subprocess.CompletedProcess(command, 2, stdout="out", stderr="boom")

    evaluator = SweBenchEvaluator(
        evaluator_python="python",
        output_dir=tmp_path / "evaluation",
        process_runner=fake_run,
    )

    result = evaluator.evaluate(
        predictions,
        "dataset",
        [],
        "run-2",
    )

    assert result.status == "error"
    assert result.returncode == 2
    assert result.error == "boom"
    assert Path(result.stderr_path or "").read_text(encoding="utf-8") == "boom"
