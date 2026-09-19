from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from minicode_harness.policy import check_command_allowed
from minicode_harness.state import ApprovalDecision, ApprovalRequest
from minicode_harness.swebench import SweBenchApprovalClient, SweBenchDockerCommandExecutor
from minicode_harness.swebench.docker_executor import _workspace_overlay
from minicode_harness.tools import CommandRunResult, ToolRegistry


class RecordingExecutor:
    sandboxed = False
    command_rules = ()

    def classify(self, argv):
        return check_command_allowed(argv, sandboxed=False, rules=self.command_rules)

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def execute(
        self,
        workspace: Path | str,
        argv: list[str],
        timeout_seconds: int,
        cancellation_token=None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        _ = (cancellation_token, approval_granted)
        normalized_argv = list(argv)
        self.calls.append((str(workspace), normalized_argv, timeout_seconds))
        return CommandRunResult(
            argv=normalized_argv,
            command=" ".join(normalized_argv),
            returncode=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            timeout_seconds=timeout_seconds,
            allowlist_rule="python_pytest",
        )


def test_tool_registry_uses_injected_command_executor(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    registry = ToolRegistry(
        str(tmp_path),
        enable_write=True,
        command_executor=executor,
    )

    result = registry.execute_admitted(
        registry.admit(
            "run_command",
            {"argv": ["python", "-m", "pytest", "-q"], "timeout_seconds": 9},
        )
    )

    assert result.stdout == "ok"
    assert executor.calls == [
        (str(tmp_path), ["python", "-m", "pytest", "-q"], 9)
    ]


def test_swebench_local_approval_rejects_commands_not_proven_safe() -> None:
    client = SweBenchApprovalClient(allow_commands=True, sandboxed_commands=False)
    diagnostic = ApprovalRequest(
        id="diagnostic",
        tool_call_id="call_1",
        tool_name="run_command",
        risk_level="medium",
        preview={"policy_category": "diagnostic"},
    )
    repository_script = ApprovalRequest(
        id="script",
        tool_call_id="call_2",
        tool_name="run_command",
        risk_level="medium",
        preview={"policy_category": "repository_script"},
    )

    assert client.decide(diagnostic).decision == ApprovalDecision.REJECT
    assert client.decide(repository_script).decision == ApprovalDecision.REJECT


def test_swebench_approval_allows_workspace_scripts_only_in_sandbox() -> None:
    client = SweBenchApprovalClient(allow_commands=True, sandboxed_commands=True)
    repository_script = ApprovalRequest(
        id="script",
        tool_call_id="call_1",
        tool_name="run_command",
        risk_level="medium",
        preview={"policy_category": "repository_script"},
    )
    network = ApprovalRequest(
        id="network",
        tool_call_id="call_2",
        tool_name="run_command",
        risk_level="medium",
        preview={"policy_category": "network"},
    )

    assert client.decide(repository_script).decision == ApprovalDecision.APPROVE
    assert client.decide(network).decision == ApprovalDecision.REJECT


def test_docker_executor_rejects_command_before_process_start(tmp_path: Path) -> None:
    executor = SweBenchDockerCommandExecutor(image="example/test:latest")

    with pytest.raises(PermissionError):
        executor.execute(tmp_path, ["bash", "-lc", "whoami"], 10)


def test_docker_executor_builds_unprivileged_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, arguments: list[str], **kwargs: Any) -> None:
            _ = kwargs
            captured.extend(arguments)

        def communicate(self, timeout: float | None = None):
            _ = timeout
            return b"passed", b""

        def poll(self):
            return self.returncode

    (tmp_path / "sample.py").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "sample.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MiniCode Tests",
            "-c",
            "user.email=minicode@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "sample.py").write_text("changed\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("new\n", encoding="utf-8")
    patch_bytes, untracked_paths = _workspace_overlay(tmp_path)
    assert b"+changed" in patch_bytes
    assert untracked_paths == ["new.py"]
    monkeypatch.setattr(
        "minicode_harness.swebench.docker_executor._workspace_overlay",
        lambda root: (patch_bytes, untracked_paths),
    )
    monkeypatch.setattr(
        "minicode_harness.swebench.docker_executor.subprocess.Popen",
        FakeProcess,
    )
    executor = SweBenchDockerCommandExecutor(image="example/test:latest")

    result = executor.execute(tmp_path, ["python", "-m", "pytest", "-q"], 10)

    assert result.returncode == 0
    assert result.stdout == "passed"
    assert captured[:3] == ["docker", "run", "--rm"]
    assert ["--network", "none"] == captured[3:5]
    assert "--mount" in captured
    assert any("target=/workspace,readonly" in value for value in captured)
    assert any("target=/minicode/changes.patch,readonly" in value for value in captured)
    assert "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" in captured
    assert "PYTHONPATH=/testbed" in captured
    image_index = captured.index("example/test:latest")
    assert captured[image_index + 1 : image_index + 3] == ["python", "-c"]
    assert captured[image_index + 4] == "/minicode/changes.patch"
    assert json.loads(captured[image_index + 5]) == ["new.py"]
    assert captured[-4:] == ["python", "-m", "pytest", "-q"]
