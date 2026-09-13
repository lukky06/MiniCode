from __future__ import annotations

import time
from pathlib import Path

from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.runtime_tasks import (
    BackgroundCommandManager,
    RuntimeTaskRecord,
    RuntimeTaskRegistry,
)
from minicode_harness.tools import CommandRunResult, ToolRegistry


class FakeCommandExecutor:
    sandboxed = False

    def __init__(self, *, delay: float = 0.0, returncode: int = 0) -> None:
        self.delay = delay
        self.returncode = returncode
        self.calls: list[tuple[list[str], bool]] = []

    def execute(
        self,
        workspace: Path | str,
        argv: list[str],
        timeout_seconds: int,
        cancellation_token: CancellationToken | None = None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        self.calls.append((list(argv), approval_granted))
        deadline = time.monotonic() + self.delay
        while time.monotonic() < deadline:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                return CommandRunResult(
                    argv=list(argv),
                    command=" ".join(argv),
                    returncode=130,
                    stdout="",
                    stderr="Command cancelled.",
                    duration_seconds=self.delay,
                    timeout_seconds=timeout_seconds,
                    allowlist_rule="fake",
                    cancelled=True,
                )
            time.sleep(0.005)
        return CommandRunResult(
            argv=list(argv),
            command=" ".join(argv),
            returncode=self.returncode,
            stdout="ok\n",
            stderr="",
            duration_seconds=self.delay,
            timeout_seconds=timeout_seconds,
            allowlist_rule="fake",
        )


def _wait_for_status(registry: RuntimeTaskRegistry, task_id: str, status: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        payload = registry.status(task_id)
        task = payload["tasks"][0]
        if task["status"] == status:
            return task
        time.sleep(0.01)
    raise AssertionError(f"runtime task {task_id} did not reach {status}")


def test_background_command_returns_immediately_and_writes_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = RuntimeTaskRegistry(max_active=2)
    executor = FakeCommandExecutor(delay=0.05)
    manager = BackgroundCommandManager(
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        command_executor=executor,
        registry=registry,
    )

    started = manager.start(["python", "-m", "pytest"], 10)

    assert started["status"] == "running"
    assert started["runtime_task_id"] == "cmd_0001"
    task = _wait_for_status(registry, "cmd_0001", "completed")
    artifact = tmp_path / "artifacts" / task["artifact_path"]
    assert artifact.is_file()
    assert "returncode: 0" in artifact.read_text(encoding="utf-8")
    notifications = registry.drain_notifications()
    assert len(notifications) == 1
    assert "[MiniCode runtime notification]" in notifications[0]
    assert "returncode: 0" in notifications[0]
    assert registry.drain_notifications() == []
    manager.shutdown()


def test_runtime_task_stop_is_cooperative_and_idempotent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = RuntimeTaskRegistry(max_active=2)
    manager = BackgroundCommandManager(
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        command_executor=FakeCommandExecutor(delay=0.5),
        registry=registry,
    )
    started = manager.start(["python", "-c", "pass"], 10)
    task_id = started["runtime_task_id"]

    assert registry.stop(task_id)["status"] == "stop_requested"
    _wait_for_status(registry, task_id, "stopped")
    assert registry.stop(task_id) == {
        "status": "not_running",
        "task_id": task_id,
        "previous_status": "stopped",
    }
    manager.shutdown()


def test_background_manager_shutdown_does_not_stop_unrelated_runtime_tasks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = RuntimeTaskRegistry(max_active=2)
    stopped: list[str] = []
    registry.register(
        RuntimeTaskRecord(
            id="worker_0001",
            kind="worktree_worker",
            summary="isolated worker",
        ),
        stop_callback=lambda: stopped.append("worker_0001"),
    )
    manager = BackgroundCommandManager(
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        command_executor=FakeCommandExecutor(),
        registry=registry,
    )

    manager.shutdown()

    assert stopped == []
    assert registry.status("worker_0001")["tasks"][0]["status"] == "running"


def test_control_tools_are_stable_without_runtime_handlers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = ToolRegistry(str(workspace), enable_write=True)
    names = {schema["function"]["name"] for schema in tools.schemas()}

    assert {
        "delegate_task",
        "delegate_worktree",
        "runtime_task_status",
        "runtime_task_stop",
    } <= names
    assert tools.execute_admitted(
        tools.admit("delegate_task", {"task": "分析模块"})
    ) == {
        "status": "unavailable",
        "reason": "subagents_disabled",
    }
    assert tools.execute_admitted(tools.admit("runtime_task_status", {})) == {
        "tasks": []
    }
    assert tools.execute_admitted(
        tools.admit("runtime_task_stop", {"task_id": "cmd_9999"})
    ) == {
        "status": "not_running",
        "task_id": "cmd_9999",
    }
    assert tools.requires_approval(
        "runtime_task_stop",
        {"task_id": "cmd_9999"},
    ) is False


def test_tool_registry_explicit_background_command_uses_manager(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = RuntimeTaskRegistry(max_active=2)
    executor = FakeCommandExecutor()
    manager = BackgroundCommandManager(
        workspace=workspace,
        artifact_dir=tmp_path / "artifacts",
        command_executor=executor,
        registry=registry,
    )
    tools = ToolRegistry(
        str(workspace),
        enable_write=True,
        command_executor=executor,
        runtime_task_registry=registry,
        background_command_manager=manager,
    )

    result = tools.execute_admitted(
        tools.admit(
            "run_command",
            {"argv": ["python", "--version"], "background": True},
        )
    )

    assert result["runtime_task_id"] == "cmd_0001"
    assert tools.execute_admitted(tools.admit("runtime_task_status", {}))["tasks"]
    manager.shutdown()
