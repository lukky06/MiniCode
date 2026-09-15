from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any
import subprocess
import time

import pytest

from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.runtime_tasks import RuntimeTaskRegistry
from minicode_harness.trace import TraceWriter
from minicode_harness.tools import LocalCommandExecutor
from minicode_harness.worktree_worker import (
    WorktreeWorkerManager,
    WorktreeWorkerResult,
    WorktreeWorkerRunner,
)
from minicode_harness.worktrees import GitWorktreeManager


class _Capabilities:
    context_window = 32_000
    reserved_output_tokens = 6_000


class _DummyModel:
    capabilities = _Capabilities()


class _ScriptedModel(ModelClient):
    capabilities = _Capabilities()

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def call_request(self, request):
        self.calls.append((request.as_chat_messages(), request.tools))
        return self.responses.pop(0)


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {str(tool["function"]["name"]) for tool in tools}


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.name", "MiniCode Tests")
    _git(path, "config", "user.email", "minicode@example.invalid")
    (path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "add", "app.py")
    _git(path, "commit", "-m", "initial")
    return path


def _wait(registry: RuntimeTaskRegistry, task_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        task = registry.status(task_id)["tasks"][0]
        if task["status"] != "running":
            return task
        time.sleep(0.01)
    raise AssertionError("worker did not finish")


def test_worker_rejects_dirty_parent_before_creating_worktree(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    (repository / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    manager = WorktreeWorkerManager(
        parent_workspace=repository,
        run_id="run_test",
        model_client=_DummyModel(),
        registry=RuntimeTaskRegistry(),
        artifact_dir=tmp_path / "artifacts",
        worktree_manager=GitWorktreeManager(tmp_path / "data"),
    )

    with pytest.raises(RuntimeError, match="Parent workspace is dirty"):
        manager.start("change app")
    assert GitWorktreeManager(tmp_path / "data").list(repository) == []
    manager.shutdown()


def test_worker_manager_propagates_parent_command_executor(tmp_path: Path, monkeypatch) -> None:
    repository = _repository(tmp_path / "repo")
    registry = RuntimeTaskRegistry()
    worktrees = GitWorktreeManager(tmp_path / "data")
    command_executor = LocalCommandExecutor()
    observed = []

    def fake_run(self, task: str) -> WorktreeWorkerResult:
        del task
        observed.append(self.tools.command_executor)
        return WorktreeWorkerResult(
            status="completed",
            summary="No change needed.",
            steps=1,
            tool_calls=0,
        )

    monkeypatch.setattr(WorktreeWorkerRunner, "run", fake_run)
    manager = WorktreeWorkerManager(
        parent_workspace=repository,
        run_id="run_executor",
        model_client=_DummyModel(),
        registry=registry,
        artifact_dir=tmp_path / "artifacts",
        command_executor=command_executor,
        worktree_manager=worktrees,
    )

    started = manager.start("inspect app")
    task = _wait(registry, started["runtime_task_id"])

    assert task["status"] == "completed"
    assert observed == [command_executor]
    manager.shutdown()


def test_clean_worker_is_removed_after_completion(tmp_path: Path, monkeypatch) -> None:
    repository = _repository(tmp_path / "repo")
    registry = RuntimeTaskRegistry()
    worktrees = GitWorktreeManager(tmp_path / "data")

    def fake_run(self, task: str) -> WorktreeWorkerResult:
        return WorktreeWorkerResult(
            status="completed",
            summary="No change needed.",
            steps=1,
            tool_calls=0,
        )

    monkeypatch.setattr(WorktreeWorkerRunner, "run", fake_run)
    manager = WorktreeWorkerManager(
        parent_workspace=repository,
        run_id="run_test",
        model_client=_DummyModel(),
        registry=registry,
        artifact_dir=tmp_path / "artifacts",
        worktree_manager=worktrees,
    )

    started = manager.start("inspect app")
    task = _wait(registry, started["runtime_task_id"])

    assert task["status"] == "completed"
    assert task["metadata"]["changed_files"] == []
    assert not Path(started["worktree"]).exists()
    manager.shutdown()


def test_worker_limit_rejects_a_third_concurrent_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path / "repo")
    registry = RuntimeTaskRegistry(max_active=4)
    worktrees = GitWorktreeManager(tmp_path / "data")
    release = Event()

    def fake_run(self, task: str) -> WorktreeWorkerResult:
        release.wait(timeout=2)
        return WorktreeWorkerResult(
            status="completed",
            summary="Done.",
            steps=1,
            tool_calls=0,
        )

    monkeypatch.setattr(WorktreeWorkerRunner, "run", fake_run)
    manager = WorktreeWorkerManager(
        parent_workspace=repository,
        run_id="run_limit",
        model_client=_DummyModel(),
        registry=registry,
        artifact_dir=tmp_path / "artifacts",
        worktree_manager=worktrees,
        max_workers=2,
    )

    first = manager.start("first")
    second = manager.start("second")
    with pytest.raises(RuntimeError, match=r"limit reached \(2\)"):
        manager.start("third")

    release.set()
    _wait(registry, first["runtime_task_id"])
    _wait(registry, second["runtime_task_id"])
    manager.shutdown()


def test_changed_worker_is_retained_with_diff_artifact(tmp_path: Path, monkeypatch) -> None:
    repository = _repository(tmp_path / "repo")
    registry = RuntimeTaskRegistry()
    worktrees = GitWorktreeManager(tmp_path / "data")

    def fake_run(self, task: str) -> WorktreeWorkerResult:
        workspace = Path(self.workspace)
        (workspace / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
        (workspace / "new.py").write_text("NEW = True\n", encoding="utf-8")
        return WorktreeWorkerResult(
            status="completed",
            summary="Changed app and added new file.",
            steps=2,
            tool_calls=2,
            verification=[{"argv": ["python", "-m", "pytest"], "returncode": 0}],
        )

    monkeypatch.setattr(WorktreeWorkerRunner, "run", fake_run)
    manager = WorktreeWorkerManager(
        parent_workspace=repository,
        run_id="run_test",
        model_client=_DummyModel(),
        registry=registry,
        artifact_dir=tmp_path / "artifacts",
        worktree_manager=worktrees,
    )

    started = manager.start("change app")
    task = _wait(registry, started["runtime_task_id"])

    assert task["status"] == "completed"
    assert set(task["metadata"]["changed_files"]) == {"app.py", "new.py"}
    assert Path(started["worktree"]).is_dir()
    artifact = tmp_path / "artifacts" / task["artifact_path"]
    assert artifact.is_file()
    patch = artifact.read_text(encoding="utf-8")
    assert "VALUE = 3" in patch
    assert "NEW = True" in patch
    assert (repository / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    manager.shutdown()


def test_worker_runner_edits_only_its_workspace_with_bounded_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "worker"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    client = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="edit",
                        name="edit",
                        arguments={
                            "path": "app.py",
                            "old_text": "VALUE = 1\n",
                            "new_text": "VALUE = 2\n",
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Updated app.py."),
        ]
    )
    runner = WorktreeWorkerRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "trace.jsonl"),
        artifact_dir=tmp_path / "artifacts",
        cancellation_token=CancellationToken(),
    )

    result = runner.run("Update app.py")

    assert result.status == "completed"
    assert (workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    names = _tool_names(client.calls[0][1])
    assert {"read", "search", "edit", "write", "apply_patch", "run_command"} <= names
    assert "delegate_task" not in names
    assert "delegate_worktree" not in names
    assert "runtime_task_status" not in names


def test_worker_runner_rejects_background_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "worker"
    workspace.mkdir()
    client = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="background",
                        name="run_command",
                        arguments={
                            "argv": ["python", "--version"],
                            "background": True,
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Stopped after the rejected command."),
        ]
    )
    runner = WorktreeWorkerRunner(
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "trace.jsonl"),
        artifact_dir=tmp_path / "artifacts",
        cancellation_token=CancellationToken(),
    )

    result = runner.run("Try a background command")

    assert result.status == "completed"
    assert result.verification == []
    second_messages = client.calls[1][0]
    assert "cannot start background commands" in second_messages[-1]["content"]
