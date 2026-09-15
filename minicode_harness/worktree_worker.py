"""Bounded write-capable workers isolated by managed Git worktrees."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import json
import subprocess
import time
from typing import Any

from pydantic import BaseModel, Field

from minicode_harness.context import (
    ContextPreparer,
    LLMSemanticHistoryCompactor,
    build_observation,
    render_tool_result_message,
)
from minicode_harness.models import ModelClient
from minicode_harness.policy import RiskLevel
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.recovery import ModelRecoveryPolicy
from minicode_harness.runtime.runtime_tasks import (
    RuntimeTaskRecord,
    RuntimeTaskRegistry,
)
from minicode_harness.subagent import build_subagent_token_budget
from minicode_harness.tools import CommandExecutor, ToolRegistry
from minicode_harness.trace import TraceWriter
from minicode_harness.worktrees import GitWorktreeManager, WorktreeManifest


class WorktreeWorkerResult(BaseModel):
    """Bounded worker outcome before deterministic Git closeout."""

    status: str
    summary: str
    steps: int
    tool_calls: int
    verification: list[dict[str, Any]] = Field(default_factory=list)


class WorktreeWorkerRunner:
    """Execute one non-recursive code task inside an isolated worktree."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        model_client: ModelClient,
        trace_writer: TraceWriter,
        artifact_dir: Path | str,
        cancellation_token: CancellationToken,
        command_executor: CommandExecutor | None = None,
        recovery_policy: ModelRecoveryPolicy | None = None,
        max_steps: int = 12,
        max_tool_calls: int = 24,
        max_elapsed_seconds: int = 900,
    ) -> None:
        self.workspace = str(Path(workspace).resolve())
        self.model_client = model_client
        self.trace_writer = trace_writer
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.cancellation_token = cancellation_token
        self.recovery_policy = recovery_policy or ModelRecoveryPolicy()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_elapsed_seconds = max_elapsed_seconds
        self.tools = ToolRegistry(
            self.workspace,
            enable_write=True,
            enable_command=True,
            artifact_dir=str(self.artifact_dir),
            skill_loader=None,
            skill_names=[],
            mcp_manager=None,
            subagent_handler=None,
            cancellation_token=self.cancellation_token,
            command_executor=command_executor,
        )
        self.preparer = ContextPreparer(
            build_subagent_token_budget(self.model_client),
            semantic_compactor=LLMSemanticHistoryCompactor(
                self.model_client,
                trace_writer=self.trace_writer,
            ),
        )

    def run(self, task: str) -> WorktreeWorkerResult:
        clean_task = task.strip()
        if not clean_task:
            raise ValueError("worktree worker task must not be empty.")
        messages: list[dict[str, Any]] = [{"role": "user", "content": clean_task}]
        tool_calls = 0
        verification: list[dict[str, Any]] = []
        started = time.monotonic()
        self.trace_writer.write_event("worktree_worker_started", task=clean_task)

        for step in range(1, self.max_steps + 1):
            if self.cancellation_token.is_cancelled:
                return WorktreeWorkerResult(
                    status="stopped",
                    summary="Worker cancellation requested.",
                    steps=max(0, step - 1),
                    tool_calls=tool_calls,
                    verification=verification,
                )
            if time.monotonic() - started >= self.max_elapsed_seconds:
                return WorktreeWorkerResult(
                    status="limit_reached",
                    summary="Worker elapsed-time limit reached.",
                    steps=max(0, step - 1),
                    tool_calls=tool_calls,
                    verification=verification,
                )
            excluded_control_tools = {
                "delegate_task",
                "delegate_worktree",
                "runtime_task_status",
                "runtime_task_stop",
            }
            schemas = [
                schema
                for schema in self.tools.schemas()
                if str(schema["function"]["name"]) not in excluded_control_tools
            ]
            prepared = self.preparer.prepare(
                system_messages=[
                    {
                        "role": "system",
                        "content": _WORKTREE_WORKER_SYSTEM_PROMPT
                        + f"\n\n工作目录：{self.workspace}",
                    }
                ],
                messages=messages,
                tools=schemas,
                tool_effects=self.tools.history_effects(),
                metadata={"worktree_worker": True, "step": step},
            )
            response = self.recovery_policy.invoke(
                lambda: self.model_client.call_request(prepared.request)
            ).enforce_turn_contract()
            if response.is_final():
                summary = response.final_text or ""
                return WorktreeWorkerResult(
                    status="completed",
                    summary=summary,
                    steps=step,
                    tool_calls=tool_calls,
                    verification=verification,
                )
            if not response.tool_calls:
                break
            if tool_calls + len(response.tool_calls) > self.max_tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                tool_calls += 1
                observation = None
                try:
                    admission = self.tools.admit(call.name, call.arguments)
                    if not admission.allowed:
                        raise PermissionError(
                            admission.command_policy.reason
                            if admission.command_policy is not None
                            else "Tool denied."
                        )
                    if call.name == "run_command":
                        if bool(admission.arguments.get("background", False)):
                            raise PermissionError(
                                "Worktree Workers cannot start background commands."
                            )
                        if admission.requires_approval:
                            raise PermissionError(
                                "Worktree Workers may run only commands allowed without approval."
                            )
                    if call.name != "run_command" and admission.risk_level == RiskLevel.HIGH:
                        raise PermissionError("High-risk Worker tools are denied.")
                    result = self.tools.execute_admitted(admission)
                    observation, compression_events = build_observation(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        result=result,
                        artifact_dir=self.artifact_dir,
                        tool_arguments=call.arguments,
                    )
                    for event in compression_events:
                        self.trace_writer.write_event(
                            "worktree_worker_context_compressed",
                            step=step,
                            **event.model_dump(mode="json"),
                        )
                    content = render_tool_result_message(observation)
                    status = str(observation.metadata.get("status") or "ok")
                    if call.name == "run_command":
                        verification.append(
                            {
                                "argv": list(admission.arguments["argv"]),
                                "returncode": observation.metadata.get("returncode"),
                                "status": status,
                            }
                        )
                except Exception as exc:
                    content = f"Tool {call.name} failed: {type(exc).__name__}: {exc}"
                    status = "error"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": content,
                    }
                )
                self.trace_writer.write_event(
                    "worktree_worker_tool_result",
                    step=step,
                    tool=call.name,
                    status=status,
                    truncated=(observation.is_truncated if observation is not None else False),
                    artifact_path=(observation.artifact_path if observation is not None else None),
                )

        return WorktreeWorkerResult(
            status="limit_reached",
            summary="Worker stopped before producing a final answer.",
            steps=self.max_steps,
            tool_calls=tool_calls,
            verification=verification,
        )


class WorktreeWorkerManager:
    """Create isolated worktrees and execute at most two bounded workers."""

    def __init__(
        self,
        *,
        parent_workspace: Path | str,
        run_id: str,
        model_client: ModelClient,
        registry: RuntimeTaskRegistry,
        artifact_dir: Path | str,
        command_executor: CommandExecutor | None = None,
        recovery_policy: ModelRecoveryPolicy | None = None,
        worktree_manager: GitWorktreeManager | None = None,
        max_workers: int = 2,
    ) -> None:
        self.parent_workspace = Path(parent_workspace).resolve()
        self.run_id = run_id
        self.model_client = model_client
        self.registry = registry
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.command_executor = command_executor
        self.recovery_policy = recovery_policy or ModelRecoveryPolicy()
        self.worktree_manager = worktree_manager or GitWorktreeManager()
        self.max_workers = max_workers
        self._lock = Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="minicode-worktree-worker",
        )
        self._futures: dict[str, Future[None]] = {}

    def start(self, task: str) -> dict[str, Any]:
        clean_task = task.strip()
        if not clean_task:
            raise ValueError("worktree worker task must not be empty.")
        if self._parent_changes():
            raise RuntimeError(
                "Parent workspace is dirty; finish, commit, or revert current changes before "
                "starting a Worktree Worker."
            )
        with self._lock:
            if len(self._futures) >= self.max_workers:
                raise RuntimeError(
                    f"Worktree Worker limit reached ({self.max_workers})."
                )
        task_id = self.registry.allocate_id("worktree_worker")
        suffix = self.run_id.replace("_", "-")[-24:]
        worktree_name = f"{suffix}-{task_id}"[:64]
        manifest = self.worktree_manager.create_worker(
            self.parent_workspace,
            worktree_name,
            base_ref="HEAD",
        )
        cancellation = CancellationToken()
        record = RuntimeTaskRecord(
            id=task_id,
            kind="worktree_worker",
            summary=clean_task,
            metadata={
                "task": clean_task,
                "worktree": manifest.workspace_root,
                "branch": manifest.branch,
            },
        )
        self.registry.register(record, stop_callback=cancellation.cancel)
        future = self._pool.submit(
            self._run_worker,
            task_id,
            clean_task,
            manifest,
            cancellation,
        )
        with self._lock:
            self._futures[task_id] = future
        future.add_done_callback(self._worker_done_callback(task_id))
        return {
            "runtime_task_id": task_id,
            "status": "running",
            "worktree": manifest.workspace_root,
            "branch": manifest.branch,
        }

    def _worker_done_callback(self, task_id: str):
        def done(_future: Future[None]) -> None:
            with self._lock:
                self._futures.pop(task_id, None)

        return done

    def _run_worker(
        self,
        task_id: str,
        task: str,
        manifest: WorktreeManifest,
        cancellation: CancellationToken,
    ) -> None:
        worker_dir = self.artifact_dir / "worktree-workers" / task_id
        worker_trace = TraceWriter(worker_dir / "trace.jsonl")
        try:
            result = WorktreeWorkerRunner(
                workspace=manifest.workspace_root,
                model_client=self.model_client,
                trace_writer=worker_trace,
                artifact_dir=worker_dir / "artifacts",
                cancellation_token=cancellation,
                command_executor=self.command_executor,
                recovery_policy=self.recovery_policy,
            ).run(task)
            status = self.worktree_manager.inspect(manifest)
            diff_path = worker_dir / "diff.patch"
            patch = self._capture_patch(Path(manifest.workspace_root), status.changed_files)
            if patch:
                diff_path.parent.mkdir(parents=True, exist_ok=True)
                diff_path.write_text(patch, encoding="utf-8")
            if not status.dirty:
                self.worktree_manager.remove(
                    self.parent_workspace,
                    manifest.name,
                    force=True,
                    kind="worker",
                )
            runtime_status = (
                "stopped"
                if result.status == "stopped"
                else "completed"
                if result.status == "completed"
                else "failed"
            )
            diff_reference = (
                diff_path.relative_to(self.artifact_dir).as_posix()
                if patch
                else None
            )
            self.registry.complete(
                task_id,
                status=runtime_status,
                summary=result.summary,
                artifact_path=diff_reference,
                metadata={
                    "changed_files": status.changed_files,
                    "worktree": manifest.workspace_root,
                    "branch": manifest.branch,
                    "verification": result.verification,
                    "steps": result.steps,
                    "tool_calls": result.tool_calls,
                },
            )
        except Exception as exc:
            self.registry.complete(
                task_id,
                status="failed",
                summary=f"{type(exc).__name__}: {exc}",
                metadata={
                    "worktree": manifest.workspace_root,
                    "branch": manifest.branch,
                },
            )

    def shutdown(self) -> None:
        with self._lock:
            task_ids = list(self._futures)
        for task_id in task_ids:
            self.registry.stop(task_id)
        self._pool.shutdown(wait=False, cancel_futures=False)

    def _parent_changes(self) -> list[str]:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.parent_workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Failed to inspect parent workspace.")
        return [line for line in result.stdout.splitlines() if line]

    @staticmethod
    def _capture_patch(workspace: Path, changed_files: list[str]) -> str:
        untracked: list[str] = []
        for path in changed_files:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", path],
                cwd=workspace,
                capture_output=True,
                check=False,
            ).returncode == 0
            if not tracked:
                untracked.append(path)
        if untracked:
            subprocess.run(
                ["git", "add", "-N", "--", *untracked],
                cwd=workspace,
                capture_output=True,
                check=False,
            )
        result = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if untracked:
            subprocess.run(
                ["git", "reset", "--", *untracked],
                cwd=workspace,
                capture_output=True,
                check=False,
            )
        return result.stdout


_WORKTREE_WORKER_SYSTEM_PROMPT = """你是 MiniCode 的有界 Worktree Worker。

规则：
- 只完成父 Agent 委派的一个独立代码任务；使用当前隔离 Worktree，不访问父工作区。
- 可以读取、精确修改文件，并运行无需审批的最窄聚焦验证。
- 不创建子 Agent，不使用 MCP，不启动后台命令，不提交、合并、变基、推送或部署。
- 修改前读取精确目标；完成后返回修改文件、验证命令、结果和不确定项的紧凑摘要。
- 达到任务边界后立即结束，不等待新任务。"""
