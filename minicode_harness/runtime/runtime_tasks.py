"""Bounded runtime tasks for background commands and isolated workers."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field

from minicode_harness.runtime.cancellation import CancellationToken


class RuntimeCommandExecutor(Protocol):
    """Small execution protocol to keep runtime tasks independent from tools."""

    def execute(
        self,
        workspace: Path | str,
        argv: list[str],
        timeout_seconds: int,
        cancellation_token: CancellationToken | None = None,
        *,
        approval_granted: bool = False,
    ) -> Any:
        ...


RuntimeTaskKind = Literal["command", "worktree_worker"]
RuntimeTaskStatus = Literal[
    "running",
    "completed",
    "failed",
    "stopped",
    "interrupted",
]


class RuntimeTaskRecord(BaseModel):
    """Compact state for one asynchronous runtime task."""

    id: str
    kind: RuntimeTaskKind
    status: RuntimeTaskStatus = "running"
    summary: str = ""
    artifact_path: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeTaskRegistry:
    """Thread-safe bounded registry with one-shot completion notifications."""

    def __init__(self, *, max_active: int = 4) -> None:
        self.max_active = max_active
        self._records: dict[str, RuntimeTaskRecord] = {}
        self._stop_callbacks: dict[str, Callable[[], None]] = {}
        self._notifications: list[str] = []
        self._lock = Lock()
        self._next_ids: dict[str, int] = {"command": 0, "worktree_worker": 0}

    def allocate_id(self, kind: RuntimeTaskKind) -> str:
        with self._lock:
            active = sum(record.status == "running" for record in self._records.values())
            if active >= self.max_active:
                raise RuntimeError(f"Runtime task limit reached ({self.max_active}).")
            self._next_ids[kind] += 1
            prefix = "cmd" if kind == "command" else "worker"
            return f"{prefix}_{self._next_ids[kind]:04d}"

    def register(
        self,
        record: RuntimeTaskRecord,
        *,
        stop_callback: Callable[[], None] | None = None,
    ) -> RuntimeTaskRecord:
        with self._lock:
            if record.id in self._records:
                raise ValueError(f"Runtime task already exists: {record.id}")
            self._records[record.id] = record.model_copy(deep=True)
            if stop_callback is not None:
                self._stop_callbacks[record.id] = stop_callback
            return self._records[record.id].model_copy(deep=True)

    def complete(
        self,
        task_id: str,
        *,
        status: RuntimeTaskStatus,
        summary: str,
        artifact_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status != "running":
                return
            record.status = status
            record.summary = summary
            record.artifact_path = artifact_path or record.artifact_path
            record.finished_at = datetime.now(timezone.utc)
            if metadata:
                record.metadata.update(metadata)
            self._stop_callbacks.pop(task_id, None)
            self._notifications.append(self._render_notification(record))

    def status(self, task_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            records = (
                [self._records[task_id]]
                if task_id is not None and task_id in self._records
                else list(self._records.values())
                if task_id is None
                else []
            )
            if task_id is not None and not records:
                return {"status": "not_running", "task_id": task_id}
            return {
                "tasks": [record.model_dump(mode="json") for record in records]
            }

    def stop(self, task_id: str) -> dict[str, Any]:
        callback: Callable[[], None] | None
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return {"status": "not_running", "task_id": task_id}
            if record.status != "running":
                return {
                    "status": "not_running",
                    "task_id": task_id,
                    "previous_status": record.status,
                }
            callback = self._stop_callbacks.get(task_id)
        if callback is not None:
            callback()
        return {"task_id": task_id, "status": "stop_requested"}

    def drain_notifications(self) -> list[str]:
        with self._lock:
            notifications = list(self._notifications)
            self._notifications.clear()
            return notifications

    def shutdown(self) -> None:
        with self._lock:
            task_ids = [
                task_id
                for task_id, record in self._records.items()
                if record.status == "running"
            ]
        for task_id in task_ids:
            self.stop(task_id)

    @staticmethod
    def _render_notification(record: RuntimeTaskRecord) -> str:
        lines = [
            "[MiniCode runtime notification]",
            f"id: {record.id}",
            f"kind: {record.kind}",
            f"status: {record.status}",
        ]
        if record.summary:
            lines.append(f"summary: {record.summary}")
        if record.artifact_path:
            lines.append(f"artifact: {record.artifact_path}")
        for key in (
            "command_status",
            "returncode",
            "timed_out",
            "cancelled",
            "duration_seconds",
            "changed_files",
            "verification",
            "worktree",
            "branch",
        ):
            if key in record.metadata:
                lines.append(f"{key}: {record.metadata[key]}")
        return "\n".join(lines)


class BackgroundCommandManager:
    """Run policy-admitted commands asynchronously with bounded output artifacts."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        artifact_dir: Path | str,
        command_executor: RuntimeCommandExecutor,
        registry: RuntimeTaskRegistry,
        max_workers: int = 2,
    ) -> None:
        self.workspace = str(Path(workspace).resolve())
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.command_executor = command_executor
        self.registry = registry
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="minicode-background-command",
        )
        self._futures: dict[str, Future[Any]] = {}
        self._lock = Lock()

    def start(
        self,
        argv: list[str],
        timeout_seconds: int,
        *,
        approval_granted: bool = False,
    ) -> dict[str, Any]:
        task_id = self.registry.allocate_id("command")
        cancellation = CancellationToken()
        artifact_path = self.artifact_dir / "runtime-tasks" / f"{task_id}.log"
        artifact_reference = artifact_path.relative_to(self.artifact_dir).as_posix()
        record = RuntimeTaskRecord(
            id=task_id,
            kind="command",
            summary=" ".join(argv),
            artifact_path=artifact_reference,
            metadata={"argv": list(argv), "timeout_seconds": timeout_seconds},
        )
        self.registry.register(record, stop_callback=cancellation.cancel)
        future = self._pool.submit(
            self.command_executor.execute,
            self.workspace,
            list(argv),
            timeout_seconds,
            cancellation,
            approval_granted=approval_granted,
        )
        with self._lock:
            self._futures[task_id] = future
        future.add_done_callback(
            lambda completed, task_id=task_id, artifact_path=artifact_path, artifact_reference=artifact_reference: self._finish(
                task_id,
                artifact_path,
                artifact_reference,
                completed,
            )
        )
        return {
            "runtime_task_id": task_id,
            "status": "running",
            "argv": list(argv),
            "artifact_path": artifact_reference,
        }

    def _finish(
        self,
        task_id: str,
        artifact_path: Path,
        artifact_reference: str,
        future: Future[Any],
    ) -> None:
        try:
            result = future.result()
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                "\n".join(
                    [
                        f"$ {result.command}",
                        f"returncode: {result.returncode}",
                        "",
                        result.stdout,
                        result.stderr,
                    ]
                ),
                encoding="utf-8",
            )
            status: RuntimeTaskStatus
            if result.cancelled:
                status = "stopped"
            elif result.returncode == 0:
                status = "completed"
            else:
                status = "failed"
            preview = (result.stderr or result.stdout).strip().splitlines()
            summary = preview[-1][:300] if preview else f"returncode {result.returncode}"
            self.registry.complete(
                task_id,
                status=status,
                summary=summary,
                artifact_path=artifact_reference,
                metadata={
                    "command_status": result.lifecycle_status,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "cancelled": result.cancelled,
                    "duration_seconds": result.duration_seconds,
                },
            )
        except Exception as exc:
            artifact_path.write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            self.registry.complete(
                task_id,
                status="failed",
                summary=f"{type(exc).__name__}: {exc}",
                artifact_path=artifact_reference,
            )
        finally:
            with self._lock:
                self._futures.pop(task_id, None)

    def shutdown(self) -> None:
        with self._lock:
            task_ids = list(self._futures)
        for task_id in task_ids:
            self.registry.stop(task_id)
        self._pool.shutdown(wait=False, cancel_futures=False)
