"""Bounded task state for one resumable Agent Run."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


TaskStatus = Literal["pending", "in_progress", "completed", "cancelled"]
MAX_TASKS = 12
_TASK_ID_PATTERN = re.compile(r"^\d+$")


class TaskRecord(BaseModel):
    """One lightweight task tracked by the Agent."""

    id: str = Field(..., pattern=r"^\d+$")
    subject: str = Field(..., min_length=1, max_length=120)
    status: TaskStatus = "pending"


class TaskListState(BaseModel):
    """Checkpoint-safe state for one Run task list."""

    next_id: int = Field(1, ge=1)
    tasks: list[TaskRecord] = Field(default_factory=list, max_length=MAX_TASKS)

    @model_validator(mode="after")
    def validate_state(self) -> "TaskListState":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Task IDs must be unique.")
        active_ids = [task.id for task in self.tasks if task.status == "in_progress"]
        if len(active_ids) > 1:
            raise ValueError("At most one task may be in_progress.")
        highest_id = max((_task_number(task_id) for task_id in ids), default=0)
        if self.next_id <= highest_id:
            raise ValueError("next_id must be greater than every existing task ID.")
        return self


class TaskStore:
    """In-memory task store with bounded, atomic mutations."""

    def __init__(self, state: TaskListState | None = None) -> None:
        self._state = (state or TaskListState()).model_copy(deep=True)

    def snapshot(self) -> TaskListState:
        """Return an isolated checkpoint snapshot."""

        return self._state.model_copy(deep=True)

    def create(self, subjects: list[str]) -> dict[str, object]:
        """Create missing tasks and return compact ID/title mappings."""

        normalized = [_normalize_subject(subject) for subject in subjects]
        existing_by_subject = {
            task.subject: task
            for task in self._state.tasks
            if task.status != "cancelled"
        }
        unique_subjects = list(dict.fromkeys(normalized))
        new_subjects = [subject for subject in unique_subjects if subject not in existing_by_subject]
        if len(self._state.tasks) + len(new_subjects) > MAX_TASKS:
            return {"error": "task_limit_reached", "limit": MAX_TASKS}

        created: list[list[str]] = []
        existing: list[list[str]] = []
        for subject in unique_subjects:
            existing_task = existing_by_subject.get(subject)
            if existing_task is not None:
                existing.append([existing_task.id, existing_task.subject])
                continue
            task = TaskRecord(
                id=_format_next_task_id(self._state),
                subject=subject,
            )
            self._state.next_id += 1
            self._state.tasks.append(task)
            existing_by_subject[subject] = task
            created.append([task.id, task.subject])

        result: dict[str, object] = {"created": created}
        if existing:
            result["existing"] = existing
        return result

    def update(self, updates: dict[str, TaskStatus]) -> dict[str, object]:
        """Atomically apply compact task status updates."""

        task_by_id = {task.id: task for task in self._state.tasks}
        for task_id in updates:
            if not _TASK_ID_PATTERN.fullmatch(task_id) or task_id not in task_by_id:
                return {"error": "task_not_found", "task_id": task_id}

        prospective = {task.id: task.status for task in self._state.tasks}
        prospective.update(updates)
        active_ids = [
            task_id
            for task_id, status in prospective.items()
            if status == "in_progress"
        ]
        if len(active_ids) > 1:
            return {"error": "multiple_in_progress", "task_ids": active_ids}

        changed: dict[str, TaskStatus] = {}
        for task_id, status in updates.items():
            task = task_by_id[task_id]
            if task.status == status:
                continue
            task.status = status
            changed[task_id] = status
        return {"updated": changed}

    def list_tasks(self) -> dict[str, object]:
        """Return compact ``[id, status, subject]`` rows."""

        return {
            "tasks": [
                [task.id, task.status, task.subject]
                for task in self._state.tasks
            ]
        }

    def open_rows(self) -> list[list[str]]:
        """Return pending and in-progress rows for compaction-time projection."""

        return [
            [task.id, task.status, task.subject]
            for task in self._state.tasks
            if task.status in {"pending", "in_progress"}
        ]


def _task_number(task_id: str) -> int:
    return int(task_id)


def _format_next_task_id(state: TaskListState) -> str:
    return str(state.next_id)


def _normalize_subject(subject: str) -> str:
    return " ".join(subject.split())
