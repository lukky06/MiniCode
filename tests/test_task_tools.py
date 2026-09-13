from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import pytest
from pydantic import ValidationError

from minicode_harness.context import compact_execution_history
from minicode_harness.context.token import estimate_tokens
from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.state import CheckpointStore, TaskListState, TaskRecord, TaskStore
from minicode_harness.tools import ToolRegistry
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def call_request(self, request) -> ModelResponse:
        self.calls.append((request.as_chat_messages(), request.tools))
        return self.responses.pop(0)


def _registry(tmp_path: Path) -> tuple[ToolRegistry, TaskStore]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = TaskStore()
    registry = ToolRegistry(
        str(workspace),
        task_create_handler=store.create,
        task_update_handler=store.update,
        task_list_handler=store.list_tasks,
    )
    return registry, store


def _execute(registry: ToolRegistry, arguments: dict) -> dict:
    return registry.execute_admitted(registry.admit("task", arguments))


def test_task_create_uses_compact_batch_protocol_and_is_idempotent(tmp_path: Path) -> None:
    registry, store = _registry(tmp_path)

    first = _execute(
        registry,
        {"action": "create", "tasks": ["定位问题", "  修改代码并测试  "]},
    )
    repeated = _execute(
        registry,
        {"action": "create", "tasks": ["定位问题"]},
    )

    assert first == {
        "created": [
            ["1", "定位问题"],
            ["2", "修改代码并测试"],
        ]
    }
    assert "open" not in first
    assert repeated == {
        "created": [],
        "existing": [["1", "定位问题"]],
    }
    assert store.list_tasks() == {
        "tasks": [
            ["1", "pending", "定位问题"],
            ["2", "pending", "修改代码并测试"],
        ]
    }
    assert store.open_rows() == [
        ["1", "pending", "定位问题"],
        ["2", "pending", "修改代码并测试"],
    ]


def test_task_update_is_atomic_and_allows_one_active_task(tmp_path: Path) -> None:
    registry, store = _registry(tmp_path)
    _execute(
        registry,
        {"action": "create", "tasks": ["定位问题", "实现修复"]},
    )

    started = _execute(
        registry,
        {"action": "update", "updates": {"1": "in_progress"}},
    )
    switched = _execute(
        registry,
        {
            "action": "update",
            "updates": {"1": "completed", "2": "in_progress"},
        },
    )
    rejected = _execute(
        registry,
        {"action": "update", "updates": {"1": "in_progress"}},
    )

    assert started == {"updated": {"1": "in_progress"}}
    assert "open" not in started
    assert switched == {
        "updated": {
            "1": "completed",
            "2": "in_progress",
        }
    }
    assert rejected == {
        "error": "multiple_in_progress",
        "task_ids": ["1", "2"],
    }
    assert store.list_tasks() == {
        "tasks": [
            ["1", "completed", "定位问题"],
            ["2", "in_progress", "实现修复"],
        ]
    }
    assert store.open_rows() == [["2", "in_progress", "实现修复"]]


def test_task_create_enforces_bounded_task_list(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)

    created = _execute(
        registry,
        {
            "action": "create",
            "tasks": [f"任务 {index}" for index in range(12)],
        },
    )
    overflow = _execute(
        registry,
        {"action": "create", "tasks": ["额外任务"]},
    )

    assert len(created["created"]) == 12
    assert overflow == {"error": "task_limit_reached", "limit": 12}


def test_task_update_returns_compact_not_found_error(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)

    result = _execute(
        registry,
        {"action": "update", "updates": {"9": "completed"}},
    )

    assert result == {"error": "task_not_found", "task_id": "9"}


def test_task_tools_have_small_schemas_and_need_no_approval(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    schemas = {
        schema["function"]["name"]: schema["function"]
        for schema in registry.schemas()
    }

    assert "task" in schemas
    assert set(schemas["task"]["parameters"]["properties"]) == {
        "action",
        "tasks",
        "updates",
    }
    assert registry.requires_approval("task") is False
    assert registry.history_effects()["task"]["read_only"] is True
    assert registry.history_effects()["task"]["result_reconstructible"] is False
    task_schema_text = json.dumps(
        [schemas["task"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert estimate_tokens(task_schema_text) <= 320


def test_task_mutations_do_not_enter_compacted_execution(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    group = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "create_tasks",
                    "type": "function",
                    "function": {
                        "name": "task",
                        "arguments": json.dumps(
                            {"action": "create", "tasks": ["定位问题"]},
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "create_tasks",
            "content": json.dumps(
                {"created": [["1", "定位问题"]]},
                ensure_ascii=False,
            ),
        },
    ]

    assert compact_execution_history(
        [group],
        tool_effects=registry.history_effects(),
    ) == ""


def test_agent_loop_persists_tasks_without_consuming_workspace_tool_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="create_tasks",
                        name="task",
                        arguments={"action": "create", "tasks": ["定位问题", "实现修复"]},
                    ),
                    NormalizedToolCall(
                        id="update_tasks",
                        name="task",
                        arguments={
                            "action": "update",
                            "updates": {
                                "1": "completed",
                                "2": "in_progress",
                            }
                        },
                    ),
                ]
            ),
            ModelResponse(final_text="任务状态已记录。"),
        ]
    )

    loop = AgentLoop(
        task="规划并开始修复",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        config=AgentLoopConfig(max_steps=3, max_tool_calls=0),
    )
    original_execute = loop.tool_runtime.execute

    def delayed_execute(**kwargs):
        tool_call = kwargs["tool_call"]
        if tool_call.name == "task" and tool_call.arguments.get("action") == "create":
            time.sleep(0.05)
        return original_execute(**kwargs)

    monkeypatch.setattr(loop.tool_runtime, "execute", delayed_execute)
    result = loop.run()

    assert result.status == "completed"
    assert result.tool_calls == 0
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert checkpoint.task_state.model_dump(mode="json") == {
        "next_id": 3,
        "tasks": [
            {"id": "1", "subject": "定位问题", "status": "completed"},
            {"id": "2", "subject": "实现修复", "status": "in_progress"},
        ],
    }
    tool_names = {
        schema["function"]["name"]
        for schema in client.calls[0][1]
    }
    assert "task" in tool_names
    assert not {"task_create", "task_update", "task_list"}.intersection(tool_names)
    first_messages = client.calls[0][0]
    second_messages = client.calls[1][0]
    assert second_messages[: len(first_messages)] == first_messages
    assert not any(
        str(message.get("content") or "").startswith("[MiniCode current task]")
        for message in second_messages
    )


def test_task_state_rejects_multiple_active_tasks() -> None:
    with pytest.raises(ValueError, match="At most one task"):
        TaskListState(
            next_id=3,
            tasks=[
                TaskRecord(id="1", subject="定位问题", status="in_progress"),
                TaskRecord(id="2", subject="实现修复", status="in_progress"),
            ],
        )


def test_task_store_restores_checkpoint_state() -> None:
    store = TaskStore(
        TaskListState(
            next_id=3,
            tasks=[
                TaskRecord(id="1", subject="定位问题", status="completed"),
                TaskRecord(id="2", subject="实现修复", status="in_progress"),
            ],
        )
    )

    assert store.create(["补充测试"]) == {
        "created": [["3", "补充测试"]]
    }
    assert store.snapshot().next_id == 4


def test_task_state_rejects_prefixed_task_ids() -> None:
    with pytest.raises(ValueError):
        TaskRecord(id="t001", subject="旧任务")


def test_task_tool_rejects_prefixed_task_ids(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)

    with pytest.raises(ValidationError):
        registry.admit(
            "task",
            {"action": "update", "updates": {"t001": "completed"}},
        )
