import json
from typing import Any

from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.state import (
    ApprovalDecision,
    CheckpointStore,
    StaticApprovalClient,
)
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def call_request(self, request) -> ModelResponse:
        self.calls.append((request.as_chat_messages(), request.tools))
        return self.responses.pop(0)


def _find_call(call_id: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            NormalizedToolCall(
                id=call_id,
                name="search",
                arguments={
                    "source": "workspace",
                    "kind": "files",
                    "query": "**/*",
                    "path": ".",
                    "limit": 20,
                },
            )
        ]
    )


def _read_call(call_id: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            NormalizedToolCall(
                id=call_id,
                name="read",
                arguments={
                    "source": "workspace",
                    "target": "source.py",
                    "start_line": 1,
                    "end_line": 2,
                },
            )
        ]
    )


def _full_read_call(call_id: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            NormalizedToolCall(
                id=call_id,
                name="read",
                arguments={"source": "workspace", "target": "source.py"},
            )
        ]
    )


def _command_call(call_id: str, *argv: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            NormalizedToolCall(
                id=call_id,
                name="run_command",
                arguments={"argv": list(argv), "timeout_seconds": 30},
            )
        ]
    )


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    return workspace


def _nudge_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
        and "[Runtime guidance:" in str(message.get("content") or "")
    ]


def _system_nudge_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
        and "[Runtime guidance:" in str(message.get("content") or "")
    ]


def _nudge_events(trace_path) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    return [event for event in events if event.get("type") == "progress_stagnation_nudge"]


def test_eight_non_write_calls_inject_progress_nudge(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [*[_find_call(f"find_{index}") for index in range(8)], ModelResponse(final_text="done")]
    )
    loop = AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        config=AgentLoopConfig(enable_progress_guidance=True),
    )

    loop.run()

    request_messages = client.calls[8][0]
    messages = _nudge_messages(request_messages)
    assert len(messages) == 1
    assert "预期不变量" in messages[0]
    assert _system_nudge_messages(request_messages) == []
    guidance_index = next(
        index
        for index, message in enumerate(request_messages)
        if message.get("role") == "user"
        and "[Runtime guidance:" in str(message.get("content") or "")
    )
    assert guidance_index > max(
        index
        for index, message in enumerate(request_messages)
        if message.get("role") == "tool"
    )
    events = _nudge_events(trace_path)
    assert len(events) == 1
    assert "最小修改" in events[0]["guidance"]
    trace_events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event.get("type") == "checkpoint_saved"
        and event.get("reason") == "progress_guidance"
        for event in trace_events
    )
    checkpoint = CheckpointStore(trace_path.parent / "checkpoints").load_latest()
    assert checkpoint is not None
    assert len(_nudge_messages(checkpoint.message_history)) == 1


def test_stagnation_guidance_is_appended_once_to_canonical_history(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    responses = [*[_find_call(f"find_{index}") for index in range(8)]]
    responses.append(_find_call("after_nudge"))
    responses.append(
        ModelResponse(
            tool_calls=[
                NormalizedToolCall(
                    id="write",
                    name="write",
                    arguments={
                        "path": "source.py",
                        "content": "changed\n",
                        "overwrite": True,
                    },
                )
            ]
        )
    )
    responses.append(ModelResponse(final_text="done"))
    client = ScriptedModelClient(responses)

    AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
        config=AgentLoopConfig(enable_progress_guidance=True),
    ).run()

    assert len(_nudge_messages(client.calls[8][0])) == 1
    assert len(_nudge_messages(client.calls[9][0])) == 1
    assert len(_nudge_messages(client.calls[10][0])) == 1
    assert _system_nudge_messages(client.calls[10][0]) == []
    assert len(_nudge_events(trace_path)) == 1


def test_stagnation_nudge_is_disabled_by_default(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [*[_find_call(f"find_{index}") for index in range(8)], ModelResponse(final_text="done")]
    )

    AgentLoop(
        task="Explain source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
    ).run()

    assert _nudge_messages(client.calls[8][0]) == []
    assert _nudge_events(trace_path) == []


def test_at_most_two_general_stagnation_nudges_are_recorded(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [*[_find_call(f"find_{index}") for index in range(14)], ModelResponse(final_text="done")]
    )

    AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        config=AgentLoopConfig(enable_progress_guidance=True),
    ).run()

    events = _nudge_events(trace_path)
    general = [event for event in events if event.get("level") in {1, 2}]
    assert len(general) == 2
    assert "停止扩散探索" in general[1]["guidance"]
    assert "当前模型调用：12 / 50" in general[1]["guidance"]
    assert "最早破坏预期不变量" in general[1]["guidance"]


def test_duplicate_reused_reads_still_count_toward_stagnation(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [*[_read_call(f"read_{index}") for index in range(8)], ModelResponse(final_text="done")]
    )
    loop = AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        config=AgentLoopConfig(enable_progress_guidance=True),
    )

    loop.run()

    assert sum(
        item.metadata.get("status") == "duplicate_reused"
        for item in loop.observations
    ) == 7
    final_messages = client.calls[8][0]
    tool_messages = [message for message in final_messages if message.get("role") == "tool"]
    assert len(tool_messages) == 8
    first_read = json.loads(tool_messages[0]["content"])
    assert first_read["path"] == "source.py"
    assert first_read["content"] == "one\ntwo"
    assert len(_nudge_messages(final_messages)) == 1
    assert len(_nudge_events(trace_path)) == 1


def test_successful_commands_do_not_reset_non_write_progress_counter(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    responses = [
        *[
            _command_call(f"command_{index}", "python", "-c", f"print({index})")
            for index in range(8)
        ],
        ModelResponse(final_text="done"),
    ]
    client = ScriptedModelClient(responses)

    AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        config=AgentLoopConfig(enable_progress_guidance=True),
    ).run()

    assert len(_nudge_messages(client.calls[8][0])) == 1
    assert len(_nudge_events(trace_path)) == 1


def test_failed_verification_after_write_injects_reassessment(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "test_source.py").write_text("def test_failure():\n    assert False\n", encoding="utf-8")
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            _full_read_call("read_before_write"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "changed\n",
                            "overwrite": True,
                        },
                    )
                ]
            ),
            _command_call("verify", "python", "-m", "pytest", "-q", "test_source.py"),
            ModelResponse(final_text="done"),
        ]
    )

    AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
        config=AgentLoopConfig(enable_progress_guidance=True),
    ).run()

    messages = _nudge_messages(client.calls[3][0])
    assert len(messages) == 1
    assert "最早错误点" in messages[0]
    assert any(event.get("level") == 3 for event in _nudge_events(trace_path))


def test_failed_verification_guidance_remains_in_canonical_history(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "test_source.py").write_text(
        "def test_failure():\n    assert False\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [
            _full_read_call("read_before_write_1"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write_1",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "changed once\n",
                            "overwrite": True,
                        },
                    )
                ]
            ),
            _command_call("verify", "python", "-m", "pytest", "-q", "test_source.py"),
            _find_call("diagnose"),
            _full_read_call("read_before_write_2"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write_2",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "changed twice\n",
                            "overwrite": True,
                        },
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )

    AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
        config=AgentLoopConfig(enable_progress_guidance=True),
    ).run()

    assert len(_nudge_messages(client.calls[3][0])) == 1
    assert len(_nudge_messages(client.calls[4][0])) == 1
    assert len(_nudge_messages(client.calls[5][0])) == 1
    assert len(_nudge_messages(client.calls[6][0])) == 1
    assert _system_nudge_messages(client.calls[6][0]) == []
    assert any(event.get("level") == 3 for event in _nudge_events(trace_path))


def test_eighty_percent_budget_injects_execution_nudge(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    client = ScriptedModelClient(
        [*[_find_call(f"find_{index}") for index in range(5)], ModelResponse(final_text="done")]
    )

    AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        config=AgentLoopConfig(
            max_steps=6,
            enable_progress_guidance=True,
        ),
    ).run()

    messages = _nudge_messages(client.calls[5][0])
    assert len(messages) == 1
    assert "大部分模型调用预算" in messages[0]
    assert any(event.get("level") == 4 for event in _nudge_events(trace_path))


def test_write_resets_non_write_progress_counter(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    trace_path = tmp_path / "run" / "trace.jsonl"
    responses = [*[_find_call(f"before_{index}") for index in range(5)]]
    responses.append(
        ModelResponse(
            tool_calls=[
                NormalizedToolCall(
                    id="write",
                    name="write",
                    arguments={
                        "path": "source.py",
                        "content": "changed\n",
                        "overwrite": True,
                    },
                )
            ]
        )
    )
    responses.extend(_find_call(f"after_{index}") for index in range(5))
    responses.append(ModelResponse(final_text="done"))
    client = ScriptedModelClient(responses)
    loop = AgentLoop(
        task="Fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
        config=AgentLoopConfig(enable_progress_guidance=True),
    )

    loop.run()

    assert loop.progress_policy.non_write_calls_since_progress == 5
    assert _nudge_messages(client.calls[-1][0]) == []
    assert _nudge_events(trace_path) == []
