from __future__ import annotations

import json
from pathlib import Path
from threading import Event

from minicode_harness.memory.extraction import (
    Phase1Extractor,
    filter_rollout_messages,
    run_pending_phase1,
)
from minicode_harness.memory.store import RepositoryMemoryStore
from minicode_harness.models import ModelResponse
from minicode_harness.state import CheckpointStore, RunCheckpoint, RunStore


class FakeModelClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests = []

    def call_request(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(final_text=json.dumps(response, ensure_ascii=False))


def _terminal_run(
    run_store: RunStore,
    workspace: Path,
    run_id: str,
    *,
    status: str = "completed",
    memory_enabled: bool = True,
    messages: list[dict] | None = None,
) -> None:
    session = run_store.create_run(
        task=f"task {run_id}",
        workspace=workspace,
        run_id=run_id,
        repository_memory_enabled=memory_enabled,
    )
    run_path = run_store.path_for(session.run_id)
    history = messages or [
        {"role": "user", "content": "Use focused tests.", "_minicode_trace": "drop"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"argv":["pytest","tests/test_memory.py","-q"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"returncode":0,"stdout":"1 passed"}',
        },
        {"role": "assistant", "content": "Focused verification passed."},
    ]
    CheckpointStore(run_path / "checkpoints").save(
        RunCheckpoint(
            run_id=run_id,
            step=1,
            task=session.task,
            workspace=session.workspace,
            status=status,
        ),
        message_history=history,
    )
    run_store.update_session_state(run_id, status=status, current_step=1)


def _setup(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    memory = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    return workspace, run_store, memory


def test_rollout_filter_removes_only_internal_message_metadata() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "visible",
            "_minicode_trace": {"presentation": True},
            "tool_calls": [{"id": "call_1"}],
        }
    ]

    filtered = filter_rollout_messages(messages)

    assert filtered == [
        {
            "role": "assistant",
            "content": "visible",
            "tool_calls": [{"id": "call_1"}],
        }
    ]


def test_phase1_valid_output_writes_memory_record_with_tool_free_request(tmp_path: Path) -> None:
    workspace, run_store, memory = _setup(tmp_path)
    _terminal_run(run_store, workspace, "run_20260918_001")
    client = FakeModelClient(
        [
            {
                "raw_memory": "The user prefers focused tests.",
                "rollout_summary": "Focused verification completed successfully.",
                "rollout_slug": "focused-tests",
            }
        ]
    )

    result = Phase1Extractor(memory, run_store, client).extract("run_20260918_001")

    assert result.status == "memory"
    record = memory.load_stage1("run_20260918_001")
    assert record is not None
    assert record.seq == 1
    assert record.raw_memory == "The user prefers focused tests."
    request = client.requests[0]
    assert request.tools == []
    payload = json.loads(request.messages[0]["content"])
    assert payload["run_id"] == "run_20260918_001"
    assert payload["repository_id"] == memory.repository_id
    assert payload["rollout"][0] == {"role": "user", "content": "Use focused tests."}
    assert payload["rollout"][2]["content"].startswith('{"returncode":0')


def test_phase1_empty_output_is_terminal_without_sequence(tmp_path: Path) -> None:
    workspace, run_store, memory = _setup(tmp_path)
    _terminal_run(run_store, workspace, "run_20260918_001")
    client = FakeModelClient(
        [{"raw_memory": "", "rollout_summary": "", "rollout_slug": ""}]
    )

    result = Phase1Extractor(memory, run_store, client).extract("run_20260918_001")

    assert result.status == "no_output"
    assert memory.load_stage1("run_20260918_001").seq is None
    assert memory.load_state().latest_stage1_seq == 0


def test_phase1_failure_leaves_run_retryable(tmp_path: Path) -> None:
    workspace, run_store, memory = _setup(tmp_path)
    _terminal_run(run_store, workspace, "run_20260918_001")
    client = FakeModelClient([RuntimeError("provider unavailable")])

    try:
        Phase1Extractor(memory, run_store, client).extract("run_20260918_001")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected extraction failure")

    assert memory.load_stage1("run_20260918_001") is None
    assert memory.load_state().latest_stage1_seq == 0


def test_pending_phase1_scans_only_earlier_terminal_top_level_runs(tmp_path: Path) -> None:
    workspace, run_store, memory = _setup(tmp_path)
    _terminal_run(run_store, workspace, "run_20260918_001", status="completed")
    _terminal_run(run_store, workspace, "run_20260918_002", status="stopped")
    _terminal_run(run_store, workspace, "run_20260918_003", status="cancelled")
    _terminal_run(run_store, workspace, "run_20260918_004", status="failed")
    _terminal_run(
        run_store,
        workspace,
        "run_20260918_005",
        status="completed",
        memory_enabled=False,
    )
    current = run_store.create_run(
        task="current",
        workspace=workspace,
        run_id="run_20260918_006",
        repository_memory_enabled=True,
    )
    memory.write_stage1_no_output("run_20260918_002")
    client = FakeModelClient(
        [
            {"raw_memory": "", "rollout_summary": "", "rollout_slug": ""},
            {"raw_memory": "", "rollout_summary": "", "rollout_slug": ""},
        ]
    )

    result = run_pending_phase1(
        store=memory,
        run_store=run_store,
        model_client=client,
        current_run_id=current.run_id,
    )

    assert result.processed_run_ids == [
        "run_20260918_001",
        "run_20260918_003",
    ]
    assert len(client.requests) == 2
    assert memory.load_stage1("run_20260918_004") is None
    assert memory.load_stage1("run_20260918_005") is None
    assert memory.load_stage1("run_20260918_006") is None


def test_pending_phase1_continues_after_one_rollout_fails(tmp_path: Path) -> None:
    workspace, run_store, memory = _setup(tmp_path)
    _terminal_run(run_store, workspace, "run_20260918_001")
    _terminal_run(run_store, workspace, "run_20260918_002")
    current = run_store.create_run(
        task="current",
        workspace=workspace,
        run_id="run_20260918_003",
    )
    client = FakeModelClient(
        [
            RuntimeError("temporary failure"),
            {"raw_memory": "", "rollout_summary": "", "rollout_slug": ""},
        ]
    )

    result = run_pending_phase1(
        store=memory,
        run_store=run_store,
        model_client=client,
        current_run_id=current.run_id,
    )

    assert result.failed_run_ids == ["run_20260918_001"]
    assert result.processed_run_ids == ["run_20260918_002"]
    assert memory.load_stage1("run_20260918_001") is None
    assert memory.load_stage1("run_20260918_002").status == "no_output"
