import json

import pytest
from pydantic import ValidationError

from minicode_harness.context import (
    ContextObservation,
    InspectedFile,
    RunState,
    VerificationState,
)
from minicode_harness.state import (
    CheckpointStore,
    RunCheckpoint,
    TaskListState,
    TaskRecord,
    detect_workspace_conflicts,
    digest_workspace_files,
)


def _checkpoint(tmp_path) -> RunCheckpoint:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    readme = workspace / "README.md"
    readme.write_text("hello\n", encoding="utf-8")
    return RunCheckpoint(
        run_id="run_20260711_001",
        step=2,
        task="Explain README",
        workspace=str(workspace),
        run_state=RunState(
            inspected_files=[
                InspectedFile(
                    path="README.md",
                    summary="Read README.",
                    last_tool_call_id="call_1",
                    last_step=1,
                    line_start=1,
                    line_end=1,
                    total_lines=1,
                    content_status="full_content_available_in_context",
                )
            ],
            verification=VerificationState(
                status="passed",
                command="python -m pytest tests/test_checkpoint.py -q",
                returncode=0,
            ),
        ),
        task_state=TaskListState(
            next_id=2,
            tasks=[
                TaskRecord(
                    id="1",
                    subject="Explain README",
                    status="in_progress",
                )
            ],
        ),
        recent_observations=[
            ContextObservation(
                tool_call_id="call_1",
                tool_name="read",
                content="hello",
                output_preview="hello",
                token_estimate=1,
                summary="Read README.",
            )
        ],
        user_turn_id="turn_1",
        model_call_count=2,
        workspace_digest=digest_workspace_files(workspace, ["README.md"]),
        tool_calls=1,
    )


def test_checkpoint_store_saves_current_schema(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    checkpoint = _checkpoint(tmp_path)

    history = [{"role": "user", "content": "Explain README"}]
    path = store.save(checkpoint, message_history=history)
    loaded = store.load_latest()

    assert path == store.latest_path
    assert path.is_file()
    assert [item.name for item in store.checkpoints_dir.iterdir()] == ["latest.json"]
    assert loaded is not None
    assert loaded.run_state.inspected_files[0].path == "README.md"
    assert loaded.run_state.verification.status == "passed"
    assert loaded.run_state.verification.returncode == 0
    assert loaded.task_state.tasks[0].id == "1"
    assert loaded.task_state.tasks[0].status == "in_progress"
    assert loaded.user_turn_id == "turn_1"
    assert loaded.model_call_count == 2
    assert loaded.history_length == 1
    assert store.load_history(loaded) == history
    assert "llm_history_summary_calls" not in loaded.model_dump()


def test_checkpoint_json_has_only_minimal_run_state(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    store.save(
        _checkpoint(tmp_path),
        message_history=[{"role": "user", "content": "Explain README"}],
    )
    payload = json.loads(store.latest_path.read_text(encoding="utf-8"))

    assert "run_state" in payload
    assert set(payload["run_state"]) == {"inspected_files", "verification"}
    assert "working_context" not in payload
    assert "progress_summary" not in payload["run_state"]
    assert "blockers" not in payload["run_state"]
    assert "task_memory" not in payload
    assert payload["task_state"] == {
        "next_id": 2,
        "tasks": [
            {
                "id": "1",
                "subject": "Explain README",
                "status": "in_progress",
            }
        ],
    }
    assert "observations" not in payload
    assert "current_plan" not in payload
    assert "message_history" not in payload
    assert payload["history_length"] == 1


@pytest.mark.parametrize(
    "legacy_field, legacy_value",
    [
        ("working_context", {"changed_files": ["src/a.py"]}),
        ("full_message_history", [{"role": "user", "content": "old"}]),
        ("current_plan", "old plan"),
        ("task_memory", {"blockers": ["old"]}),
    ],
)
def test_checkpoint_rejects_retired_fields(
    tmp_path,
    legacy_field: str,
    legacy_value: object,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = {
        "run_id": "run_20260711_001",
        "step": 1,
        "task": "current schema only",
        "workspace": str(workspace),
        legacy_field: legacy_value,
    }

    with pytest.raises(ValidationError):
        RunCheckpoint.model_validate(payload)


def test_checkpoint_preserves_current_tool_calls_without_rewriting(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    message_history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "search_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps(
                            {
                                "source": "workspace",
                                "kind": "files",
                                "query": "**/*.py",
                                "path": "tests",
                            }
                        ),
                    },
                }
            ],
        }
    ]
    store = CheckpointStore(tmp_path / "checkpoints")
    checkpoint = RunCheckpoint(
        run_id="run_20260711_001",
        step=1,
        task="current tools",
        workspace=str(workspace),
    )
    store.save(checkpoint, message_history=message_history)
    loaded = store.load_latest()

    assert loaded is not None
    assert store.load_history(loaded) == message_history


def test_workspace_conflict_detection_uses_current_digest(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    workspace = tmp_path / "workspace"
    assert detect_workspace_conflicts(workspace, checkpoint) == []

    (workspace / "README.md").write_text("changed\n", encoding="utf-8")
    conflicts = detect_workspace_conflicts(workspace, checkpoint)
    assert [conflict.path for conflict in conflicts] == ["README.md"]


def test_checkpoint_externalizes_canonical_history(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "run" / "checkpoints")
    checkpoint = RunCheckpoint(
        run_id="run_20260711_001",
        step=1,
        task="resume later",
        workspace=str(tmp_path),
    )
    history = [
        {"role": "user", "content": "inspect README"},
        {"role": "assistant", "content": "done"},
    ]

    store.save(checkpoint, message_history=history)
    payload = json.loads(store.latest_path.read_text(encoding="utf-8"))
    loaded = store.load_latest()

    assert "message_history" not in payload
    assert payload["history_length"] == 2
    assert len(payload["history_sha256"]) == 64
    assert store.history_path.is_file()
    assert loaded is not None
    assert store.load_history(loaded) == history


def test_checkpoint_history_uses_checkpoint_prefix_after_newer_history_write(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "run" / "checkpoints")
    checkpoint = RunCheckpoint(
        run_id="run_20260711_001",
        step=1,
        task="resume later",
        workspace=str(tmp_path),
    )
    original = [{"role": "user", "content": "first"}]
    store.save(checkpoint, message_history=original)
    checkpoint_payload = store.latest_path.read_text(encoding="utf-8")

    store.save(
        checkpoint.model_copy(update={"step": 2}),
        message_history=[*original, {"role": "assistant", "content": "second"}],
    )
    store.latest_path.write_text(checkpoint_payload, encoding="utf-8")

    loaded = store.load_latest()
    assert loaded is not None
    assert store.load_history(loaded) == original

    store.history_path.write_text(
        json.dumps([{"role": "user", "content": "mutated"}], ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="history hash mismatch"):
        store.load_history(loaded)
