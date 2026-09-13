import pytest

from minicode_harness.context import (
    ExecutionCompactionState,
    SessionCompactionState,
)
from minicode_harness.state import ReplSessionMemory, ReplSessionStore
from minicode_harness.state import UserTurnState


def test_repl_session_persists_canonical_message_history(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)
    messages = [
        {"role": "user", "content": "Explain README"},
        {"role": "assistant", "content": "README summary"},
    ]

    session.replace_message_history(messages)
    reloaded = store.load(workspace, session.session_id)

    assert reloaded.load_message_history() == messages


def test_user_turn_keeps_one_append_only_canonical_history() -> None:
    initial = [{"role": "user", "content": "Inspect service"}]
    turn = UserTurnState.create(
        turn_id="run_1",
        task="Inspect service",
        messages=initial,
        append_task=False,
    )

    turn.append_message({"role": "assistant", "content": "Final answer"})

    assert turn.snapshot_messages() == [
        *initial,
        {"role": "assistant", "content": "Final answer"},
    ]
    assert "full_messages" not in turn.model_dump()


def test_repl_session_persists_session_command_approval_grants(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)

    session.grant_command_approval("C:/Python/python.exe")
    session.grant_command_approval("C:/Python/python.exe")
    reloaded = store.load(workspace, session.session_id)

    assert reloaded.command_approval_grants == ["C:/Python/python.exe"]
    assert reloaded.has_command_approval_grant("C:/Python/python.exe") is True


def test_repl_session_persists_compaction_state_without_second_history(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)
    complete = [
        {"role": "user", "content": "Inspect service"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "full source"},
    ]
    state = SessionCompactionState(
        execution=ExecutionCompactionState(
            boundary_group_id="g00000001-deadbeefdeadbeef",
            source_digest="a" * 64,
        )
    )

    session.replace_session_state(messages=complete, compaction_state=state)
    reloaded = store.load(workspace, session.session_id)

    assert reloaded.load_message_history() == complete
    assert reloaded.load_compaction_state() == state
    dumped = reloaded.model_dump()
    assert "model_message_history" not in dumped
    assert "full_message_history" not in dumped


def test_repl_session_rejects_retired_parallel_history_field() -> None:
    with pytest.raises(ValueError):
        ReplSessionMemory.model_validate(
            {
                "session_id": "session_abcdef123456",
                "workspace": "D:/repo",
                "message_history": [],
                "model_message_history": [
                    {"role": "assistant", "content": "retired projection"}
                ],
            }
        )


def test_repl_session_does_not_silently_trim_more_than_120_messages(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)
    messages = [
        {"role": "user", "content": f"message {index}"}
        for index in range(130)
    ]

    session.replace_message_history(messages)
    reloaded = store.load(workspace, session.session_id)

    assert reloaded.load_message_history() == messages


def test_repl_session_store_uses_creation_date_directory(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = ReplSessionMemory(
        session_id="session_abcdef123456",
        workspace=str(workspace.resolve()),
        created_at="2026-07-03T23:45:00+00:00",
        updated_at="2026-07-03T23:45:00+00:00",
    )

    path = store.save(session)

    assert path == (
        tmp_path
        / "data"
        / "sessions"
        / "2026"
        / "07"
        / "03"
        / session.session_id
        / "session.json"
    )

    original_path = path
    session.updated_at = "2026-07-04T08:00:00+00:00"
    assert store.save(session) == original_path


def test_repl_session_store_creates_isolated_sessions_and_loads_latest(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")

    first = store.create(workspace)
    first.replace_message_history([{"role": "user", "content": "first"}])
    second = store.create(workspace)
    second.replace_message_history([{"role": "user", "content": "second"}])

    assert first.session_id != second.session_id
    assert store.load(workspace, first.session_id).message_history[0]["content"] == "first"
    assert store.load(workspace, second.session_id).message_history[0]["content"] == "second"
    assert store.load_latest(workspace).session_id == second.session_id
    assert [item.session_id for item in store.list(workspace)] == [
        second.session_id,
        first.session_id,
    ]


def test_repl_session_load_latest_requires_previous_session(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")

    with pytest.raises(FileNotFoundError, match="No previous session"):
        store.load_latest(workspace)


def test_repl_session_keeps_dialogue_for_history_display(tmp_path) -> None:
    session = ReplSessionMemory(workspace=str(tmp_path.resolve()))

    session.add_user_turn("Explain README", run_id="run_1")
    session.add_assistant_turn("README summary", run_id="run_1")

    assert [turn.role for turn in session.dialogue] == ["user", "assistant"]
    assert [turn.run_id for turn in session.dialogue] == ["run_1", "run_1"]
    assert all(turn.turn_id.startswith("turn_") for turn in session.dialogue)


def test_repl_session_persists_dialogue_without_memory_watermark(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)
    turn = session.add_user_turn("以后统一使用 pytest", run_id="run_1")

    reloaded = store.load(workspace, session.session_id)

    assert reloaded.dialogue[0].turn_id == turn.turn_id
    assert "last_extracted_run_id" not in reloaded.model_dump()



def test_repl_session_rejects_invalid_session_id(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")

    with pytest.raises(ValueError, match="session_<hex>"):
        store.load(workspace, "../active")


def test_repl_session_has_no_parallel_tool_result_store(tmp_path) -> None:
    session = ReplSessionMemory(workspace=str(tmp_path.resolve()))

    assert not hasattr(session, "tool_results")
    assert not hasattr(session, "record_tool_result")
    assert not hasattr(session, "lookup_read_file")
    assert not hasattr(session, "lookup_list_files")
