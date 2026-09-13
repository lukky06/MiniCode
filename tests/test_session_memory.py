from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from minicode_harness.context import SessionCompactionState
import minicode_harness.state.session_memory as session_memory_module
from minicode_harness.state import ReplSessionStore


def test_latest_session_prefers_later_creation_when_timestamps_tie(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 8, tzinfo=timezone.utc)

    ids = iter([
        "ffffffffffff0000",
        "0000000000010000",
        "0000000000000000",
    ])
    monkeypatch.setattr(session_memory_module, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        session_memory_module,
        "uuid4",
        lambda: SimpleNamespace(hex=next(ids)),
    )

    first = store.create(workspace)
    second = store.create(workspace)

    assert first.updated_at < second.updated_at
    assert store.load_latest(workspace).session_id == second.session_id

    forked = store.fork(second)

    assert second.updated_at < forked.updated_at
    assert store.load_latest(workspace).session_id == forked.session_id


def test_session_rename_persists_normalized_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)

    session.rename("  API   cleanup  ")

    loaded = store.load(workspace, session.session_id)
    assert loaded.name == "API cleanup"

    with pytest.raises(ValueError, match="80"):
        session.rename("x" * 81)


def test_session_partial_fork_uses_completed_run_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)

    session.replace_message_history(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
        ]
    )
    session.add_user_turn("first", run_id="run_1")
    session.add_assistant_turn(
        "first answer",
        run_id="run_1",
        history_length=2,
    )
    session.replace_message_history(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "second answer"},
        ]
    )
    session.add_user_turn("second", run_id="run_2")
    session.add_assistant_turn(
        "second answer",
        run_id="run_2",
        history_length=4,
    )
    session.grant_command_approval("git")

    forked = store.fork(session, through_turn=1)

    assert forked.session_id != session.session_id
    assert [turn.content for turn in forked.dialogue] == ["first", "first answer"]
    assert forked.load_message_history() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
    ]
    assert forked.load_compaction_state() == SessionCompactionState()
    assert forked.command_approval_grants == []
    assert forked.name is None


def test_session_full_fork_supports_legacy_turns_but_partial_requires_boundary(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ReplSessionStore(tmp_path / "data")
    session = store.create(workspace)
    session.replace_message_history(
        [
            {"role": "user", "content": "legacy"},
            {"role": "assistant", "content": "legacy answer"},
        ]
    )
    session.add_user_turn("legacy", run_id="run_legacy")
    session.add_assistant_turn("legacy answer", run_id="run_legacy")

    full = store.fork(session)
    assert full.load_message_history() == session.load_message_history()
    assert [turn.content for turn in full.dialogue] == [
        "legacy",
        "legacy answer",
    ]

    with pytest.raises(ValueError, match="boundary"):
        store.fork(session, through_turn=1)
