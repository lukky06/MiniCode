"""Workspace-scoped REPL session persistence.

A terminal process owns one explicit conversation Session. Sessions are grouped
by creation date and each Session owns the Runs produced from that conversation.
Canonical tool calls and results remain in the append-only ``message_history``.
Model requests derive a temporary projection from the separate lightweight
``compaction_state``; that projection is never persisted as another history.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from minicode_harness.context.compaction_state import SessionCompactionState
from minicode_harness.storage import default_data_dir


MAX_DIALOGUE_TURNS = 24
MAX_COMMAND_APPROVAL_GRANTS = 128
SESSION_ID_PATTERN = re.compile(r"^session_[0-9a-f]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DialogueTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: f"turn_{uuid4().hex[:12]}")
    role: str
    content: str
    run_id: str | None = None
    history_length: int | None = Field(default=None, ge=0)
    created_at: str = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def require_completed_run_boundary(self) -> "DialogueTurn":
        if self.role == "assistant" and self.run_id is not None and self.history_length is None:
            raise ValueError("Completed assistant turns require history_length.")
        return self


class ReplSessionMemory(BaseModel):
    """Canonical conversation history for one interactive terminal Session."""

    model_config = {"extra": "forbid"}

    session_id: str = Field(default_factory=lambda: f"session_{uuid4().hex[:12]}")
    workspace: str
    name: str | None = Field(default=None, max_length=80)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    dialogue: list[DialogueTurn] = Field(default_factory=list)
    message_history: list[dict[str, object]] = Field(default_factory=list)
    compaction_state: SessionCompactionState = Field(
        default_factory=SessionCompactionState
    )
    command_approval_grants: list[str] = Field(default_factory=list, max_length=MAX_COMMAND_APPROVAL_GRANTS)
    _persist_callback: Callable[["ReplSessionMemory"], None] | None = PrivateAttr(default=None)
    _is_persisted: bool = PrivateAttr(default=False)

    def bind_persistence(
        self,
        callback: Callable[["ReplSessionMemory"], None],
        *,
        persisted: bool = False,
    ) -> None:
        self._persist_callback = callback
        self._is_persisted = persisted

    def ensure_persisted(self) -> None:
        """Persist a lazily-created Session before its first durable Run."""

        if self._is_persisted:
            return
        if self._persist_callback is None:
            return
        self._persist_callback(self)

    def add_user_turn(self, content: str, *, run_id: str | None = None) -> DialogueTurn:
        turn = DialogueTurn(role="user", content=content, run_id=run_id)
        self._append_dialogue(turn)
        return turn

    def add_assistant_turn(
        self,
        content: str,
        *,
        history_length: int,
        run_id: str | None = None,
    ) -> DialogueTurn | None:
        if not content.strip():
            return None
        turn = DialogueTurn(
            role="assistant",
            content=content,
            run_id=run_id,
            history_length=history_length,
        )
        self._append_dialogue(turn)
        return turn

    def rename(self, name: str) -> None:
        normalized = " ".join(name.split())
        if not normalized:
            raise ValueError("Session name must not be empty.")
        if len(normalized) > 80:
            raise ValueError("Session name must not exceed 80 characters.")
        self.name = normalized
        self._persist()

    def load_message_history(self) -> list[dict[str, object]]:
        """Return the complete persisted canonical protocol history."""

        return deepcopy(self.message_history)

    def load_compaction_state(self) -> SessionCompactionState:
        """Return an isolated copy of the model-projection state."""

        return self.compaction_state.model_copy(deep=True)

    def replace_message_history(self, messages: list[dict[str, object]]) -> None:
        """Replace the single canonical history."""

        self.message_history = deepcopy(messages)
        self._persist()

    def replace_compaction_state(self, state: SessionCompactionState) -> None:
        """Persist only lightweight projection cursors and semantic summary."""

        self.compaction_state = state.model_copy(deep=True)
        self._persist()

    def replace_session_state(
        self,
        *,
        messages: list[dict[str, object]],
        compaction_state: SessionCompactionState,
    ) -> None:
        """Atomically update canonical history and its projection state."""

        self.message_history = deepcopy(messages)
        self.compaction_state = compaction_state.model_copy(deep=True)
        self._persist()

    def has_command_approval_grant(self, executable: str) -> bool:
        return executable in self.command_approval_grants

    def grant_command_approval(self, executable: str) -> None:
        if executable in self.command_approval_grants:
            return
        if len(self.command_approval_grants) >= MAX_COMMAND_APPROVAL_GRANTS:
            raise ValueError("Session command approval grant limit reached.")
        self.command_approval_grants.append(executable)
        self._persist()

    def _append_dialogue(self, turn: DialogueTurn) -> None:
        self.dialogue.append(turn)
        del self.dialogue[:-MAX_DIALOGUE_TURNS]
        self._persist()

    def _persist(self) -> None:
        if self._persist_callback is not None:
            self._persist_callback(self)


class ReplSessionStore:
    """Persist independent Sessions under ``sessions/YYYY/MM/DD``."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()

    @property
    def sessions_root(self) -> Path:
        return self.data_dir / "sessions"

    def create(
        self,
        workspace: Path | str,
        *,
        persist: bool = True,
    ) -> ReplSessionMemory:
        resolved = self._resolved_workspace(workspace)
        session = ReplSessionMemory(workspace=resolved)
        session.bind_persistence(self.save, persisted=False)
        if persist:
            self.save(session)
        return session

    def fork(
        self,
        session: ReplSessionMemory,
        *,
        through_turn: int | None = None,
    ) -> ReplSessionMemory:
        """Create a new linear Session from the full history or one completed Run boundary."""

        resolved = self._resolved_workspace(session.workspace)
        if session.workspace != resolved:
            raise ValueError("Session workspace does not match the current workspace.")

        dialogue = deepcopy(session.dialogue)
        messages = session.load_message_history()
        compaction_state = session.load_compaction_state()
        if through_turn is not None:
            if through_turn < 1:
                raise ValueError("Fork turn must be at least 1.")
            completed = [
                (index, turn)
                for index, turn in enumerate(dialogue)
                if turn.role == "assistant" and turn.run_id is not None
            ]
            if through_turn > len(completed):
                raise ValueError(
                    f"Fork turn {through_turn} exceeds completed Run count {len(completed)}."
                )
            dialogue_index, boundary = completed[through_turn - 1]
            if boundary.history_length > len(messages):
                raise ValueError("Persisted fork boundary exceeds canonical message history.")
            dialogue = dialogue[: dialogue_index + 1]
            messages = messages[: boundary.history_length]
            compaction_state = SessionCompactionState()

        forked = ReplSessionMemory(
            workspace=resolved,
            dialogue=dialogue,
            message_history=messages,
            compaction_state=compaction_state,
        )
        forked.bind_persistence(self.save, persisted=False)
        self.save(forked)
        return forked

    def load(
        self,
        workspace: Path | str,
        session_id: str,
    ) -> ReplSessionMemory:
        resolved = self._resolved_workspace(workspace)
        self._validate_session_id(session_id)
        path = self.session_path(resolved, session_id)
        session = self._read_session(path, resolved)
        session.bind_persistence(self.save, persisted=True)
        return session

    def load_latest(self, workspace: Path | str) -> ReplSessionMemory:
        sessions = self.list(workspace, limit=1)
        if not sessions:
            raise FileNotFoundError("No previous session exists for this workspace.")
        return sessions[0]

    def list(
        self,
        workspace: Path | str,
        *,
        limit: int | None = None,
    ) -> list[ReplSessionMemory]:
        resolved = self._resolved_workspace(workspace)
        sessions: list[ReplSessionMemory] = []
        if self.sessions_root.is_dir():
            for path in self.sessions_root.glob("*/*/*/session_*/session.json"):
                session = ReplSessionMemory.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                self._validate_session_id(session.session_id)
                if session.workspace != resolved:
                    continue
                session.bind_persistence(self.save, persisted=True)
                sessions.append(session)
        sessions.sort(
            key=lambda item: (item.updated_at, item.created_at, item.session_id),
            reverse=True,
        )
        if limit is None:
            return sessions
        return sessions[:max(0, limit)]

    def save(self, session: ReplSessionMemory) -> Path:
        resolved = self._resolved_workspace(session.workspace)
        self._validate_session_id(session.session_id)
        session.workspace = resolved
        now = _parse_timestamp(_utc_now())
        updated = _parse_timestamp(session.updated_at)
        session.message_history = deepcopy(session.message_history)
        session.compaction_state = session.compaction_state.model_copy(deep=True)
        existing = self._find_session_path(session.session_id)
        if existing is not None:
            persisted = self._read_session(existing, resolved)
            floor = max(updated, _parse_timestamp(persisted.updated_at))
            updated = now if now > floor else floor + timedelta(microseconds=1)
        else:
            latest = self.list(resolved, limit=1)
            if latest:
                floor = max(updated, _parse_timestamp(latest[0].updated_at))
                updated = now if now > floor else floor + timedelta(microseconds=1)
            elif now > updated:
                updated = now
        session.updated_at = updated.isoformat()
        path = existing or self._new_session_path(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(session.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        session._is_persisted = True
        return path

    def delete(self, workspace: Path | str, session_id: str) -> bool:
        self._validate_session_id(session_id)
        resolved = self._resolved_workspace(workspace)
        path = self._find_session_path(session_id)
        if path is None:
            return False
        self._read_session(path, resolved)
        path.unlink()
        session_directory = path.parent
        runs_directory = session_directory / "runs"
        if not runs_directory.exists() and not any(session_directory.iterdir()):
            session_directory.rmdir()
        return True

    def session_directory(self, session: ReplSessionMemory) -> Path:
        existing = self._find_session_path(session.session_id)
        return existing.parent if existing is not None else self._new_session_path(session).parent

    def session_path(self, workspace: Path | str, session_id: str) -> Path:
        self._validate_session_id(session_id)
        resolved = self._resolved_workspace(workspace)
        path = self._find_session_path(session_id)
        if path is None:
            raise FileNotFoundError(f"Session not found: {session_id}")
        try:
            session = ReplSessionMemory.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid Session file: {path}") from exc
        if session.workspace != resolved:
            raise FileNotFoundError(f"Session not found for workspace: {session_id}")
        return path

    def _new_session_path(self, session: ReplSessionMemory) -> Path:
        created = _parse_timestamp(session.created_at)
        return (
            self.sessions_root
            / f"{created.year:04d}"
            / f"{created.month:02d}"
            / f"{created.day:02d}"
            / session.session_id
            / "session.json"
        )

    def _find_session_path(self, session_id: str) -> Path | None:
        if not self.sessions_root.is_dir():
            return None
        matches = [
            path
            for path in self.sessions_root.glob(f"*/*/*/{session_id}/session.json")
            if path.is_file()
        ]
        if len(matches) > 1:
            raise ValueError(f"Session ID is duplicated: {session_id}")
        return matches[0] if matches else None

    def _read_session(self, path: Path, workspace: str) -> ReplSessionMemory:
        session = ReplSessionMemory.model_validate_json(path.read_text(encoding="utf-8"))
        self._validate_session_id(session.session_id)
        if session.workspace != workspace:
            raise ValueError("Session workspace does not match the requested workspace.")
        return session

    def _resolved_workspace(self, workspace: Path | str) -> str:
        self._ensure_outside_workspace(workspace)
        return str(Path(workspace).expanduser().resolve())

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError("Session ID must use the format session_<hex>.")

    def _ensure_outside_workspace(self, workspace: Path | str) -> None:
        workspace_path = Path(workspace).expanduser().resolve()
        data_path = self.data_dir.expanduser().resolve()
        if data_path == workspace_path or data_path.is_relative_to(workspace_path):
            raise ValueError("REPL session data_dir must not be inside the workspace.")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid Session timestamp: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
