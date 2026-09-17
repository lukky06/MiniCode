"""File-backed Run Store primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from pydantic import BaseModel, Field

from minicode_harness.storage import default_data_dir


RUN_ID_PATTERN = re.compile(r"^run_(?P<date>\d{8})_(?P<sequence>\d{3})$")
SESSION_ID_PATTERN = re.compile(r"^session_[0-9a-f]+$")
RUN_CHILD_DIRECTORIES = ("artifacts", "checkpoints", "approvals", "debug")
RUN_METADATA_FILE = "run.json"


@dataclass(frozen=True)
class RunDirectory:
    """Created run directory metadata."""

    run_id: str
    path: Path


class RunSession(BaseModel):
    """Persisted metadata for one Agent Run."""

    run_id: str
    task: str
    workspace: str
    conversation_session_id: str | None = None
    provider: str = "openai"
    model: str | None = None
    no_write: bool = False
    approval_policy: str = "on-request"
    permission_mode: str = "read-only"
    sandbox_mode: str = "local"
    sandbox_image: str | None = None
    command_rules: list[dict[str, object]] = Field(default_factory=list)
    collaboration_mode: str = "default"
    skills: str | None = None
    no_skills: bool = False
    repository_memory_enabled: bool = True
    mcp_config: str | None = None
    subagents_enabled: bool = True
    worktree_workers_enabled: bool = True
    status: str = "created"
    current_step: int = 0
    max_steps: int = 50
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def generate_run_id(root: Path, now: datetime | None = None) -> str:
    """Generate the next date-scoped run ID for one flat run root."""

    timestamp = now or datetime.now()
    date_prefix = timestamp.strftime("%Y%m%d")
    highest_sequence = 0

    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = RUN_ID_PATTERN.match(child.name)
            if match is None or match.group("date") != date_prefix:
                continue
            highest_sequence = max(highest_sequence, int(match.group("sequence")))

    return f"run_{date_prefix}_{highest_sequence + 1:03d}"


def default_run_root() -> Path:
    """Return the standalone Run root.

    Interactive Runs are stored below their conversation Session. This root is
    retained for explicit custom stores, benchmark outputs, and standalone Runs.
    """

    configured = os.environ.get("MINICODE_RUNS_DIR")
    if configured:
        return Path(configured)
    return default_data_dir() / "runs"


class RunStore:
    """Create, locate, and load Run directories.

    A default Store acts as a catalog over dated conversation Sessions. New
    Runs with ``conversation_session_id`` are created below that Session's
    ``runs`` directory. A Store constructed with an explicit ``root`` remains
    a flat, isolated Store for tests, benchmarks, and adapters.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        sessions_root: Path | str | None = None,
    ) -> None:
        self._catalog_enabled = root is None
        self.root = Path(root) if root is not None else default_run_root()
        self.sessions_root = (
            Path(sessions_root)
            if sessions_root is not None
            else default_data_dir() / "sessions"
        )

    def create_run_directory(self, run_id: str | None = None) -> RunDirectory:
        """Create one run directory under this Store's explicit root."""

        return self._create_run_directory(self.root, run_id)

    def create_run(
        self,
        task: str,
        workspace: Path,
        run_id: str | None = None,
        max_steps: int = 50,
        provider: str = "openai",
        model: str | None = None,
        conversation_session_id: str | None = None,
        no_write: bool = False,
        approval_policy: str = "on-request",
        permission_mode: str = "read-only",
        sandbox_mode: str = "local",
        sandbox_image: str | None = None,
        command_rules: list[dict[str, object]] | None = None,
        collaboration_mode: str = "default",
        skills: str | None = None,
        no_skills: bool = False,
        repository_memory_enabled: bool = True,
        mcp_config: str | None = None,
        subagents_enabled: bool = True,
        worktree_workers_enabled: bool = True,
    ) -> RunSession:
        """Create the Run layout and persist ``run.json`` metadata."""

        target_root = self._target_root(conversation_session_id)
        self._ensure_root_outside_workspace(workspace, target_root)
        generated_id = run_id
        if generated_id is None and self._catalog_enabled:
            generated_id = self._generate_catalog_run_id()
        run_directory = self._create_run_directory(target_root, generated_id)
        for child_directory in RUN_CHILD_DIRECTORIES:
            (run_directory.path / child_directory).mkdir()

        now = datetime.now(timezone.utc)
        session = RunSession(
            run_id=run_directory.run_id,
            task=task,
            workspace=str(workspace.resolve()),
            conversation_session_id=conversation_session_id,
            provider=provider,
            model=model,
            no_write=no_write,
            approval_policy=approval_policy,
            permission_mode=permission_mode,
            sandbox_mode=sandbox_mode,
            sandbox_image=sandbox_image,
            command_rules=list(command_rules or []),
            collaboration_mode=collaboration_mode,
            skills=skills,
            no_skills=no_skills,
            repository_memory_enabled=repository_memory_enabled,
            mcp_config=mcp_config,
            subagents_enabled=subagents_enabled,
            worktree_workers_enabled=worktree_workers_enabled,
            max_steps=max_steps,
            created_at=now,
            updated_at=now,
        )
        self._write_session(run_directory.path, session)
        (run_directory.path / "trace.jsonl").write_text("", encoding="utf-8")
        return session

    def path_for(self, run_id: str) -> Path:
        """Return the path for a Run ID, including dated Session storage."""

        self._validate_run_id(run_id)
        if not self._catalog_enabled:
            return self.root / run_id
        matches = self._catalog_paths_for(run_id)
        if len(matches) > 1:
            rendered = ", ".join(str(path) for path in matches)
            raise ValueError(f"Run ID is ambiguous across Sessions: {run_id}: {rendered}")
        if matches:
            return matches[0]
        return self.root / run_id

    def list_run_ids(
        self,
        *,
        workspace: Path | str | None = None,
        conversation_session_id: str | None = None,
    ) -> list[str]:
        """Return matching Run IDs sorted from oldest to newest."""

        if not self._catalog_enabled:
            if not self.root.exists():
                return []
            run_ids = sorted(
                child.name
                for child in self.root.iterdir()
                if child.is_dir() and RUN_ID_PATTERN.match(child.name) is not None
            )
        else:
            run_ids = sorted(
                {
                    path.name
                    for path in self._all_catalog_run_paths()
                    if RUN_ID_PATTERN.match(path.name) is not None
                }
            )

        if workspace is None and conversation_session_id is None:
            return run_ids
        if conversation_session_id is not None:
            self._validate_session_id(conversation_session_id)
        workspace_key = (
            self._workspace_key(workspace)
            if workspace is not None
            else None
        )
        matched: list[str] = []
        for run_id in run_ids:
            try:
                session = self.load_session(run_id)
            except (FileNotFoundError, OSError, ValueError):
                continue
            if (
                workspace_key is not None
                and self._workspace_key(session.workspace) != workspace_key
            ):
                continue
            if (
                conversation_session_id is not None
                and session.conversation_session_id != conversation_session_id
            ):
                continue
            matched.append(run_id)
        return matched

    def latest_run_id(
        self,
        *,
        workspace: Path | str | None = None,
        conversation_session_id: str | None = None,
    ) -> str:
        """Return the newest matching Run ID in this Store."""

        run_ids = self.list_run_ids(
            workspace=workspace,
            conversation_session_id=conversation_session_id,
        )
        if not run_ids:
            location = self.sessions_root if self._catalog_enabled else self.root
            raise FileNotFoundError(f"No runs found under {location}.")
        return run_ids[-1]

    @staticmethod
    def _workspace_key(workspace: Path | str) -> str:
        resolved = str(Path(workspace).expanduser().resolve())
        return os.path.normcase(os.path.normpath(resolved))

    def load_session(self, run_id: str) -> RunSession:
        """Load persisted Run metadata from ``run.json``."""

        run_path = self.path_for(run_id)
        metadata_path = self._metadata_path(run_path)
        if metadata_path is None:
            raise FileNotFoundError(f"Run metadata not found: {run_path / RUN_METADATA_FILE}")
        return RunSession.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    def update_session_state(
        self,
        run_id: str,
        *,
        status: str,
        current_step: int,
    ) -> RunSession:
        """Persist the latest terminal Run state in ``run.json``."""

        session = self.load_session(run_id)
        updated = session.model_copy(
            update={
                "status": status,
                "current_step": max(0, int(current_step)),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._write_session(self.path_for(run_id), updated)
        return updated

    def _target_root(self, conversation_session_id: str | None) -> Path:
        if not self._catalog_enabled or conversation_session_id is None:
            return self.root
        self._validate_session_id(conversation_session_id)
        session_directory = self._find_session_directory(conversation_session_id)
        if session_directory is None:
            raise FileNotFoundError(
                f"Conversation Session not found: {conversation_session_id}"
            )
        return session_directory / "runs"

    def _create_run_directory(
        self,
        root: Path,
        run_id: str | None,
    ) -> RunDirectory:
        root.mkdir(parents=True, exist_ok=True)

        if run_id is not None:
            self._validate_run_id(run_id)
            path = root / run_id
            path.mkdir(parents=False, exist_ok=False)
            return RunDirectory(run_id=run_id, path=path)

        while True:
            generated_run_id = generate_run_id(root)
            path = root / generated_run_id
            try:
                path.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return RunDirectory(run_id=generated_run_id, path=path)

    def _generate_catalog_run_id(self, now: datetime | None = None) -> str:
        timestamp = now or datetime.now()
        date_prefix = timestamp.strftime("%Y%m%d")
        highest_sequence = 0
        for run_id in self.list_run_ids():
            match = RUN_ID_PATTERN.match(run_id)
            if match is None or match.group("date") != date_prefix:
                continue
            highest_sequence = max(highest_sequence, int(match.group("sequence")))
        return f"run_{date_prefix}_{highest_sequence + 1:03d}"

    def _write_session(self, run_path: Path, session: RunSession) -> None:
        session_json = session.model_dump(mode="json")
        metadata_path = run_path / RUN_METADATA_FILE
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(session_json, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)

    def _all_catalog_run_paths(self) -> list[Path]:
        paths: list[Path] = []
        if self.sessions_root.is_dir():
            paths.extend(
                path
                for path in self.sessions_root.glob("*/*/*/session_*/runs/run_*")
                if path.is_dir() and RUN_ID_PATTERN.match(path.name) is not None
            )
        if self.root.is_dir():
            paths.extend(
                path
                for path in self.root.iterdir()
                if path.is_dir() and RUN_ID_PATTERN.match(path.name) is not None
            )
        return paths

    def _catalog_paths_for(self, run_id: str) -> list[Path]:
        return [path for path in self._all_catalog_run_paths() if path.name == run_id]

    def _find_session_directory(self, session_id: str) -> Path | None:
        if not self.sessions_root.is_dir():
            return None
        matches = [
            path.parent
            for path in self.sessions_root.glob(f"*/*/*/{session_id}/session.json")
            if path.is_file()
        ]
        if len(matches) > 1:
            raise ValueError(f"Conversation Session ID is duplicated: {session_id}")
        return matches[0] if matches else None

    @staticmethod
    def _metadata_path(run_path: Path) -> Path | None:
        current = run_path / RUN_METADATA_FILE
        return current if current.is_file() else None

    @staticmethod
    def _ensure_root_outside_workspace(workspace: Path | str, root: Path) -> None:
        workspace_path = Path(workspace).resolve()
        run_root = root.resolve()
        if run_root == workspace_path or run_root.is_relative_to(workspace_path):
            raise ValueError("Run store root must not be inside the workspace.")

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if RUN_ID_PATTERN.match(run_id) is None:
            raise ValueError(
                "Run ID must use the format run_YYYYMMDD_NNN, for example run_20260630_001."
            )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError("Session ID must use the format session_<hex>.")
