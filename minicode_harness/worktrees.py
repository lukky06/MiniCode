"""Git worktree lifecycle for isolated Sessions and bounded workers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Literal

from pydantic import BaseModel, Field

from minicode_harness.storage import default_data_dir, workspace_hash


WorktreeKind = Literal["session", "worker"]
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class WorktreeManifest(BaseModel):
    """Persistent binding between one project, branch, and worktree path."""

    name: str
    kind: WorktreeKind
    project_root: str
    workspace_root: str
    branch: str
    base_ref: str
    base_commit: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorktreeStatus(BaseModel):
    """Deterministic closeout facts for one managed worktree."""

    manifest: WorktreeManifest
    registered: bool
    exists: bool
    changed_files: list[str] = Field(default_factory=list)
    commits_ahead: int = 0

    @property
    def dirty(self) -> bool:
        return bool(self.changed_files or self.commits_ahead)


class GitWorktreeManager:
    """Create, reopen, inspect, and safely remove managed Git worktrees."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()

    def resolve_project_root(self, workspace: Path | str) -> Path:
        path = Path(workspace).expanduser().resolve()
        result = self._git(path, ["rev-parse", "--show-toplevel"])
        return Path(result.stdout.strip()).resolve()

    def open_or_create_session(
        self,
        project: Path | str,
        name: str,
        *,
        base_ref: str = "HEAD",
    ) -> WorktreeManifest:
        return self._open_or_create(
            project,
            name,
            kind="session",
            branch=f"minicode/session/{name}",
            base_ref=base_ref,
        )

    def create_worker(
        self,
        project: Path | str,
        name: str,
        *,
        base_ref: str = "HEAD",
    ) -> WorktreeManifest:
        return self._open_or_create(
            project,
            name,
            kind="worker",
            branch=f"minicode/worker/{name}",
            base_ref=base_ref,
            reopen=False,
        )

    def list(self, project: Path | str) -> list[WorktreeStatus]:
        project_root = self.resolve_project_root(project)
        manifest_dir = self._project_dir(project_root) / "manifests"
        if not manifest_dir.exists():
            return []
        statuses: list[WorktreeStatus] = []
        for path in sorted(manifest_dir.glob("*.json")):
            manifest = WorktreeManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            statuses.append(self.inspect(manifest))
        return statuses

    def inspect(self, manifest: WorktreeManifest) -> WorktreeStatus:
        project_root = Path(manifest.project_root).resolve()
        workspace_root = Path(manifest.workspace_root).resolve()
        registered_paths = self._registered_paths(project_root)
        changed_files: list[str] = []
        commits_ahead = 0
        if workspace_root.is_dir():
            status = self._git(workspace_root, ["status", "--porcelain"])
            changed_files = [line[3:] if len(line) > 3 else line for line in status.stdout.splitlines() if line]
            ahead = self._git(
                workspace_root,
                ["rev-list", "--count", f"{manifest.base_commit}..HEAD"],
            )
            commits_ahead = int(ahead.stdout.strip() or "0")
        return WorktreeStatus(
            manifest=manifest,
            registered=workspace_root in registered_paths,
            exists=workspace_root.is_dir(),
            changed_files=changed_files,
            commits_ahead=commits_ahead,
        )

    def remove(
        self,
        project: Path | str,
        name: str,
        *,
        force: bool = False,
        kind: WorktreeKind | None = None,
    ) -> WorktreeStatus:
        project_root = self.resolve_project_root(project)
        manifest = self._load_manifest(project_root, name, kind=kind)
        status = self.inspect(manifest)
        if status.dirty and not force:
            raise RuntimeError(
                f"Worktree '{name}' has {len(status.changed_files)} changed file(s) "
                f"and {status.commits_ahead} commit(s) ahead; use force to discard it."
            )
        workspace_root = Path(manifest.workspace_root)
        if status.registered:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(str(workspace_root))
            self._git(project_root, args)
        elif workspace_root.exists() and force:
            shutil.rmtree(workspace_root, ignore_errors=True)
        self._git(project_root, ["worktree", "prune"], check=False)
        self._git(project_root, ["branch", "-D", manifest.branch], check=False)
        self._manifest_path(project_root, manifest.kind, manifest.name).unlink(missing_ok=True)
        return status

    def _open_or_create(
        self,
        project: Path | str,
        name: str,
        *,
        kind: WorktreeKind,
        branch: str,
        base_ref: str,
        reopen: bool = True,
    ) -> WorktreeManifest:
        self._validate_name(name)
        project_root = self.resolve_project_root(project)
        manifest_path = self._manifest_path(project_root, kind, name)
        if manifest_path.exists():
            if not reopen:
                raise FileExistsError(f"Managed worktree already exists: {name}")
            manifest = WorktreeManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            status = self.inspect(manifest)
            if not status.exists or not status.registered:
                raise RuntimeError(
                    f"Managed worktree '{name}' is inconsistent; remove it explicitly before recreating."
                )
            return manifest

        base_commit = self._git(project_root, ["rev-parse", f"{base_ref}^{{commit}}"] ).stdout.strip()
        workspace_root = self._project_dir(project_root) / kind / name
        if workspace_root.exists():
            raise FileExistsError(f"Worktree path already exists: {workspace_root}")
        workspace_root.parent.mkdir(parents=True, exist_ok=True)
        self._git(
            project_root,
            ["worktree", "add", "-b", branch, str(workspace_root), base_commit],
        )
        manifest = WorktreeManifest(
            name=name,
            kind=kind,
            project_root=str(project_root),
            workspace_root=str(workspace_root),
            branch=branch,
            base_ref=base_ref,
            base_commit=base_commit,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        return manifest

    def _load_manifest(
        self,
        project_root: Path,
        name: str,
        *,
        kind: WorktreeKind | None,
    ) -> WorktreeManifest:
        self._validate_name(name)
        kinds: tuple[WorktreeKind, ...] = (kind,) if kind is not None else ("session", "worker")
        matches = [
            self._manifest_path(project_root, candidate, name)
            for candidate in kinds
            if self._manifest_path(project_root, candidate, name).exists()
        ]
        if not matches:
            raise FileNotFoundError(f"Managed worktree not found: {name}")
        if len(matches) > 1:
            raise ValueError(f"Worktree name is ambiguous across kinds: {name}")
        return WorktreeManifest.model_validate_json(matches[0].read_text(encoding="utf-8"))

    def _project_dir(self, project_root: Path) -> Path:
        return self.data_dir / "worktrees" / workspace_hash(project_root)

    def _manifest_path(self, project_root: Path, kind: WorktreeKind, name: str) -> Path:
        return self._project_dir(project_root) / "manifests" / f"{kind}-{name}.json"

    @staticmethod
    def _validate_name(name: str) -> None:
        if _VALID_NAME.fullmatch(name) is None:
            raise ValueError(
                "Worktree name must be 1-64 letters, digits, dots, underscores, or dashes "
                "and must start with a letter or digit."
            )

    def _registered_paths(self, project_root: Path) -> set[Path]:
        result = self._git(project_root, ["worktree", "list", "--porcelain"])
        paths: set[Path] = set()
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                paths.add(Path(line[len("worktree "):]).resolve())
        return paths

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _git(
        cwd: Path,
        args: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
        return result
