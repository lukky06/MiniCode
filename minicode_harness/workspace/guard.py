"""Workspace path validation and sensitive path filtering."""

from __future__ import annotations

from pathlib import Path


SENSITIVE_DIRECTORY_NAMES = frozenset({".git", ".mini-code", "runs"})
SENSITIVE_FILE_NAMES = frozenset({".env", "id_rsa"})
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".jks"})


class WorkspaceAccessError(ValueError):
    """Raised when a path is not safe to access from the workspace tools."""


class WorkspaceGuard:
    """Resolve tool paths and enforce the workspace boundary."""

    def __init__(self, workspace: Path | str) -> None:
        workspace_path = Path(workspace)
        self.root = workspace_path.resolve()
        if not self.root.exists():
            raise WorkspaceAccessError(f"Workspace does not exist: {self.root}")
        if not self.root.is_dir():
            raise WorkspaceAccessError(f"Workspace is not a directory: {self.root}")

    def resolve(self, path: Path | str = ".") -> Path:
        """Resolve a path inside the workspace and reject unsafe targets."""

        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else self.root / raw_path
        resolved = candidate.resolve()

        if not self.is_within_workspace(resolved):
            raise WorkspaceAccessError(
                f"Path escapes workspace: {path!s}"
            )
        if self.is_sensitive(resolved):
            raise WorkspaceAccessError(
                f"Path is denied by the sensitive path policy: {self.relative_path(resolved)}"
            )
        return resolved

    def is_within_workspace(self, path: Path | str) -> bool:
        """Return whether a resolved path stays inside the workspace."""

        resolved = Path(path).resolve()
        return resolved == self.root or resolved.is_relative_to(self.root)

    def is_sensitive(self, path: Path | str) -> bool:
        """Return whether a workspace path matches the sensitive path denylist."""

        resolved = Path(path).resolve()
        if not self.is_within_workspace(resolved):
            return True

        relative = resolved.relative_to(self.root)
        parts = [part.lower() for part in relative.parts]

        if any(part in SENSITIVE_DIRECTORY_NAMES for part in parts[:-1]):
            return True

        name = resolved.name.lower()
        if name in SENSITIVE_FILE_NAMES:
            return True
        if any(name.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
            return True
        if resolved.is_dir() and name in SENSITIVE_DIRECTORY_NAMES:
            return True

        return False

    def relative_path(self, path: Path | str) -> str:
        """Return a POSIX-style path relative to the workspace."""

        resolved = Path(path).resolve()
        if resolved == self.root:
            return "."
        return resolved.relative_to(self.root).as_posix()
