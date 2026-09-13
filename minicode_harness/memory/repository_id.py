"""Path-independent repository identity for Repository Memory storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from minicode_harness.storage import default_data_dir


_ZERO_OID = re.compile(r"^0{40}$|^0{64}$")
_OID = re.compile(r"^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$")
_REMOTE_ORIGIN_SECTION = re.compile(r'^\s*\[\s*remote\s+"origin"\s*\]\s*$', re.IGNORECASE)
_SECTION = re.compile(r"^\s*\[.*\]\s*$")
_ORIGIN_URL = re.compile(r"^\s*url\s*=\s*(.*?)\s*$", re.IGNORECASE)


class RepositoryIdentityUnavailable(RuntimeError):
    """Raised when a Git repository cannot produce a stable identity quickly."""


@dataclass(frozen=True)
class RepositoryIdentity:
    repository_id: str
    source: str
    workspace_root: Path

    def memory_dir(self, data_dir: Path | str | None = None) -> Path:
        root = Path(data_dir) if data_dir is not None else default_data_dir()
        return root / "repositories" / self.repository_id / "memory"


def resolve_repository_identity(workspace: Path | str) -> RepositoryIdentity:
    """Resolve a stable identity without hashing the absolute checkout path."""

    resolved = Path(workspace).expanduser().resolve()
    root = _find_git_root(resolved)
    if root is None:
        root = resolved
        markers = _repository_markers(root)
        basis = f"local\n{root.name}\n" + "\n".join(markers)
        source = "local_markers"
    else:
        _, common_git_dir = _resolve_git_dirs(root)
        remote = _read_origin_remote(common_git_dir)
        if remote:
            basis = f"remote\n{_normalize_remote(remote)}"
            source = "git_remote"
        else:
            root_commit = _read_initial_commit(common_git_dir)
            if not root_commit:
                raise RepositoryIdentityUnavailable(
                    f"Git repository identity is unavailable for {root}."
                )
            basis = f"root_commit\n{root_commit}"
            source = "git_root_commit"

    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
    return RepositoryIdentity(
        repository_id=f"repo_{digest}",
        source=source,
        workspace_root=root,
    )


def _find_git_root(workspace: Path) -> Path | None:
    current = workspace if workspace.is_dir() else workspace.parent
    for candidate in (current, *current.parents):
        try:
            if (candidate / ".git").exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _resolve_git_dirs(root: Path) -> tuple[Path, Path]:
    marker = root / ".git"
    try:
        if marker.is_dir():
            git_dir = marker.resolve()
        elif marker.is_file():
            content = marker.read_text(encoding="utf-8", errors="replace").strip()
            if not content.lower().startswith("gitdir:"):
                raise RepositoryIdentityUnavailable(
                    f"Invalid Git directory marker for {root}."
                )
            value = content.split(":", 1)[1].strip()
            candidate = Path(value)
            git_dir = (
                candidate.resolve()
                if candidate.is_absolute()
                else (root / candidate).resolve()
            )
        else:
            raise RepositoryIdentityUnavailable(
                f"Git directory marker is unavailable for {root}."
            )

        common_marker = git_dir / "commondir"
        if not common_marker.is_file():
            return git_dir, git_dir
        common_value = common_marker.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
        common_candidate = Path(common_value)
        common_dir = (
            common_candidate.resolve()
            if common_candidate.is_absolute()
            else (git_dir / common_candidate).resolve()
        )
        return git_dir, common_dir
    except OSError as exc:
        raise RepositoryIdentityUnavailable(
            f"Git repository metadata is unavailable for {root}: {type(exc).__name__}."
        ) from exc


def _read_origin_remote(common_git_dir: Path) -> str:
    config = common_git_dir / "config"
    try:
        lines = config.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise RepositoryIdentityUnavailable(
            f"Git config is unavailable for {common_git_dir}: {type(exc).__name__}."
        ) from exc

    in_origin = False
    for line in lines:
        if _SECTION.match(line):
            in_origin = bool(_REMOTE_ORIGIN_SECTION.match(line))
            continue
        if not in_origin:
            continue
        match = _ORIGIN_URL.match(line)
        if match:
            return match.group(1).strip()
    return ""


def _read_initial_commit(common_git_dir: Path) -> str:
    reflog = common_git_dir / "logs" / "HEAD"
    try:
        lines = reflog.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise RepositoryIdentityUnavailable(
            f"Git HEAD reflog is unavailable for {common_git_dir}: {type(exc).__name__}."
        ) from exc

    for line in lines:
        fields = line.split(maxsplit=2)
        if len(fields) < 3:
            continue
        previous, current, metadata = fields
        if (
            _ZERO_OID.fullmatch(previous)
            and _OID.fullmatch(current)
            and "\tcommit (initial):" in metadata
        ):
            return current.lower()
    return ""


def _normalize_remote(remote: str) -> str:
    normalized = remote.strip().replace("\\", "/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _repository_markers(root: Path) -> list[str]:
    names = {
        "pyproject.toml",
        "setup.py",
        "package.json",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "Cargo.toml",
        "go.mod",
        "CMakeLists.txt",
        "Makefile",
    }
    try:
        present = sorted(path.name for path in root.iterdir() if path.name in names)
    except OSError:
        present = []
    return present or ["no-build-marker"]
