"""Repository mirror cache managed outside the agent-visible workspace."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess


class RepositoryCache:
    """Create and reuse bare mirrors for SWE-bench repositories."""

    def __init__(self, cache_root: Path | str) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.repositories_dir = self.cache_root / "repositories"

    def mirror_path(self, repo: str) -> Path:
        return self.repositories_dir / f"{_safe_repo_name(repo)}.git"

    def ensure(
        self,
        *,
        repo: str,
        base_commit: str,
        source_url: str | None = None,
        update: bool = True,
    ) -> Path:
        """Ensure a mirror contains ``base_commit`` and return its path."""

        mirror = self.mirror_path(repo)
        self.repositories_dir.mkdir(parents=True, exist_ok=True)
        url = source_url or _github_url(repo)

        if mirror.exists() and not mirror.is_dir():
            raise RuntimeError(f"Repository cache path is not a directory: {mirror}")
        if mirror.exists() and not _is_bare_repository(mirror):
            shutil.rmtree(mirror)

        if not mirror.exists():
            _run_git(
                ["init", "--bare", str(mirror)],
                error_prefix=f"Failed to initialize repository cache for {repo}",
            )
            _run_git(
                ["--git-dir", str(mirror), "remote", "add", "origin", url],
                error_prefix=f"Failed to configure repository cache for {repo}",
            )

        if update and not self.contains_commit(mirror, base_commit):
            _run_git(
                [
                    "--git-dir",
                    str(mirror),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    base_commit,
                ],
                error_prefix=(
                    f"Failed to fetch base commit {base_commit} for {repo}"
                ),
            )
        if not self.contains_commit(mirror, base_commit):
            raise LookupError(
                f"base_commit {base_commit} was not found in repository mirror {mirror}"
            )
        return mirror

    @staticmethod
    def contains_commit(mirror: Path | str, commit: str) -> bool:
        result = subprocess.run(
            [
                "git",
                "--git-dir",
                str(Path(mirror)),
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.returncode == 0


def _is_bare_repository(path: Path | str) -> bool:
    result = subprocess.run(
        [
            "git",
            "--git-dir",
            str(Path(path)),
            "rev-parse",
            "--is-bare-repository",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _github_url(repo: str) -> str:
    normalized = repo.strip().strip("/")
    if not normalized or "/" not in normalized:
        raise ValueError(
            "repo must use owner/name form unless an explicit source_url is supplied"
        )
    return f"https://github.com/{normalized}.git"


def _safe_repo_name(repo: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "__", repo.strip())
    if not normalized:
        raise ValueError("repo must not be empty")
    return normalized


def _run_git(arguments: list[str], *, error_prefix: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{error_prefix}: {detail}")
    return result
