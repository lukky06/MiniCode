from __future__ import annotations

from pathlib import Path
import subprocess


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def init_git_repo(root: Path, files: dict[str, str] | None = None) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init")
    git(root, "config", "user.email", "minicode@example.test")
    git(root, "config", "user.name", "MiniCode Test")
    for relative, content in (files or {"README.md": "base\n"}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    return root, git(root, "rev-parse", "HEAD")
