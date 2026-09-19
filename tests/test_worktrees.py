from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from minicode_harness.worktrees import GitWorktreeManager


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.name", "MiniCode Tests")
    _git(path, "config", "user.email", "minicode@example.invalid")
    (path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "add", "app.py")
    _git(path, "commit", "-m", "initial")
    return path


def test_session_worktree_is_reopenable_and_isolates_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    manager = GitWorktreeManager(tmp_path / "data")

    manifest = manager.open_or_create_session(repository, "feature-auth")
    worktree = Path(manifest.workspace_root)
    (worktree / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

    reopened = manager.open_or_create_session(repository, "feature-auth")
    status = manager.inspect(reopened)
    assert reopened == manifest
    assert status.registered is True
    assert status.changed_files == ["app.py"]
    assert (repository / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_worktree_name_rejects_path_traversal(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    manager = GitWorktreeManager(tmp_path / "data")

    with pytest.raises(ValueError, match="Worktree name"):
        manager.open_or_create_session(repository, "../escape")


def test_remove_refuses_dirty_worktree_without_force(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    manager = GitWorktreeManager(tmp_path / "data")
    manifest = manager.open_or_create_session(repository, "docs")
    worktree = Path(manifest.workspace_root)
    (worktree / "app.py").write_text("VALUE = 3\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed file"):
        manager.remove(repository, "docs")

    removed = manager.remove(repository, "docs", force=True)
    assert removed.dirty is True
    assert not worktree.exists()
    assert manager.list(repository) == []


def test_worker_worktree_does_not_reopen_existing_name(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    manager = GitWorktreeManager(tmp_path / "data")
    manager.create_worker(repository, "worker-0001")

    with pytest.raises(FileExistsError):
        manager.create_worker(repository, "worker-0001")


def test_worktree_list_fails_on_invalid_current_manifest(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    data_dir = tmp_path / "data"
    manager = GitWorktreeManager(data_dir)
    manager.open_or_create_session(repository, "broken")
    manifest_path = next(data_dir.rglob("*.json"))
    manifest_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError):
        manager.list(repository)
