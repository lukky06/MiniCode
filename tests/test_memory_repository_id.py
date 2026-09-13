from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import subprocess

import pytest

from minicode_harness.memory import resolve_repository_identity
from minicode_harness.memory.repository_id import RepositoryIdentityUnavailable


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_repository_identity_survives_checkout_path_change(tmp_path: Path) -> None:
    source = tmp_path / "source" / "project"
    source.mkdir(parents=True)
    _git(source, "init")
    _git(source, "config", "user.email", "memory@example.com")
    _git(source, "config", "user.name", "Memory Test")
    (source / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
    _git(source, "add", "pyproject.toml")
    _git(source, "commit", "-m", "initial")

    moved = tmp_path / "moved" / "different-name"
    shutil.copytree(source, moved)

    first = resolve_repository_identity(source)
    second = resolve_repository_identity(moved)

    assert first.repository_id == second.repository_id
    assert first.source == "git_root_commit"
    assert str(source.resolve()) not in first.repository_id


def test_repository_identity_does_not_invoke_git_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "memory@example.com")
    _git(workspace, "config", "user.name", "Memory Test")
    (workspace / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
    _git(workspace, "add", "pyproject.toml")
    _git(workspace, "commit", "-m", "initial")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("repository identity must not spawn git")
        ),
    )

    identity = resolve_repository_identity(workspace)

    assert identity.source == "git_root_commit"


def test_repository_identity_does_not_fallback_when_git_metadata_is_incomplete(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    git_dir = workspace / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")

    with pytest.raises(RepositoryIdentityUnavailable, match="Git repository identity"):
        resolve_repository_identity(workspace)


def test_repository_identity_does_not_treat_clone_reflog_as_root_commit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    git_dir = workspace / ".git"
    (git_dir / "logs").mkdir(parents=True)
    (git_dir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    cloned_head = "ba8c6b361b832e0e62715dd683b7f0e80c381f88"
    (git_dir / "logs" / "HEAD").write_text(
        f"{'0' * 40} {cloned_head} Test <test@example.com> 1 +0000\tclone: from example\n",
        encoding="utf-8",
    )

    with pytest.raises(RepositoryIdentityUnavailable, match="Git repository identity"):
        resolve_repository_identity(workspace)


def test_repository_identity_uses_origin_without_git_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    _git(workspace, "init")
    _git(workspace, "remote", "add", "origin", "https://github.com/example/project.git")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("repository identity must not spawn git")
        ),
    )

    identity = resolve_repository_identity(workspace)

    assert identity.source == "git_remote"


def test_repository_identity_reads_common_git_dir_for_linked_worktree(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    common = tmp_path / "repo" / ".git"
    git_dir = common / "worktrees" / "worktree"
    git_dir.mkdir(parents=True)
    (workspace / ".git").write_text(
        f"gitdir: {git_dir.as_posix()}\n",
        encoding="utf-8",
    )
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    (common / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (common / "logs").mkdir()
    root_commit = "5a18b6b43554c9130bb723dcdcc30e0b670720cf"
    (common / "logs" / "HEAD").write_text(
        f"{'0' * 40} {root_commit} Test <test@example.com> 1 +0000\tcommit (initial): init\n",
        encoding="utf-8",
    )

    identity = resolve_repository_identity(workspace)

    assert identity.source == "git_root_commit"
    expected_digest = hashlib.sha256(
        f"root_commit\n{root_commit}".encode("utf-8")
    ).hexdigest()[:24]
    assert identity.repository_id == f"repo_{expected_digest}"
