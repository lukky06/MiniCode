from __future__ import annotations

from pathlib import Path

from minicode_harness.swebench import (
    RepositoryCache,
    SweBenchInstance,
    SweBenchWorkspaceManager,
)

from .helpers import git, init_git_repo


def test_repository_cache_repairs_incomplete_mirror(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(tmp_path / "source")
    cache = RepositoryCache(tmp_path / "cache")
    mirror = cache.mirror_path("owner/repo")
    mirror.mkdir(parents=True)
    (mirror / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    repaired = cache.ensure(
        repo="owner/repo",
        base_commit=base_commit,
        source_url=str(source),
    )

    assert repaired == mirror
    assert RepositoryCache.contains_commit(repaired, base_commit)
    assert (repaired / "config").is_file()


def test_workspace_uses_base_commit_and_reuses_mirror(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(tmp_path / "source")
    cache = RepositoryCache(tmp_path / "cache")
    manager = SweBenchWorkspaceManager(
        output_root=tmp_path / "output",
        repository_cache=cache,
    )
    first = SweBenchInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit=base_commit,
        problem_statement="Fix one",
    )
    second = first.model_copy(update={"instance_id": "owner__repo-2"})

    first_manifest = manager.prepare(first, source_url=str(source))
    second_manifest = manager.prepare(second, source_url=str(source))

    assert first_manifest.mirror_path == second_manifest.mirror_path
    assert Path(first_manifest.workspace) != Path(second_manifest.workspace)
    assert git(Path(first_manifest.workspace), "rev-parse", "HEAD") == base_commit
    assert git(Path(first_manifest.workspace), "status", "--porcelain") == ""

    (Path(first_manifest.workspace) / "README.md").write_text("changed\n", encoding="utf-8")
    assert (Path(second_manifest.workspace) / "README.md").read_text(
        encoding="utf-8"
    ) == "base\n"

    manager.cleanup(first_manifest)
    assert not Path(first_manifest.workspace).exists()
    assert Path(second_manifest.workspace).is_dir()
    assert Path(first_manifest.mirror_path).is_dir()
