"""Detached base-commit workspaces for individual SWE-bench instances."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from .models import SweBenchInstance, WorkspaceManifest
from .prediction import atomic_write_json
from .repository import RepositoryCache


class SweBenchWorkspaceManager:
    """Prepare one isolated Git worktree per SWE-bench instance."""

    def __init__(
        self,
        *,
        output_root: Path | str,
        repository_cache: RepositoryCache,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.repository_cache = repository_cache

    def instance_dir(self, instance_id: str, *, attempt: int = 1) -> Path:
        suffix = "" if attempt == 1 else f"__attempt_{attempt}"
        return self.output_root / "instances" / f"{instance_id}{suffix}"

    def prepare(
        self,
        instance: SweBenchInstance,
        *,
        attempt: int = 1,
        source_url: str | None = None,
        reset: bool = True,
    ) -> WorkspaceManifest:
        """Create and validate a detached worktree at ``base_commit``."""

        mirror = self.repository_cache.ensure(
            repo=instance.repo,
            base_commit=instance.base_commit,
            source_url=source_url,
        )
        instance_dir = self.instance_dir(instance.instance_id, attempt=attempt)
        workspace = instance_dir / "workspace"
        instance_dir.mkdir(parents=True, exist_ok=True)

        if workspace.exists() and reset:
            self._remove_worktree(mirror, workspace)
        if not workspace.exists():
            _run_git(
                [
                    "--git-dir",
                    str(mirror),
                    "worktree",
                    "add",
                    "--detach",
                    str(workspace),
                    instance.base_commit,
                ],
                error_prefix=f"Failed to create worktree for {instance.instance_id}",
            )

        head = _run_git(
            ["-C", str(workspace), "rev-parse", "HEAD"],
            error_prefix="Failed to resolve prepared worktree HEAD",
        ).stdout.strip()
        expected = _run_git(
            [
                "--git-dir",
                str(mirror),
                "rev-parse",
                f"{instance.base_commit}^{{commit}}",
            ],
            error_prefix="Failed to resolve requested base_commit",
        ).stdout.strip()
        if head != expected:
            raise RuntimeError(
                f"Prepared worktree HEAD mismatch: expected {expected}, found {head}"
            )
        status = _run_git(
            ["-C", str(workspace), "status", "--porcelain"],
            error_prefix="Failed to inspect prepared worktree",
        ).stdout
        if status.strip():
            raise RuntimeError("Prepared SWE-bench worktree is not clean")

        manifest = WorkspaceManifest(
            instance_id=instance.instance_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            resolved_head=head,
            workspace=str(workspace),
            mirror_path=str(mirror),
        )
        atomic_write_json(instance_dir / "workspace-manifest.json", manifest)
        return manifest

    def cleanup(self, manifest: WorkspaceManifest) -> None:
        """Remove only the instance worktree; retain the shared mirror cache."""

        self._remove_worktree(Path(manifest.mirror_path), Path(manifest.workspace))

    @staticmethod
    def _remove_worktree(mirror: Path, workspace: Path) -> None:
        if workspace.exists():
            result = subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(mirror),
                    "worktree",
                    "remove",
                    "--force",
                    str(workspace),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode != 0:
                shutil.rmtree(workspace, ignore_errors=True)
        subprocess.run(
            ["git", "--git-dir", str(mirror), "worktree", "prune"],
            capture_output=True,
            check=False,
        )


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
