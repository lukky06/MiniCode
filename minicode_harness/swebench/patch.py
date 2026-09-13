"""Standard Git patch export and validation for SWE-bench predictions."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import subprocess
import tempfile

from .models import PatchResult
from .prediction import atomic_write_text


DEFAULT_EXCLUDED_PATTERNS = (
    ".git/**",
    "runs/**",
    "__pycache__/**",
    ".pytest_cache/**",
    "*.pyc",
    "coverage.xml",
    "htmlcov/**",
    "workspace-manifest.json",
    "trace.jsonl",
    "checkpoints/**",
    "artifacts/**",
)


class PatchExporter:
    """Export a deterministic binary-capable patch relative to ``HEAD``."""

    def __init__(
        self,
        *,
        max_patch_bytes: int = 5_000_000,
        excluded_patterns: tuple[str, ...] = DEFAULT_EXCLUDED_PATTERNS,
    ) -> None:
        if max_patch_bytes < 1:
            raise ValueError("max_patch_bytes must be positive")
        self.max_patch_bytes = max_patch_bytes
        self.excluded_patterns = excluded_patterns

    def export(
        self,
        workspace: Path | str,
        *,
        output_path: Path | str | None = None,
        expected_head: str | None = None,
    ) -> PatchResult:
        """Export, validate, and optionally persist the current workspace patch."""

        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Patch workspace does not exist: {root}")
        head = _git(root, ["rev-parse", "HEAD"]).stdout.strip()
        if expected_head is not None:
            resolved_expected = _git(
                root,
                ["rev-parse", f"{expected_head}^{{commit}}"],
            ).stdout.strip()
            if head != resolved_expected:
                raise RuntimeError(
                    f"Patch workspace HEAD mismatch: expected {resolved_expected}, found {head}"
                )

        untracked = _untracked_files(root)
        included_untracked = [
            path for path in untracked if not self._is_excluded(path)
        ]
        excluded = [path for path in untracked if self._is_excluded(path)]
        if included_untracked:
            _git(root, ["add", "--intent-to-add", "--", *included_untracked])

        pathspecs = ["."] + [
            f":(exclude){pattern}" for pattern in self.excluded_patterns
        ]
        process = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--binary",
                "--find-renames",
                "--no-ext-diff",
                "--full-index",
                "HEAD",
                "--",
                *pathspecs,
            ],
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            return PatchResult(
                status="invalid",
                excluded_paths=excluded,
                validation_error=detail or "git diff failed",
            )

        patch_bytes = process.stdout
        try:
            patch_text = patch_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            return PatchResult(
                status="invalid",
                bytes=len(patch_bytes),
                excluded_paths=excluded,
                validation_error=(
                    "Git patch is not valid UTF-8 and cannot be represented "
                    f"losslessly in predictions.jsonl: {exc}"
                ),
            )
        changed_files = _changed_files(root, pathspecs)
        if not patch_bytes:
            path = None
            if output_path is not None:
                path = str(atomic_write_text(output_path, ""))
            return PatchResult(
                status="no_patch",
                patch="",
                patch_path=path,
                bytes=0,
                changed_files=[],
                excluded_paths=excluded,
            )
        if len(patch_bytes) > self.max_patch_bytes:
            return PatchResult(
                status="invalid",
                patch=patch_text,
                bytes=len(patch_bytes),
                changed_files=changed_files,
                excluded_paths=excluded,
                validation_error=(
                    f"Patch exceeds maximum size of {self.max_patch_bytes} bytes"
                ),
            )

        validation_error = _validate_patch_in_clean_worktree(root, patch_bytes)
        if validation_error:
            return PatchResult(
                status="invalid",
                patch=patch_text,
                bytes=len(patch_bytes),
                changed_files=changed_files,
                excluded_paths=excluded,
                validation_error=validation_error,
            )

        path = None
        if output_path is not None:
            path = str(atomic_write_text(output_path, patch_text))
        return PatchResult(
            status="ok",
            patch=patch_text,
            patch_path=path,
            bytes=len(patch_bytes),
            changed_files=changed_files,
            excluded_paths=excluded,
        )

    def _is_excluded(self, path: str) -> bool:
        normalized = path.replace("\\", "/")
        return any(fnmatch(normalized, pattern) for pattern in self.excluded_patterns)


def _untracked_files(workspace: Path) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or "Failed to list untracked files"
        )
    return [
        item.decode("utf-8", errors="replace")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _changed_files(workspace: Path, pathspecs: list[str]) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "diff",
            "--name-only",
            "--find-renames",
            "-z",
            "HEAD",
            "--",
            *pathspecs,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or "Failed to list changed files"
        )
    return sorted(
        path.decode("utf-8", errors="replace").replace("\\", "/")
        for path in result.stdout.split(b"\0")
        if path
    )


def _validate_patch_in_clean_worktree(workspace: Path, patch: bytes) -> str | None:
    with tempfile.TemporaryDirectory(prefix="minicode-swebench-patch-") as temp_dir:
        validation_root = Path(temp_dir) / "workspace"
        add = subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "worktree",
                "add",
                "--detach",
                str(validation_root),
                "HEAD",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if add.returncode != 0:
            return add.stderr.strip() or "Failed to create patch validation worktree"
        try:
            check = subprocess.run(
                ["git", "-C", str(validation_root), "apply", "--check", "-"],
                input=patch,
                capture_output=True,
                check=False,
            )
            if check.returncode != 0:
                return (
                    check.stderr.decode("utf-8", errors="replace").strip()
                    or "git apply --check failed"
                )
            return None
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(workspace),
                    "worktree",
                    "remove",
                    "--force",
                    str(validation_root),
                ],
                capture_output=True,
                check=False,
            )


def _git(workspace: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git command failed: {detail}")
    return result
