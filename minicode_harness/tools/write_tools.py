"""Write and command workspace tools."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import difflib
import hashlib
import locale
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from pydantic import BaseModel

from minicode_harness.policy import CommandRule, check_command_allowed, render_argv
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.workspace import WorkspaceGuard


DEFAULT_COMMAND_TIMEOUT_SECONDS = 120
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 1.0
_WINDOWS_TREE_KILL_TIMEOUT_SECONDS = 2.0


class StaleWriteError(RuntimeError):
    """Existing file changed or was not read before a whole-file replacement."""


class PatchApplyResult(BaseModel):
    """Result returned by apply_patch."""

    files: list[str]
    stdout: str = ""
    stderr: str = ""


class WriteFileResult(BaseModel):
    """Result returned by write_file."""

    path: str
    bytes_written: int
    created: bool
    overwritten: bool = False
    changed: bool = True


class WriteFilePreview(BaseModel):
    """Deterministic preview for a new file or explicit full replacement."""

    path: str
    created: bool
    overwrite: bool
    old_bytes: int
    new_bytes: int
    changed_lines: int
    diff: str = ""


class EditFileResult(BaseModel):
    """Result returned by edit_file."""

    path: str
    replacements: int
    bytes_written: int
    changed: bool = True


class EditFilePreview(BaseModel):
    """Deterministic preview for one exact edit."""

    path: str
    matches: int
    old_chars: int
    new_chars: int
    diff: str = ""


class CommandRunResult(BaseModel):
    """Result returned by run_command."""

    argv: list[str]
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timeout_seconds: int
    allowlist_rule: str
    resolve_duration_ms: int | None = None
    spawn_duration_ms: int | None = None
    execute_duration_ms: int | None = None
    timed_out: bool = False
    cancelled: bool = False

    @property
    def lifecycle_status(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.timed_out:
            return "timed_out"
        return "completed" if self.returncode == 0 else "failed"


def apply_patch(workspace: Path | str, patch: str) -> PatchApplyResult:
    """Apply a unified diff after validating touched paths."""

    if not patch.strip():
        raise ValueError("patch must not be empty.")
    guard = WorkspaceGuard(workspace)
    files = extract_patch_paths(patch)
    if not files:
        raise ValueError("patch did not contain any file paths.")
    for file_path in files:
        guard.resolve(file_path)

    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=guard.root,
        input=patch,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(f"Patch validation failed: {check.stderr.strip()}")

    applied = subprocess.run(
        ["git", "apply", "-"],
        cwd=guard.root,
        input=patch,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if applied.returncode != 0:
        raise RuntimeError(f"Patch application failed: {applied.stderr.strip()}")

    return PatchApplyResult(files=files, stdout=applied.stdout, stderr=applied.stderr)


def preview_write_file(
    workspace: Path | str,
    path: Path | str,
    content: str,
    *,
    overwrite: bool = False,
) -> WriteFilePreview:
    """Preview creation or an explicitly requested whole-file replacement."""

    guard = WorkspaceGuard(workspace)
    file_path = guard.resolve(path)
    display_path = guard.relative_path(file_path)
    if file_path.exists() and file_path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {display_path}")

    created = not file_path.exists()
    if not created and not overwrite:
        raise FileExistsError(
            "Target file already exists. Use edit for a focused change, "
            "apply_patch for complex changes, or set overwrite=true only when "
            "replacing the entire file is intentional."
        )

    old_content = file_path.read_text(encoding="utf-8") if not created else ""
    old_bytes = len(old_content.encode("utf-8"))
    new_bytes = len(content.encode("utf-8"))
    diff = ""
    if created or old_content != content:
        diff = "\n".join(
            difflib.unified_diff(
                old_content.splitlines(),
                content.splitlines(),
                fromfile="/dev/null" if created else f"a/{display_path}",
                tofile=f"b/{display_path}",
                lineterm="",
                n=3,
            )
        )
    changed_lines = sum(
        1
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    return WriteFilePreview(
        path=display_path,
        created=created,
        overwrite=overwrite,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        changed_lines=changed_lines,
        diff=diff,
    )


def write_file(
    workspace: Path | str,
    path: Path | str,
    content: str,
    *,
    overwrite: bool = False,
    expected_sha256: str | None = None,
) -> WriteFileResult:
    """Create a UTF-8 file or replace one exact previously observed version."""

    preview = preview_write_file(
        workspace,
        path,
        content,
        overwrite=overwrite,
    )
    guard = WorkspaceGuard(workspace)
    file_path = guard.resolve(path)
    encoded_content = content.encode("utf-8")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if preview.created:
        try:
            with file_path.open("xb") as handle:
                handle.write(encoded_content)
        except FileExistsError as exc:
            raise StaleWriteError(
                "Target file appeared after preview; read it before retrying."
            ) from exc
        return WriteFileResult(
            path=preview.path,
            bytes_written=len(encoded_content),
            created=True,
            overwritten=False,
            changed=True,
        )

    current_bytes = file_path.read_bytes()
    current_sha256 = hashlib.sha256(current_bytes).hexdigest()
    if expected_sha256 is not None and current_sha256 != expected_sha256:
        raise StaleWriteError(
            "Target file changed after it was read; read it again before overwriting."
        )
    changed = current_bytes != encoded_content
    if changed:
        _atomic_replace_file(
            file_path,
            encoded_content,
            expected_sha256=expected_sha256,
        )
    return WriteFileResult(
        path=preview.path,
        bytes_written=len(encoded_content),
        created=False,
        overwritten=True,
        changed=changed,
    )


def _atomic_replace_file(
    file_path: Path,
    content: bytes,
    *,
    expected_sha256: str | None,
) -> None:
    """Best-effort compare-before-replace using a same-directory temporary file."""

    mode = file_path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.minicode-",
        suffix=".tmp",
        dir=file_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        if expected_sha256 is not None:
            before_replace_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if before_replace_sha256 != expected_sha256:
                raise StaleWriteError(
                    "Target file changed while the replacement was being prepared; read it again."
                )
        os.replace(temporary_path, file_path)
        expected_result = hashlib.sha256(content).hexdigest()
        if hashlib.sha256(file_path.read_bytes()).hexdigest() != expected_result:
            raise RuntimeError("Atomic file replacement verification failed.")
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def preview_edit_file(
    workspace: Path | str,
    path: Path | str,
    old_text: str,
    new_text: str,
) -> EditFilePreview:
    """Preview one exact edit without modifying the workspace."""

    if not old_text:
        raise ValueError("old_text must not be empty.")
    guard = WorkspaceGuard(workspace)
    file_path = guard.resolve(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {guard.relative_path(file_path)}")
    content = file_path.read_text(encoding="utf-8")
    occurrences = content.count(old_text)
    updated = content.replace(old_text, new_text, 1) if occurrences == 1 else content
    diff = ""
    if occurrences == 1 and updated != content:
        diff = "\n".join(
            difflib.unified_diff(
                content.splitlines(),
                updated.splitlines(),
                fromfile=f"a/{guard.relative_path(file_path)}",
                tofile=f"b/{guard.relative_path(file_path)}",
                lineterm="",
                n=3,
            )
        )
    return EditFilePreview(
        path=guard.relative_path(file_path),
        matches=occurrences,
        old_chars=len(old_text),
        new_chars=len(new_text),
        diff=diff,
    )


def edit_file(
    workspace: Path | str,
    path: Path | str,
    old_text: str,
    new_text: str,
) -> EditFileResult:
    """Replace one uniquely matching text block in a UTF-8 workspace file."""

    preview = preview_edit_file(workspace, path, old_text, new_text)
    if preview.matches == 0:
        raise ValueError("old_text was not found in the target file.")
    if preview.matches > 1:
        raise ValueError(
            f"old_text matched {preview.matches} locations; provide a more specific unique block."
        )
    guard = WorkspaceGuard(workspace)
    file_path = guard.resolve(path)
    content = file_path.read_text(encoding="utf-8")
    updated = content.replace(old_text, new_text, 1)
    encoded = updated.encode("utf-8")
    changed = updated != content
    if changed:
        file_path.write_bytes(encoded)
    return EditFileResult(
        path=guard.relative_path(file_path),
        replacements=1,
        bytes_written=len(encoded),
        changed=changed,
    )


def run_command(
    workspace: Path | str,
    argv: list[str],
    *,
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    cancellation_token: CancellationToken | None = None,
    approval_granted: bool = False,
    command_rules: Sequence[CommandRule] = (),
) -> CommandRunResult:
    """Run a policy-approved command with timeout and cooperative cancellation."""

    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1.")
    guard = WorkspaceGuard(workspace)
    policy_result = check_command_allowed(argv, sandboxed=False, rules=command_rules)
    if not policy_result.allowed:
        raise PermissionError(policy_result.reason)
    if policy_result.requires_approval and not approval_granted:
        raise PermissionError(policy_result.reason)

    normalized_argv = list(policy_result.argv)
    command = render_argv(normalized_argv)
    started_at = time.monotonic()
    resolve_started_at = time.monotonic()
    resolved_argv = _resolve_command_argv(normalized_argv)
    resolve_duration_ms = _elapsed_ms(resolve_started_at)
    spawn_started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            resolved_argv,
            cwd=guard.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_sanitized_command_environment(),
        )
    except FileNotFoundError as exc:
        return CommandRunResult(
            argv=normalized_argv,
            command=command,
            returncode=127,
            stdout="",
            stderr=str(exc),
            duration_seconds=time.monotonic() - started_at,
            timeout_seconds=timeout_seconds,
            allowlist_rule=policy_result.rule or "",
            resolve_duration_ms=resolve_duration_ms,
            spawn_duration_ms=_elapsed_ms(spawn_started_at),
        )

    spawn_duration_ms = _elapsed_ms(spawn_started_at)
    execute_started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    while True:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            stdout, stderr = _terminate_process(process)
            return CommandRunResult(
                argv=normalized_argv,
                command=command,
                returncode=130,
                stdout=_decode_command_output(stdout),
                stderr=_decode_command_output(stderr) or "Command cancelled.",
                duration_seconds=time.monotonic() - started_at,
                timeout_seconds=timeout_seconds,
                allowlist_rule=policy_result.rule or "",
                resolve_duration_ms=resolve_duration_ms,
                spawn_duration_ms=spawn_duration_ms,
                execute_duration_ms=_elapsed_ms(execute_started_at),
                cancelled=True,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stdout, stderr = _terminate_process(process)
            return CommandRunResult(
                argv=normalized_argv,
                command=command,
                returncode=124,
                stdout=_decode_command_output(stdout),
                stderr=_decode_command_output(stderr),
                duration_seconds=time.monotonic() - started_at,
                timeout_seconds=timeout_seconds,
                allowlist_rule=policy_result.rule or "",
                resolve_duration_ms=resolve_duration_ms,
                spawn_duration_ms=spawn_duration_ms,
                execute_duration_ms=_elapsed_ms(execute_started_at),
                timed_out=True,
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
        return CommandRunResult(
            argv=normalized_argv,
            command=command,
            returncode=process.returncode,
            stdout=_decode_command_output(stdout),
            stderr=_decode_command_output(stderr),
            duration_seconds=time.monotonic() - started_at,
            timeout_seconds=timeout_seconds,
            allowlist_rule=policy_result.rule or "",
            resolve_duration_ms=resolve_duration_ms,
            spawn_duration_ms=spawn_duration_ms,
            execute_duration_ms=_elapsed_ms(execute_started_at),
        )


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


def _sanitized_command_environment() -> dict[str, str]:
    """Return the host environment without common credential-bearing variables."""

    environment = dict(os.environ)
    protected_names = {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SSH_AUTH_SOCK",
    }
    sensitive_markers = (
        "API_KEY",
        "ACCESS_KEY",
        "AUTH_TOKEN",
        "CREDENTIAL",
        "PRIVATE_KEY",
        "PASSWORD",
        "SECRET",
    )
    sensitive_suffixes = ("_TOKEN", "_KEY", "_PASSWORD", "_SECRET", "_CREDENTIALS")
    for name in list(environment):
        upper = name.upper()
        if (
            upper in protected_names
            or any(marker in upper for marker in sensitive_markers)
            or upper.endswith(sensitive_suffixes)
        ):
            environment.pop(name, None)
    return environment


def _terminate_process(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    if os.name == "nt":
        _terminate_windows_process_tree(process)
    elif process.poll() is None:
        process.terminate()

    try:
        return process.communicate(timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as first_timeout:
        if process.poll() is None:
            process.kill()
        try:
            return process.communicate(timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as final_timeout:
            _close_process_pipes(process)
            try:
                process.wait(timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS)
            except (subprocess.TimeoutExpired, OSError):
                pass
            stdout = final_timeout.stdout or first_timeout.stdout or b""
            stderr = final_timeout.stderr or first_timeout.stderr or b""
            return stdout, stderr


def _terminate_windows_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    pid = getattr(process, "pid", None)
    if pid is None:
        process.terminate()
        return
    try:
        completed = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_WINDOWS_TREE_KILL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if (completed is None or completed.returncode != 0) and process.poll() is None:
        process.terminate()


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for pipe in (process.stdout, process.stderr):
        if pipe is None:
            continue
        try:
            pipe.close()
        except OSError:
            pass


def _decode_command_output(value: bytes | str | None) -> str:
    """Decode subprocess output without corrupting non-UTF-8 build diagnostics."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value

    # UTF-8 is unambiguous when strict decoding succeeds. If it does not,
    # prefer the host console encoding before broad fallbacks. Build tools on
    # Chinese Windows commonly emit CP936/GBK bytes; decoding those as UTF-8
    # destroys actionable diagnostics.
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        pass

    encodings = [locale.getpreferredencoding(False), "gb18030", "utf-8"]
    candidates: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(value.decode(encoding, errors="replace"))
    return min(candidates, key=lambda decoded: decoded.count("\ufffd"))


def _resolve_command_argv(argv: list[str]) -> list[str]:
    """Resolve the executable path before subprocess.run.

    On Windows, build-tool command shims are often exposed as ``*.cmd``.
    Passing the bare command name to subprocess.run with shell=False can fail
    with FileNotFoundError even when the command is available on PATH.
    """

    if not argv:
        return argv
    if argv[0].lower() in {"python", "python.exe", "python3", "python3.exe"}:
        return [sys.executable, *argv[1:]]
    executable = shutil.which(argv[0])
    if executable is None:
        return argv
    return [executable, *argv[1:]]


def extract_patch_paths(patch: str) -> list[str]:
    """Extract workspace-relative file paths from a unified diff."""

    paths: list[str] = []
    seen: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw_path = line[4:].split("\t", maxsplit=1)[0].strip()
        if raw_path == "/dev/null":
            continue
        normalized = _normalize_patch_path(raw_path)
        if normalized and normalized not in seen:
            paths.append(normalized)
            seen.add(normalized)
    return paths


def _normalize_patch_path(path: str) -> str:
    path = path.strip().strip('"')
    path = re.sub(r"^[ab]/", "", path)
    return path.replace("\\", "/")
