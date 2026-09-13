"""Injectable execution boundary for the existing ``run_command`` tool."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Protocol
import unicodedata

from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.policy import check_command_allowed, render_argv

from .write_tools import (
    CommandRunResult,
    _decode_command_output,
    _sanitized_command_environment,
    run_command,
)


_CONTAINER_WORKSPACE = "/workspace"
_CONTAINER_TMPFS = "/tmp:rw,nosuid,nodev,noexec,size=64m"
_DOCKER_CLEANUP_TIMEOUT_SECONDS = 5
_DOCKER_STOP_TIMEOUT_SECONDS = 1


class SandboxMode(StrEnum):
    """Execution environment for policy-admitted commands."""

    LOCAL = "local"
    DOCKER = "docker"


class CommandExecutor(Protocol):
    """Execute one command already admitted by policy and approval."""

    sandboxed: bool

    def execute(
        self,
        workspace: Path | str,
        argv: list[str],
        timeout_seconds: int,
        cancellation_token: CancellationToken | None = None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        """Execute validated ``argv`` for ``workspace`` and return a normalized result."""


def create_command_executor(
    mode: SandboxMode | str = SandboxMode.LOCAL,
    *,
    image: str | None = None,
) -> CommandExecutor:
    """Create the selected command execution environment."""

    resolved = SandboxMode(mode)
    if resolved == SandboxMode.LOCAL:
        return LocalCommandExecutor()
    if not image:
        raise ValueError("Docker sandbox requires an image.")
    return DockerCommandExecutor(image=image)


class LocalCommandExecutor:
    """Default executor for policy-approved local workspace commands."""

    sandboxed = False

    def execute(
        self,
        workspace: Path | str,
        argv: list[str],
        timeout_seconds: int,
        cancellation_token: CancellationToken | None = None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        return run_command(
            workspace,
            argv,
            timeout_seconds=timeout_seconds,
            cancellation_token=cancellation_token,
            approval_granted=approval_granted,
        )


class DockerCommandExecutor:
    """Execute policy-approved commands in a constrained Docker container.

    This is an opt-in executor.  It deliberately owns only the OS/container
    boundary; command policy and approval are checked again immediately before
    starting Docker so callers cannot accidentally bypass either control.
    """

    sandboxed = True

    def __init__(self, *, image: str, docker_binary: str = "docker") -> None:
        if not image:
            raise ValueError("Docker image must not be empty")
        if image != image.strip():
            raise ValueError("Docker image must not contain whitespace")
        if any(char.isspace() or _is_unicode_control_or_format(char) for char in image):
            raise ValueError("Docker image must not contain whitespace or control characters")
        if image.startswith("-"):
            raise ValueError("Docker image must not start with '-'")
        if not docker_binary:
            raise ValueError("Docker executable must not be empty")
        if docker_binary != docker_binary.strip():
            raise ValueError("Docker executable must not contain whitespace")
        if any(
            (char.isspace() and char != " ") or _is_unicode_control_or_format(char)
            for char in docker_binary
        ):
            raise ValueError("Docker executable must not contain whitespace or control characters")
        self.image = image
        self.docker_binary = docker_binary

    def execute(
        self,
        workspace: Path | str,
        argv: list[str],
        timeout_seconds: int,
        cancellation_token: CancellationToken | None = None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        """Run one command with workspace-only, network-disabled isolation."""

        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")

        # Re-check both gates at the execution boundary.  ToolRegistry already
        # performs admission, but this executor must remain safe when called
        # directly or through another runtime integration.
        policy_result = check_command_allowed(argv)
        if not policy_result.allowed:
            raise PermissionError(policy_result.reason or "Command is not allowed")
        if policy_result.requires_approval and not approval_granted:
            raise PermissionError(policy_result.reason or "Command requires approval")

        normalized_argv = list(policy_result.argv)
        command = render_argv(normalized_argv)
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Command workspace does not exist: {root}")

        with TemporaryDirectory(prefix="minicode-docker-") as temp_dir:
            cidfile = Path(temp_dir) / "container.cid"
            docker_argv = [
                self.docker_binary,
                "run",
                "--rm",
                "--cidfile",
                str(cidfile),
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                _CONTAINER_TMPFS,
                "--mount",
                f"type=bind,source={root},target={_CONTAINER_WORKSPACE}",
                "--workdir",
                _CONTAINER_WORKSPACE,
                "--entrypoint",
                "",
                self.image,
                *normalized_argv,
            ]
            return _run_docker_process(
                arguments=docker_argv,
                argv=normalized_argv,
                command=command,
                timeout_seconds=timeout_seconds,
                cancellation_token=cancellation_token,
                allowlist_rule=policy_result.rule or "",
                docker_binary=self.docker_binary,
                cidfile=cidfile,
            )


def _run_docker_process(
    *,
    arguments: list[str],
    argv: list[str],
    command: str,
    timeout_seconds: int,
    cancellation_token: CancellationToken | None,
    allowlist_rule: str,
    docker_binary: str,
    cidfile: Path,
) -> CommandRunResult:
    started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            arguments,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_sanitized_command_environment(),
        )
    except FileNotFoundError as exc:
        return _docker_result(
            argv=argv,
            command=command,
            returncode=127,
            stderr=str(exc),
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            allowlist_rule=allowlist_rule,
        )

    deadline = started_at + timeout_seconds
    while True:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            stdout, stderr = _terminate_docker_process(process)
            cancellation_stderr = _decode_command_output(stderr) or "Command cancelled."
            stderr = _append_cleanup_diagnostic(
                stderr=cancellation_stderr,
                cleanup_diagnostic=_cleanup_docker_container(
                    docker_binary=docker_binary,
                    cidfile=cidfile,
                ),
            )
            return _docker_result(
                argv=argv,
                command=command,
                returncode=130,
                stdout=_decode_command_output(stdout),
                stderr=stderr,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                allowlist_rule=allowlist_rule,
                cancelled=True,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stdout, stderr = _terminate_docker_process(process)
            stderr = _append_cleanup_diagnostic(
                stderr=_decode_command_output(stderr),
                cleanup_diagnostic=_cleanup_docker_container(
                    docker_binary=docker_binary,
                    cidfile=cidfile,
                ),
            )
            return _docker_result(
                argv=argv,
                command=command,
                returncode=124,
                stdout=_decode_command_output(stdout),
                stderr=stderr,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                allowlist_rule=allowlist_rule,
                timed_out=True,
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
        return _docker_result(
            argv=argv,
            command=command,
            returncode=int(process.returncode or 0),
            stdout=_decode_command_output(stdout),
            stderr=_decode_command_output(stderr),
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            allowlist_rule=allowlist_rule,
        )


def _docker_result(
    *,
    argv: list[str],
    command: str,
    returncode: int,
    started_at: float,
    timeout_seconds: int,
    allowlist_rule: str,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
    cancelled: bool = False,
) -> CommandRunResult:
    return CommandRunResult(
        argv=argv,
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=time.monotonic() - started_at,
        timeout_seconds=timeout_seconds,
        allowlist_rule=allowlist_rule,
        timed_out=timed_out,
        cancelled=cancelled,
    )


def _terminate_docker_process(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    if process.poll() is None:
        process.terminate()
    try:
        return process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def _cleanup_docker_container(*, docker_binary: str, cidfile: Path) -> str:
    """Stop and force-remove the container after client-side interruption."""

    try:
        container_id = cidfile.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return (
            "Docker container ID was not available; daemon-side cleanup was not "
            "attempted (container state is unknown)."
        )
    except (OSError, UnicodeDecodeError) as exc:
        return f"Docker container ID could not be read; container state is unknown: {exc}"

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container_id):
        return "Docker container ID was invalid; daemon-side cleanup was not attempted (container state is unknown)."

    stop = _run_docker_cleanup_command(
        [docker_binary, "stop", "--time", str(_DOCKER_STOP_TIMEOUT_SECONDS), container_id]
    )
    remove = _run_docker_cleanup_command([docker_binary, "rm", "--force", container_id])
    failures: list[str] = []
    if stop:
        failures.append(f"stop: {stop}")
    if remove:
        failures.append(f"remove: {remove}")
    if failures:
        return (
            "Docker container cleanup failed; container may still be running: "
            + "; ".join(failures)
        )
    return "Docker container cleanup completed."


def _run_docker_cleanup_command(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            shell=False,
            capture_output=True,
            env=_sanitized_command_environment(),
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {_DOCKER_CLEANUP_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return str(exc)
    if completed.returncode == 0:
        return None
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    detail = stderr or stdout or f"exit code {completed.returncode}"
    if "no such container" in detail.lower():
        return None
    return detail


def _append_cleanup_diagnostic(*, stderr: str, cleanup_diagnostic: str) -> str:
    if stderr and cleanup_diagnostic:
        return f"{stderr}\n{cleanup_diagnostic}"
    return cleanup_diagnostic or stderr


def _is_unicode_control_or_format(value: str) -> bool:
    return unicodedata.category(value) in {"Cc", "Cf", "Cs", "Cn"}
