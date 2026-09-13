"""Docker-backed execution for the existing bounded ``run_command`` tool."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

from minicode_harness.policy import check_command_allowed, render_argv
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.tools import CommandRunResult


_TESTBED_PATH = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_HOST_WORKSPACE = "/workspace"
_PATCH_PATH = "/minicode/changes.patch"
_SYNC_AND_EXEC = """
import json
import os
import shutil
import subprocess
import sys

patch_path = sys.argv[1]
untracked_paths = json.loads(sys.argv[2])
command = sys.argv[3:]
if os.path.getsize(patch_path):
    subprocess.run(
        ["git", "apply", "--binary", patch_path],
        cwd="/testbed",
        check=True,
    )
for relative_path in untracked_paths:
    source = os.path.join("/workspace", relative_path)
    target = os.path.join("/testbed", relative_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.islink(source):
        if os.path.lexists(target):
            os.unlink(target)
        os.symlink(os.readlink(source), target)
    else:
        shutil.copy2(source, target)
os.chdir("/testbed")
os.execvpe(command[0], command, os.environ)
""".strip()


class SweBenchDockerCommandExecutor:
    """Execute policy-approved commands inside a fixed container image."""

    sandboxed = True

    def __init__(
        self,
        *,
        image: str,
        docker_binary: str = "docker",
        container_workspace: str = "/testbed",
        network_disabled: bool = True,
        read_only_root: bool = False,
    ) -> None:
        if not image.strip():
            raise ValueError("Docker image must not be empty")
        self.image = image.strip()
        self.docker_binary = docker_binary
        self.container_workspace = container_workspace
        self.network_disabled = network_disabled
        self.read_only_root = read_only_root

    def execute(
        self,
        workspace: Path | str,
        argv: list[str],
        timeout_seconds: int,
        cancellation_token: CancellationToken | None = None,
        *,
        approval_granted: bool = False,
    ) -> CommandRunResult:
        """Run a policy-approved command with no network or extra privileges."""

        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
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

        patch_bytes, untracked_paths = _workspace_overlay(root)
        with TemporaryDirectory(prefix="minicode-swebench-command-") as temp_dir:
            patch_path = Path(temp_dir) / "changes.patch"
            patch_path.write_bytes(patch_bytes)
            arguments = [self.docker_binary, "run", "--rm"]
            if self.network_disabled:
                arguments.extend(["--network", "none"])
            if self.read_only_root:
                arguments.append("--read-only")
            arguments.extend(
                [
                    "--mount",
                    f"type=bind,source={root},target={_HOST_WORKSPACE},readonly",
                    "--mount",
                    f"type=bind,source={patch_path},target={_PATCH_PATH},readonly",
                    "--workdir",
                    self.container_workspace,
                    "--env",
                    f"PATH={_TESTBED_PATH}",
                    "--env",
                    f"PYTHONPATH={self.container_workspace}",
                    self.image,
                    "python",
                    "-c",
                    _SYNC_AND_EXEC,
                    _PATCH_PATH,
                    json.dumps(untracked_paths),
                    *normalized_argv,
                ]
            )
            return _run_process(
                arguments=arguments,
                argv=normalized_argv,
                command=command,
                timeout_seconds=timeout_seconds,
                cancellation_token=cancellation_token,
                allowlist_rule=policy_result.rule or "",
            )


def _workspace_overlay(root: Path) -> tuple[bytes, list[str]]:
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0:
        raise RuntimeError(_decode(diff.stderr) or "Failed to build workspace patch")
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
    )
    if untracked.returncode != 0:
        raise RuntimeError(_decode(untracked.stderr) or "Failed to list untracked files")
    paths = [
        value
        for value in _decode(untracked.stdout).split("\0")
        if value
    ]
    for value in paths:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"Unsafe untracked workspace path: {value}")
    return diff.stdout, paths


def _run_process(
    *,
    arguments: list[str],
    argv: list[str],
    command: str,
    timeout_seconds: int,
    cancellation_token: CancellationToken | None,
    allowlist_rule: str,
) -> CommandRunResult:
    started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return _result(
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
            stdout, stderr = _terminate(process)
            return _result(
                argv=argv,
                command=command,
                returncode=130,
                stdout=_decode(stdout),
                stderr=_decode(stderr) or "Command cancelled.",
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                allowlist_rule=allowlist_rule,
                cancelled=True,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stdout, stderr = _terminate(process)
            return _result(
                argv=argv,
                command=command,
                returncode=124,
                stdout=_decode(stdout),
                stderr=_decode(stderr),
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                allowlist_rule=allowlist_rule,
                timed_out=True,
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
        return _result(
            argv=argv,
            command=command,
            returncode=int(process.returncode or 0),
            stdout=_decode(stdout),
            stderr=_decode(stderr),
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            allowlist_rule=allowlist_rule,
        )


def _result(
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


def _terminate(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    if process.poll() is None:
        process.terminate()
    try:
        return process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def _decode(value: bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace")
