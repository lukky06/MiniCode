from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import pytest

from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.tools import (
    DockerCommandExecutor,
    LocalCommandExecutor,
    SandboxMode,
    create_command_executor,
)
import minicode_harness.tools.command_executor as command_executor_module


def test_command_executor_factory_keeps_sandbox_choice_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_checks: list[list[str]] = []

    def docker_ready(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        docker_checks.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "ready\n", "")

    monkeypatch.setattr(command_executor_module.subprocess, "run", docker_ready)

    local = create_command_executor(SandboxMode.LOCAL)
    assert isinstance(local, LocalCommandExecutor)
    assert local.sandboxed is False
    assert docker_checks == []

    docker = create_command_executor(
        SandboxMode.DOCKER,
        image="python:3.11-slim",
    )
    assert isinstance(docker, DockerCommandExecutor)
    assert docker.sandboxed is True
    assert docker_checks == [
        ["docker", "version", "--format", "{{.Server.Version}}"],
        ["docker", "image", "inspect", "python:3.11-slim"],
    ]


def test_docker_sandbox_requires_an_explicit_image() -> None:
    with pytest.raises(ValueError, match="Docker sandbox requires an image"):
        create_command_executor(SandboxMode.DOCKER)


def test_docker_factory_fails_fast_when_daemon_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def daemon_down(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 1, "", "daemon unavailable")

    monkeypatch.setattr(command_executor_module.subprocess, "run", daemon_down)

    with pytest.raises(RuntimeError, match="Start Docker Desktop.*--sandbox local"):
        create_command_executor(SandboxMode.DOCKER, image="python:3.11-slim")


def test_docker_factory_refuses_missing_local_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def image_missing(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            arguments,
            0 if calls == 1 else 1,
            "29.2.1\n" if calls == 1 else "",
            "" if calls == 1 else "No such image",
        )

    monkeypatch.setattr(command_executor_module.subprocess, "run", image_missing)

    with pytest.raises(RuntimeError, match="not available locally.*--sandbox local"):
        create_command_executor(SandboxMode.DOCKER, image="python:3.11-slim")


class FakeDockerProcess:
    returncode = 0

    def __init__(self, stdout: bytes = b"ok\n", stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.terminated = False
        self.terminate_calls = 0
        self.kill_calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        _ = timeout
        return self.stdout, self.stderr

    def poll(self) -> int | None:
        return self.returncode if self.terminated else None

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.terminated = True

    def kill(self) -> None:
        self.kill_calls += 1
        self.terminated = True


@pytest.mark.parametrize(
    "image",
    ["", "  ", "-bad", "bad image", "bad\nimage", "bad\timage", "bad\x1fimage", "bad\x80image", "bad\u200bimage"],
)
def test_docker_executor_rejects_unsafe_image_values(image: str) -> None:
    with pytest.raises(ValueError, match="image"):
        DockerCommandExecutor(image=image)


@pytest.mark.parametrize("docker_binary", [" docker", "docker ", "docker\tbin", "docker\nbin", "docker\x80bin"])
def test_docker_executor_rejects_unsafe_docker_binary_values(docker_binary: str) -> None:
    with pytest.raises(ValueError, match="executable"):
        DockerCommandExecutor(image="python:3.11-slim", docker_binary=docker_binary)


def test_docker_executor_keeps_windows_docker_binary_with_spaces_as_one_argv0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    process = FakeDockerProcess()

    def fake_popen(arguments: list[str], **kwargs: Any) -> FakeDockerProcess:
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return process

    docker_binary = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    monkeypatch.setattr(command_executor_module.subprocess, "Popen", fake_popen)

    DockerCommandExecutor(image="python:3.11-slim", docker_binary=docker_binary).execute(
        tmp_path,
        ["python", "--version"],
        10,
    )

    assert captured["arguments"][0] == docker_binary
    assert captured["arguments"].count(docker_binary) == 1
    assert captured["kwargs"]["shell"] is False


def test_docker_executor_rechecks_policy_and_approval_before_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_popen(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Docker must not start for rejected commands")

    monkeypatch.setattr(command_executor_module.subprocess, "Popen", unexpected_popen)
    executor = DockerCommandExecutor(image="python:3.11-slim")

    with pytest.raises(PermissionError):
        executor.execute(tmp_path, ["bash", "-lc", "whoami"], 10, approval_granted=True)
    with pytest.raises(PermissionError, match="side-effect free|approval"):
        executor.execute(tmp_path, ["python", "script.py"], 10)


def test_docker_executor_builds_workspace_only_restricted_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    process = FakeDockerProcess(stdout="通过\n".encode("utf-8"))

    def fake_popen(arguments: list[str], **kwargs: Any) -> FakeDockerProcess:
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(command_executor_module.subprocess, "Popen", fake_popen)
    executor = DockerCommandExecutor(image="python:3.11-slim", docker_binary="docker")

    result = executor.execute(tmp_path, ["python", "-m", "pytest", "-q"], 10)

    arguments = captured["arguments"]
    assert result.stdout == "通过\n"
    assert result.argv == ["python", "-m", "pytest", "-q"]
    assert result.allowlist_rule == "python -m pytest [focused args]"
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert arguments[:5] == ["docker", "run", "--rm", "--pull", "never"]
    assert arguments[5] == "--cidfile"
    assert Path(arguments[6]).name == "container.cid"
    assert arguments[7:10] == ["--network", "none", "--read-only"]
    assert arguments[10:18] == [
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--mount",
        f"type=bind,source={tmp_path.resolve()},target=/workspace",
    ]
    image_index = arguments.index("python:3.11-slim")
    assert arguments[image_index - 4 : image_index + 1] == [
        "--workdir",
        "/workspace",
        "--entrypoint",
        "",
        "python:3.11-slim",
    ]
    assert arguments[image_index + 1 :] == ["python", "-m", "pytest", "-q"]
    bind_mounts = [value for value in arguments if value.startswith("type=bind,")]
    assert bind_mounts == [
        f"type=bind,source={tmp_path.resolve()},target=/workspace"
    ]
    assert not any(value.startswith("--privileged") for value in arguments)
    assert not any(value.startswith("--cap-add") for value in arguments)


def test_docker_executable_missing_returns_normalized_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_popen(*args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        raise FileNotFoundError("docker executable not found")

    monkeypatch.setattr(command_executor_module.subprocess, "Popen", missing_popen)
    result = DockerCommandExecutor(image="python:3.11-slim").execute(
        tmp_path,
        ["python", "--version"],
        10,
    )

    assert result.returncode == 127
    assert result.stderr == "docker executable not found"
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.lifecycle_status == "failed"


def test_docker_executor_preserves_timeout_and_cancellation_result_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess(FakeDockerProcess):
        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            _ = timeout
            if not self.terminated:
                raise subprocess.TimeoutExpired("docker", timeout or 0)
            return b"partial", b""

    process = HangingProcess()
    monkeypatch.setattr(
        command_executor_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    clock = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(command_executor_module.time, "monotonic", lambda: next(clock))

    timed_out = DockerCommandExecutor(image="python:3.11-slim").execute(
        tmp_path,
        ["python", "--version"],
        1,
    )
    assert timed_out.returncode == 124
    assert timed_out.timed_out is True
    assert timed_out.cancelled is False
    assert timed_out.lifecycle_status == "timed_out"
    assert process.terminate_calls == 1
    assert process.kill_calls == 0

    cancelled_process = FakeDockerProcess()
    cancel_clock = iter([3.0, 3.0, 3.0])
    monkeypatch.setattr(
        command_executor_module.time,
        "monotonic",
        lambda: next(cancel_clock),
    )
    monkeypatch.setattr(
        command_executor_module.subprocess,
        "Popen",
        lambda *args, **kwargs: cancelled_process,
    )
    token = CancellationToken()
    token.cancel()
    cancelled = DockerCommandExecutor(image="python:3.11-slim").execute(
        tmp_path,
        ["python", "--version"],
        10,
        token,
    )
    assert cancelled.returncode == 130
    assert cancelled.stderr.startswith("Command cancelled.")
    assert cancelled.cancelled is True
    assert cancelled.timed_out is False
    assert cancelled.lifecycle_status == "cancelled"
    assert cancelled_process.terminate_calls == 1
    assert cancelled_process.kill_calls == 0


def test_docker_timeout_stops_and_removes_recorded_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess(FakeDockerProcess):
        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            _ = timeout
            if not self.terminated:
                raise subprocess.TimeoutExpired("docker", timeout or 0)
            return b"partial", b""

    process = HangingProcess()
    cleanup_calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_popen(arguments: list[str], **kwargs: Any) -> HangingProcess:
        cidfile = Path(arguments[arguments.index("--cidfile") + 1])
        cidfile.write_text("container123\n", encoding="ascii")
        return process

    def fake_run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cleanup_calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(command_executor_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(command_executor_module.subprocess, "run", fake_run)
    clock = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(command_executor_module.time, "monotonic", lambda: next(clock))

    result = DockerCommandExecutor(image="python:3.11-slim").execute(
        tmp_path,
        ["python", "--version"],
        1,
    )

    assert result.timed_out is True
    assert "Docker container cleanup completed." in result.stderr
    assert [call[0][:4] for call in cleanup_calls] == [
        ["docker", "stop", "--time", "1"],
        ["docker", "rm", "--force", "container123"],
    ]
    assert cleanup_calls[0][0][-1] == "container123"
    assert cleanup_calls[1][0][-1] == "container123"
    assert all(call[1]["shell"] is False for call in cleanup_calls)


def test_docker_cancellation_reports_missing_cidfile_without_cleanup_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeDockerProcess()
    cleanup_calls: list[list[str]] = []
    monkeypatch.setattr(
        command_executor_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        command_executor_module.subprocess,
        "run",
        lambda arguments, **kwargs: cleanup_calls.append(arguments),
    )
    clock = iter([0.0, 0.0])
    monkeypatch.setattr(command_executor_module.time, "monotonic", lambda: next(clock))
    token = CancellationToken()
    token.cancel()

    result = DockerCommandExecutor(image="python:3.11-slim").execute(
        tmp_path,
        ["python", "--version"],
        10,
        token,
    )

    assert result.cancelled is True
    assert "container ID was not available" in result.stderr
    assert "state is unknown" in result.stderr
    assert cleanup_calls == []
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


def test_docker_timeout_reports_cleanup_failure_and_container_risk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess(FakeDockerProcess):
        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            _ = timeout
            if not self.terminated:
                raise subprocess.TimeoutExpired("docker", timeout or 0)
            return b"", b""

    process = HangingProcess()

    def fake_popen(arguments: list[str], **kwargs: Any) -> HangingProcess:
        cidfile = Path(arguments[arguments.index("--cidfile") + 1])
        cidfile.write_text("container123", encoding="ascii")
        return process

    def failed_run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 125, "", "daemon unavailable")

    monkeypatch.setattr(command_executor_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(command_executor_module.subprocess, "run", failed_run)
    clock = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(command_executor_module.time, "monotonic", lambda: next(clock))

    result = DockerCommandExecutor(image="python:3.11-slim").execute(
        tmp_path,
        ["python", "--version"],
        1,
    )

    assert result.timed_out is True
    assert "Docker container cleanup failed" in result.stderr
    assert "container may still be running" in result.stderr
    assert "stop: daemon unavailable" in result.stderr
    assert "remove: daemon unavailable" in result.stderr


def test_docker_client_cleanup_never_uses_unbounded_communicate() -> None:
    class Pipe:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class StuckProcess:
        returncode = None

        def __init__(self) -> None:
            self.stdout = Pipe()
            self.stderr = Pipe()
            self.timeouts: list[float | None] = []
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_timeouts: list[float | None] = []

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            self.timeouts.append(timeout)
            raise subprocess.TimeoutExpired(
                "docker",
                timeout or 0,
                output=b"partial",
                stderr=b"diagnostic",
            )

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            raise subprocess.TimeoutExpired("docker", timeout or 0)

    process = StuckProcess()

    stdout, stderr = command_executor_module._terminate_docker_process(process)

    assert stdout == b"partial"
    assert stderr == b"diagnostic"
    assert process.timeouts == [1.0, 1.0]
    assert process.wait_timeouts == [1.0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_docker_timeout_kills_client_when_terminate_does_not_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class KillRequiredProcess(FakeDockerProcess):
        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            _ = timeout
            if self.kill_calls == 0:
                raise subprocess.TimeoutExpired("docker", timeout or 0)
            return b"after kill", b""

        def poll(self) -> int | None:
            return self.returncode if self.kill_calls else None

    process = KillRequiredProcess()
    monkeypatch.setattr(
        command_executor_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    clock = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(command_executor_module.time, "monotonic", lambda: next(clock))

    result = DockerCommandExecutor(image="python:3.11-slim").execute(
        tmp_path,
        ["python", "--version"],
        1,
    )

    assert result.timed_out is True
    assert result.stdout == "after kill"
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
