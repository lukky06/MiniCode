from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

import pytest

from minicode_harness.policy import (
    CommandCategory,
    CommandPolicyAction,
    CommandRule,
    CommandRuleDecision,
    RiskLevel,
    check_command_allowed,
    classify_argv,
    resolve_command_session_grant,
)
from minicode_harness.tools import (
    ToolRegistry,
    apply_patch,
    extract_patch_paths,
    run_command,
    write_file,
)
from minicode_harness.tools.write_tools import (
    _decode_command_output,
    _resolve_command_argv,
    _sanitized_command_environment,
    _terminate_process,
)
import minicode_harness.tools.write_tools as write_tools_module
from minicode_harness.workspace import WorkspaceAccessError


def _argv(command: str) -> list[str]:
    if "\n" in command or "\r" in command:
        return [command]
    return shlex.split(command)


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest -q tests/test_api.py",
        "python -m compileall -q src/app.py",
        "node --check src/app.js",
        "npm test",
        "pnpm test -- src/api.test.ts",
        "npm run lint",
        "go test ./...",
        "cargo test parser",
        "dotnet build app.sln",
        "mvn -q -DskipTests compile",
        "./gradlew test",
        "bundle exec rspec spec/service_spec.rb",
        "mix test test/service_test.exs",
        "swift test",
        "make lint",
        "ctest --test-dir build",
        "cmake --build build",
    ],
)
def test_sandbox_default_accepts_development_commands_without_family_allowlists(command: str) -> None:
    result = check_command_allowed(_argv(command), sandboxed=True)

    assert result.action == CommandPolicyAction.ALLOW
    assert result.allowed is True
    assert result.requires_approval is False
    assert result.rule == "sandbox default"
    assert result.argv


def test_sandboxed_unknown_development_command_runs_without_tool_specific_allowlist() -> None:
    result = check_command_allowed(["mvn", "package", "-DskipTests"], sandboxed=True)

    assert result.action == CommandPolicyAction.ALLOW
    assert result.requires_approval is False
    assert result.rule == "sandbox default"


def test_local_unknown_command_defaults_to_approval() -> None:
    result = check_command_allowed(["custom-check", "--verify"])

    assert result.action == CommandPolicyAction.REQUIRE_APPROVAL
    assert result.requires_approval is True
    assert result.rule == "local execution default"


def test_declarative_prefix_rules_override_sandbox_default() -> None:
    ask = CommandRule(decision=CommandRuleDecision.ASK, prefix=("npm", "install"))
    deny = CommandRule(decision=CommandRuleDecision.DENY, prefix=("git", "push"))
    allow = CommandRule(decision=CommandRuleDecision.ALLOW, prefix=("custom-check",))

    assert check_command_allowed(["npm", "install", "left-pad"], sandboxed=True, rules=[ask]).action == CommandPolicyAction.REQUIRE_APPROVAL
    assert check_command_allowed(["git", "push", "origin", "main"], sandboxed=True, rules=[deny]).action == CommandPolicyAction.DENY
    assert check_command_allowed(["custom-check", "--verify"], rules=[allow]).action == CommandPolicyAction.ALLOW


def test_rule_precedence_is_deny_then_ask_then_allow() -> None:
    rules = [
        CommandRule(decision=CommandRuleDecision.ALLOW, prefix=("tool",)),
        CommandRule(decision=CommandRuleDecision.ASK, prefix=("tool", "deploy")),
        CommandRule(decision=CommandRuleDecision.DENY, prefix=("tool", "deploy", "prod")),
    ]

    assert check_command_allowed(["tool", "status"], sandboxed=True, rules=rules).action == CommandPolicyAction.ALLOW
    assert check_command_allowed(["tool", "deploy", "staging"], sandboxed=True, rules=rules).action == CommandPolicyAction.REQUIRE_APPROVAL
    assert check_command_allowed(["tool", "deploy", "prod"], sandboxed=True, rules=rules).action == CommandPolicyAction.DENY


def test_hard_safety_gate_cannot_be_overridden_by_allow_rule() -> None:
    rule = CommandRule(decision=CommandRuleDecision.ALLOW, prefix=("bash",))

    result = check_command_allowed(["bash", "-lc", "whoami"], sandboxed=True, rules=[rule])

    assert result.action == CommandPolicyAction.DENY


def test_sandboxed_commands_do_not_create_session_grants(tmp_path) -> None:
    assert resolve_command_session_grant(
        tmp_path,
        [sys.executable, "script.py"],
        sandboxed=True,
    ) is None

    ask_rule = CommandRule(
        decision=CommandRuleDecision.ASK,
        prefix=(Path(sys.executable).name, "script.py"),
    )
    assert resolve_command_session_grant(
        tmp_path,
        [sys.executable, "script.py"],
        sandboxed=True,
        rules=[ask_rule],
    ) is None


def test_session_grant_scopes_interpreter_payload_to_exact_argv(tmp_path) -> None:
    first = resolve_command_session_grant(
        tmp_path,
        [sys.executable, "-c", "open('a.txt','w').write('a')"],
    )
    second = resolve_command_session_grant(
        tmp_path,
        [sys.executable, "-c", "open('b.txt','w').write('b')"],
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_sandboxed_git_inspection_does_not_create_session_grant(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for executable identity resolution")

    assert resolve_command_session_grant(tmp_path, ["git", "show", "HEAD"], sandboxed=True) is None
    assert resolve_command_session_grant(tmp_path, ["git", "log", "-1"], sandboxed=True) is None


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.com",
        "pip install requests",
        "npm install",
        "python script.py",
        "python -c \"open('diagnostic.txt', 'w').write('x')\"",
        "mvn package",
        "git commit -m fix",
        "git log --output=history.txt -1",
        "git reflog expire --all",
    ],
)
def test_local_commands_default_to_approval_without_tool_specific_categories(command: str) -> None:
    result = check_command_allowed(_argv(command))

    assert result.action == CommandPolicyAction.REQUIRE_APPROVAL
    assert result.allowed is True
    assert result.requires_approval is True
    assert result.category == CommandCategory.UNKNOWN
    assert result.reason


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest ../outside/test_api.py",
        "dotnet test C:/outside/app.sln",
        "bash -lc whoami",
        "make test\nrm -rf .",
        "git reset --hard HEAD",
        "git clean -fdx",
        "git push --force origin main",
        "rm -rf .",
        "cat .env",
        "cat .git",
        "",
    ],
)
def test_command_policy_denies_dangerous_or_boundary_crossing_commands(command: str) -> None:
    result = check_command_allowed(_argv(command))

    assert result.action == CommandPolicyAction.DENY
    assert result.allowed is False
    assert result.requires_approval is False
    assert result.reason


def test_sandbox_policy_denies_sensitive_path_embedded_in_opaque_argument() -> None:
    result = check_command_allowed(
        ["python", "-c", "print(open('.env').read())"],
        sandboxed=True,
    )

    assert result.action == CommandPolicyAction.DENY
    assert "sensitive" in (result.reason or "").lower()


def test_sandbox_policy_allows_container_absolute_and_parent_paths() -> None:
    absolute = check_command_allowed(
        ["python", "/workspace/scripts/check.py"],
        sandboxed=True,
    )
    parent = check_command_allowed(
        ["python", "../tmp/check.py"],
        sandboxed=True,
    )

    assert absolute.action == CommandPolicyAction.ALLOW
    assert parent.action == CommandPolicyAction.ALLOW


def test_command_policy_classifies_argv_without_reparsing_display_text() -> None:
    ordinary = classify_argv(
        ["python", "-m", "pytest", "-q", "tests/test_api.py"],
        sandboxed=True,
    )
    literal_operator = classify_argv(
        ["python", "-m", "pytest", "|", "cat"],
        sandboxed=True,
    )

    assert ordinary.action == CommandPolicyAction.ALLOW
    assert ordinary.argv == ["python", "-m", "pytest", "-q", "tests/test_api.py"]
    assert literal_operator.action == CommandPolicyAction.ALLOW
    assert literal_operator.argv[-2:] == ["|", "cat"]


def test_run_command_schema_accepts_only_argv_payloads(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(str(workspace), enable_write=True)
    function = next(
        item["function"]
        for item in registry.schemas()
        if item["function"]["name"] == "run_command"
    )
    schema = function["parameters"]

    assert set(schema["properties"]) == {
        "argv",
        "timeout_seconds",
        "background",
    }
    assert "command" not in schema["properties"]
    assert "Commands that normally run without approval" not in function["description"]
    assert "explicit command rules may allow, ask, or deny" in function["description"]
    assert registry.admit(
        "run_command",
        {"argv": ["python", "-m", "pytest", "-q", "tests/test_api.py"]},
    ).arguments["argv"] == ["python", "-m", "pytest", "-q", "tests/test_api.py"]
    with pytest.raises(ValueError):
        registry.admit(
            "run_command",
            {"command": "python -m pytest -q tests/test_api.py"},
        )


def test_execute_admitted_rejects_denied_command_before_executor(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(str(workspace), enable_write=True)
    admission = registry.admit(
        "run_command",
        {"argv": ["bash", "-lc", "whoami"]},
    )

    assert admission.allowed is False
    with pytest.raises(PermissionError):
        registry.execute_admitted(admission, approval_granted=True)


def test_decode_command_output_preserves_utf8_and_local_console_diagnostics() -> None:
    assert _decode_command_output("语法错误".encode("utf-8")) == "语法错误"
    assert _decode_command_output("找不到符号".encode("gb18030")) == "找不到符号"
    mixed_gb18030 = "找不到符号".encode("gb18030") + b"\xff"
    assert _decode_command_output(mixed_gb18030).startswith("找不到符号")
    assert _decode_command_output("already decoded") == "already decoded"
    assert _decode_command_output(None) == ""


def test_resolve_command_argv_uses_running_python_for_bare_python_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: (
            "C:/mock/WindowsApps/python.exe"
            if command == "python"
            else None
        ),
    )

    assert _resolve_command_argv(["python", "-m", "pytest"]) == [
        sys.executable,
        "-m",
        "pytest",
    ]


def test_resolve_command_argv_still_uses_path_for_non_python_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: "C:/Tools/git.exe" if command == "git" else None,
    )

    assert _resolve_command_argv(["git", "status"]) == [
        "C:/Tools/git.exe",
        "status",
    ]


def test_run_command_requires_explicit_approval_for_side_effect_command(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "script.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="side-effect free|approval"):
        run_command(workspace, ["python", "script.py"])

    result = run_command(workspace, ["python", "script.py"], approval_granted=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize(
    "command",
    [
        ["git", "status"],
        ["git", "status", "--short", "--branch"],
        ["git", "branch", "--show-current"],
        ["git", "ls-files"],
        ["git", "ls-files", "src/main/java"],
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "--show-toplevel"],
        ["git", "log", "--oneline", "-30"],
        ["git", "show", "HEAD"],
        ["git", "blame", "README.md"],
        ["git", "cat-file", "-p", "HEAD"],
        ["git", "ls-tree", "HEAD"],
        ["git", "reflog", "show", "HEAD"],
    ],
)
def test_sandbox_allows_git_inspection_without_git_specific_policy(command) -> None:
    result = check_command_allowed(command, sandboxed=True)

    assert result.action == CommandPolicyAction.ALLOW
    assert result.allowed is True
    assert result.requires_approval is False
    assert result.category == CommandCategory.UNKNOWN


def test_tool_registry_admits_git_ls_files_without_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(str(workspace), enable_write=True)

    admission = registry.admit(
        "run_command",
        {"argv": ["git", "ls-files", "src/main/java"]},
    )

    assert admission.risk_level == RiskLevel.MEDIUM
    assert admission.requires_approval is True
    assert admission.command_policy is not None
    assert admission.command_policy.category == CommandCategory.UNKNOWN


def test_run_command_allows_read_only_git_inspection_without_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = run_command(
        workspace,
        ["git", "status"],
        timeout_seconds=10,
        approval_granted=True,
    )

    assert result.returncode != 0
    assert "not a git repository" in result.stderr.lower()


def test_run_command_isolates_child_stdin(tmp_path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class CompletedProcess:
        returncode = 0
        pid = 123

        def communicate(self, timeout=None):
            return b"ok\n", b""

        def poll(self):
            return 0

    def fake_popen(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return CompletedProcess()

    monkeypatch.setattr(write_tools_module.subprocess, "Popen", fake_popen)

    result = run_command(
        tmp_path,
        ["python", "--version"],
        timeout_seconds=10,
        approval_granted=True,
    )

    assert result.returncode == 0
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL


def test_windows_process_cleanup_kills_tree_and_never_communicates_without_timeout(
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class HangingProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            raise AssertionError("Windows cleanup should terminate the process tree first")

        def kill(self):
            self.returncode = -9

        def communicate(self, timeout=None):
            assert timeout is not None, "process cleanup must never use unbounded communicate()"
            if self.returncode is None:
                raise subprocess.TimeoutExpired("git", timeout)
            return b"partial", b""

        def wait(self, timeout=None):
            assert timeout is not None
            return self.returncode

    process = HangingProcess()

    def fake_run(arguments, **kwargs):
        calls.append((list(arguments), dict(kwargs)))
        process.returncode = 1
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(write_tools_module.os, "name", "nt")
    monkeypatch.setattr(write_tools_module.subprocess, "run", fake_run)

    stdout, stderr = _terminate_process(process)

    assert stdout == b"partial"
    assert stderr == b""
    assert calls == [
        (
            ["taskkill", "/F", "/T", "/PID", "4242"],
            {
                "shell": False,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "timeout": 2.0,
                "check": False,
            },
        )
    ]


def test_process_cleanup_stays_bounded_when_pipes_never_reach_eof(monkeypatch) -> None:
    communicate_timeouts: list[float | None] = []

    class Pipe:
        def close(self) -> None:
            pass

    class HangingProcess:
        pid = 4242
        returncode = None
        stdout = Pipe()
        stderr = Pipe()

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def communicate(self, timeout=None):
            communicate_timeouts.append(timeout)
            raise subprocess.TimeoutExpired("git", timeout, output=b"partial", stderr=b"")

        def wait(self, timeout=None):
            assert timeout is not None
            return self.returncode

    monkeypatch.setattr(write_tools_module.os, "name", "posix")

    stdout, stderr = _terminate_process(HangingProcess())

    assert stdout == b"partial"
    assert stderr == b""
    assert communicate_timeouts == [1.0, 1.0]


def test_run_command_reports_resolve_spawn_and_execute_timings(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = run_command(
        workspace,
        [sys.executable, "-c", "print('ok')"],
        timeout_seconds=10,
        approval_granted=True,
    )

    assert result.returncode == 0
    assert result.resolve_duration_ms >= 0
    assert result.spawn_duration_ms >= 0
    assert result.execute_duration_ms >= 0
    assert (
        result.resolve_duration_ms
        + result.spawn_duration_ms
        + result.execute_duration_ms
        <= round(result.duration_seconds * 1000) + 50
    )


def test_sanitized_command_environment_removes_provider_credentials(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    monkeypatch.setenv("CUSTOM_AUTH_TOKEN", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("PATH", "tool-path")

    environment = _sanitized_command_environment()

    assert "DASHSCOPE_API_KEY" not in environment
    assert "CUSTOM_AUTH_TOKEN" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert environment["PATH"] == "tool-path"


def test_write_file_respects_workspace_guard(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = write_file(workspace, "src/main.py", "VALUE = 1\n")

    assert result.created is True
    assert result.changed is True
    assert result.path == "src/main.py"
    assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    with pytest.raises(FileExistsError, match="Target file already exists"):
        write_file(workspace, "src/main.py", "VALUE = 1\n")

    unchanged = write_file(
        workspace,
        "src/main.py",
        "VALUE = 1\n",
        overwrite=True,
    )
    assert unchanged.created is False
    assert unchanged.overwritten is True
    assert unchanged.changed is False

    updated = write_file(
        workspace,
        "src/main.py",
        "VALUE = 2\n",
        overwrite=True,
    )
    assert updated.created is False
    assert updated.overwritten is True
    assert updated.changed is True

    with pytest.raises(WorkspaceAccessError, match="escapes workspace"):
        write_file(workspace, "../outside.txt", "nope\n")
    with pytest.raises(WorkspaceAccessError, match="sensitive path"):
        write_file(workspace, ".env", "TOKEN=secret\n")


def test_apply_patch_extracts_and_validates_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("old\n", encoding="utf-8")
    patch = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""

    assert extract_patch_paths(patch) == ["README.md"]
    result = apply_patch(workspace, patch)

    assert result.files == ["README.md"]
    assert (workspace / "README.md").read_text(encoding="utf-8") == "new\n"


def test_apply_patch_rejects_paths_outside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    patch = """diff --git a/../outside.txt b/../outside.txt
--- a/../outside.txt
+++ b/../outside.txt
@@ -1 +1 @@
-old
+new
"""

    with pytest.raises(WorkspaceAccessError, match="escapes workspace"):
        apply_patch(workspace, patch)


def test_tool_registry_only_exposes_write_tools_when_enabled(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    read_only_tools = {
        schema["function"]["name"]
        for schema in ToolRegistry(str(workspace), enable_write=False).schemas()
    }
    write_enabled_tools = {
        schema["function"]["name"]
        for schema in ToolRegistry(str(workspace), enable_write=True).schemas()
    }

    assert "write" not in read_only_tools
    assert {"apply_patch", "write", "run_command"}.issubset(write_enabled_tools)
