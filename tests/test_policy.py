import shlex
import shutil
import sys

import pytest

from minicode_harness.policy import (
    CommandCategory,
    CommandPolicyAction,
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
)
from minicode_harness.workspace import WorkspaceAccessError


def _argv(command: str) -> list[str]:
    if "\n" in command or "\r" in command:
        return [command]
    return shlex.split(command)


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        ("python -m pytest", "python -m pytest [focused args]"),
        ("python -m pytest -q tests/test_api.py", "python -m pytest [focused args]"),
        (
            "python -c \"from sympy import Symbol; print(Symbol.mro())\"",
            "python -c read-only diagnostic",
        ),
        ("python -m compileall -q src/app.py", "python -m compileall [paths]"),
        ("python -m ruff check src", "python -m ruff check [paths]"),
        ("node --check src/app.js", "node --check <file>"),
        ("npm test", "npm test [focused args]"),
        ("pnpm test -- src/api.test.ts", "pnpm test [focused args]"),
        ("npm run lint", "npm run lint [focused args]"),
        ("go test ./...", "go test [packages]"),
        ("go vet ./...", "go vet [packages]"),
        ("cargo test parser", "cargo test [filter]"),
        ("cargo check --all-targets", "cargo check [filter]"),
        ("dotnet test app.sln", "dotnet test [project]"),
        ("dotnet build app.sln", "dotnet build [project]"),
        ("mvn -q test", "mvn test"),
        ("mvn -q -Dtest=CouponServiceTest test", "mvn -q -Dtest=<selector> test"),
        ("mvn -q test -Dtest=CouponServiceTest", "mvn -q -Dtest=<selector> test"),
        ("mvn -q -DskipTests compile", "mvn -q -DskipTests compile"),
        ("./gradlew test", "gradle test"),
        ("./gradlew check", "gradle check"),
        ("gradle test --tests com.example.ServiceTest", "gradle test --tests <selector>"),
        ("bundle exec rspec spec/service_spec.rb", "bundle exec rspec [path]"),
        ("mix test test/service_test.exs", "mix test [path]"),
        ("swift test", "swift test"),
        ("make test", "make test"),
        ("make lint", "make lint"),
        ("ctest --test-dir build", "ctest --test-dir <path>"),
        ("cmake --build build", "cmake --build <path>"),
    ],
)
def test_command_allowlist_accepts_supported_verification_families(command: str, rule: str) -> None:
    result = check_command_allowed(_argv(command))

    assert result.action == CommandPolicyAction.ALLOW
    assert result.allowed is True
    assert result.requires_approval is False
    assert result.rule == rule
    assert result.argv


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


def test_session_grant_reuses_only_the_same_git_history_family(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for executable identity resolution")

    first_show = resolve_command_session_grant(tmp_path, ["git", "show", "HEAD"])
    second_show = resolve_command_session_grant(tmp_path, ["git", "show", "HEAD~1"])
    git_log = resolve_command_session_grant(tmp_path, ["git", "log", "-1"])

    assert first_show is not None
    assert first_show == second_show
    assert git_log is not None
    assert first_show != git_log


@pytest.mark.parametrize(
    ("command", "category"),
    [
        ("curl https://example.com", CommandCategory.NETWORK),
        ("pip install requests", CommandCategory.DEPENDENCY),
        ("npm install", CommandCategory.DEPENDENCY),
        ("python script.py", CommandCategory.REPOSITORY_SCRIPT),
        (
            "python -c \"open('diagnostic.txt', 'w').write('x')\"",
            CommandCategory.DIAGNOSTIC,
        ),
        ("mvn package", CommandCategory.UNKNOWN),
        ("git commit -m fix", CommandCategory.GIT_MUTATION),
        ("git show HEAD", CommandCategory.GIT_HISTORY),
    ],
)
def test_command_policy_routes_side_effects_to_approval(
    command: str,
    category: CommandCategory,
) -> None:
    result = check_command_allowed(_argv(command))

    assert result.action == CommandPolicyAction.REQUIRE_APPROVAL
    assert result.allowed is True
    assert result.requires_approval is True
    assert result.category == category
    assert result.reason


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest ../outside/test_api.py",
        "dotnet test C:/outside/app.sln",
        "bash -lc whoami",
        "go test ./... && rm -rf .",
        "go test ./... & echo done",
        "cargo test | cat",
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


def test_command_policy_classifies_argv_without_reparsing_display_text() -> None:
    allowed = classify_argv(["python", "-m", "pytest", "-q", "tests/test_api.py"])
    denied = classify_argv(["python", "-m", "pytest", "|", "cat"])

    assert allowed.action == CommandPolicyAction.ALLOW
    assert allowed.argv == ["python", "-m", "pytest", "-q", "tests/test_api.py"]
    assert denied.action == CommandPolicyAction.DENY
    assert "Shell chaining" in str(denied.reason)


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
    assert "Policy may allow, require approval, or deny" in function["description"]
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
    ],
)
def test_command_policy_allows_read_only_git_inspection_without_approval(command) -> None:
    result = check_command_allowed(command)

    assert result.action == CommandPolicyAction.ALLOW
    assert result.allowed is True
    assert result.requires_approval is False
    assert result.category == CommandCategory.INFORMATION


def test_tool_registry_admits_git_ls_files_without_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(str(workspace), enable_write=True)

    admission = registry.admit(
        "run_command",
        {"argv": ["git", "ls-files", "src/main/java"]},
    )

    assert admission.risk_level == RiskLevel.LOW
    assert admission.requires_approval is False
    assert admission.command_policy is not None
    assert admission.command_policy.category == CommandCategory.INFORMATION


def test_run_command_allows_read_only_git_inspection_without_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = run_command(workspace, ["git", "status"], timeout_seconds=10)

    assert result.returncode != 0
    assert "not a git repository" in result.stderr.lower()


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
