from minicode_harness.workspace import (
    extract_source_paths,
    is_code_path,
    preferred_verification_command_for_paths,
    scan_workspace_profile,
    workspace_profile_may_change,
)


def test_workspace_profile_detects_python_repository(tmp_path) -> None:
    workspace = tmp_path / "python-project"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (workspace / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "tests" / "test_app.py").write_text("def test_value(): pass\n", encoding="utf-8")

    profile = scan_workspace_profile(workspace)

    assert profile.language == "Python"
    assert profile.build_system == "pyproject"
    assert profile.source_roots == ["src"]
    assert profile.test_roots == ["tests"]
    assert profile.has_tests is True
    assert profile.preferred_verification_commands == ["python -m pytest -q"]
    assert preferred_verification_command_for_paths(
        workspace,
        ["tests/test_app.py"],
    ) == "python -m pytest -q tests/test_app.py"


def test_workspace_profile_detects_multiple_toolchains_without_language_priority_rules(tmp_path) -> None:
    workspace = tmp_path / "mixed-project"
    workspace.mkdir()
    (workspace / "package.json").write_text('{"scripts":{"test":"vitest"}}\n', encoding="utf-8")
    (workspace / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (workspace / "web").mkdir()
    (workspace / "web" / "app.ts").write_text("export const value = 1;\n", encoding="utf-8")
    (workspace / "service").mkdir()
    (workspace / "service" / "main.go").write_text("package service\n", encoding="utf-8")

    profile = scan_workspace_profile(workspace)

    assert set(profile.languages) == {"TypeScript", "Go"}
    assert profile.build_systems == ["npm", "Go modules"]
    assert profile.preferred_verification_commands == ["npm test", "go test ./..."]


def test_verification_command_is_inferred_from_changed_test_path(tmp_path) -> None:
    workspace = tmp_path / "node-project"
    (workspace / "src").mkdir(parents=True)
    (workspace / "package.json").write_text('{"scripts":{"test":"vitest"}}\n', encoding="utf-8")
    (workspace / "src" / "api.test.ts").write_text("test('ok', () => {});\n", encoding="utf-8")

    command = preferred_verification_command_for_paths(workspace, ["src/api.test.ts"])

    assert command == "npm test -- src/api.test.ts"


def test_verification_command_falls_back_to_safe_syntax_check(tmp_path) -> None:
    workspace = tmp_path / "single-file-project"
    workspace.mkdir()
    (workspace / "script.py").write_text("VALUE = 1\n", encoding="utf-8")

    command = preferred_verification_command_for_paths(workspace, ["script.py"])

    assert command == "python -m compileall -q script.py"


def test_workspace_profile_invalidation_is_limited_to_build_files() -> None:
    assert workspace_profile_may_change(["pom.xml"]) is True
    assert workspace_profile_may_change(["modules/api/build.gradle.kts"]) is True
    assert workspace_profile_may_change(["src/app.py", "tests/test_app.py"]) is False


def test_source_path_helpers_are_language_neutral() -> None:
    text = "tests/test_api.py:12 failed\nsrc/parser.rs:8:4 error\nweb/app.ts(9,2): error"

    assert extract_source_paths(text) == [
        "tests/test_api.py",
        "src/parser.rs",
        "web/app.ts",
    ]
    assert is_code_path("src/parser.rs") is True
    assert is_code_path("README.md") is False
