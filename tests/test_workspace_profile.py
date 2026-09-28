from minicode_harness.workspace import (
    WorkspaceProfile,
    extract_source_paths,
    is_code_path,
    scan_workspace_profile,
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
    assert "has_tests" not in WorkspaceProfile.model_fields
    assert "preferred_verification_commands" not in WorkspaceProfile.model_fields


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


def test_source_path_helpers_are_language_neutral() -> None:
    text = "tests/test_api.py:12 failed\nsrc/parser.rs:8:4 error\nweb/app.ts(9,2): error"

    assert extract_source_paths(text) == [
        "tests/test_api.py",
        "src/parser.rs",
        "web/app.ts",
    ]
    assert is_code_path("src/parser.rs") is True
    assert is_code_path("README.md") is False
