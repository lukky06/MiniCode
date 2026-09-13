from pathlib import Path
import hashlib
import shutil
import subprocess

import pytest

import minicode_harness.tools.read_tools as read_tools_module
from minicode_harness.context import build_observation
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.tools import ToolRegistry, find_files, inspect_git_diff, read_artifact, read_file, search_text
from minicode_harness.workspace import WorkspaceAccessError


def test_find_files_returns_sorted_non_sensitive_files(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "B.java").write_text("class B {}\n", encoding="utf-8")
    (workspace / "src" / "A.java").write_text("class A {}\n", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    result = find_files(workspace)

    assert result.root == "."
    assert result.pattern == "**/*"
    assert result.files == ["src/A.java", "src/B.java"]
    assert [item.path for item in result.file_details] == result.files
    assert all(item.size_bytes > 0 for item in result.file_details)
    assert all(item.estimated_lines >= 1 for item in result.file_details)
    assert result.truncated is False


def test_find_files_reports_size_and_estimated_lines_without_changing_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    content = "x" * 161
    (workspace / "large.txt").write_text(content, encoding="utf-8")

    result = find_files(workspace, pattern="*.txt")

    assert result.files == ["large.txt"]
    detail = result.file_details[0]
    assert detail.path == "large.txt"
    assert detail.size_bytes == len(content.encode("utf-8"))
    assert detail.estimated_lines == 3


def test_find_files_filesystem_walker_prunes_standard_ignored_directories(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "Main.java").write_text("class Main {}\n", encoding="utf-8")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "ignored.js").write_text("ignored\n", encoding="utf-8")
    (workspace / "target").mkdir()
    (workspace / "target" / "Generated.class").write_bytes(b"generated")

    result = find_files(workspace, pattern="**/*", max_depth=8)

    assert "src/Main.java" in result.files
    assert not any(path.startswith("node_modules/") for path in result.files)
    assert not any(path.startswith("target/") for path in result.files)
    assert result.truncated is False
    assert result.truncation_reason is None


def test_find_files_filesystem_walker_does_not_probe_beyond_max_depth(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "module" / "src" / "main" / "Main.java"
    nested.parent.mkdir(parents=True)
    nested.write_text("class Main {}\n", encoding="utf-8")

    result = find_files(workspace, pattern="*.java", max_depth=2)

    assert result.files == []
    assert result.truncated is False


def test_find_files_returns_matches_within_requested_depth(tmp_path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Top.java").write_text("class Top {}\n", encoding="utf-8")
    nested = workspace / "deep" / "nested" / "Deep.java"
    nested.parent.mkdir(parents=True)
    nested.write_text("class Deep {}\n", encoding="utf-8")
    subprocess.run([git, "init", "-q"], cwd=workspace, check=True)

    result = find_files(workspace, pattern="**/*.java", max_depth=1)

    assert result.files == ["Top.java"]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("**/pom.xml", ["pom.xml"]),
        (
            "src/main/java/**/*.java",
            ["src/main/java/App.java", "src/main/java/com/acme/Nested.java"],
        ),
        (
            "**/*.java",
            [
                "src/main/java/App.java",
                "src/main/java/com/acme/Nested.java",
                "src/test/java/AppTest.java",
            ],
        ),
    ],
)
def test_find_files_supports_standard_double_star_globs(tmp_path, pattern, expected) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src" / "main" / "java" / "com" / "acme").mkdir(parents=True)
    (workspace / "src" / "test" / "java").mkdir(parents=True)
    (workspace / "pom.xml").write_text("<project/>\n", encoding="utf-8")
    (workspace / "src" / "main" / "java" / "App.java").write_text(
        "class App {}\n",
        encoding="utf-8",
    )
    (workspace / "src" / "main" / "java" / "com" / "acme" / "Nested.java").write_text(
        "class Nested {}\n",
        encoding="utf-8",
    )
    (workspace / "src" / "test" / "java" / "AppTest.java").write_text(
        "class AppTest {}\n",
        encoding="utf-8",
    )

    result = find_files(workspace, pattern=pattern, max_depth=12)

    assert result.files == expected


def test_find_files_fallback_prunes_generated_directories(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "Main.java"
    ignored = workspace / "node_modules" / "hmdp-vue3" / "node_modules" / "ignored.js"
    source.parent.mkdir(parents=True)
    ignored.parent.mkdir(parents=True)
    source.write_text("class Main {}\n", encoding="utf-8")
    ignored.write_text("ignored\n", encoding="utf-8")

    result = find_files(workspace, pattern="**/*", max_depth=12)

    assert result.files == ["src/Main.java"]
    assert result.truncated is False
    assert result.scanned_entries == 3


def test_find_files_applies_depth_before_descending(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "a" / "deep" / "hidden.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("hidden\n", encoding="utf-8")

    result = find_files(workspace, pattern="*.py", max_depth=1)

    assert result.files == []
    assert result.truncated is False
    assert result.scanned_entries == 1


def test_find_files_stops_after_one_extra_match(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("a\n", encoding="utf-8")
    (workspace / "b.py").write_text("b\n", encoding="utf-8")
    (workspace / "z").mkdir()
    (workspace / "z" / "never_scanned.py").write_text("z\n", encoding="utf-8")

    result = find_files(workspace, pattern="*.py", max_results=1, max_depth=4)

    assert result.files == ["a.py"]
    assert result.truncated is True
    assert result.truncation_reason == "result_limit"
    assert result.scanned_entries == 2


def test_find_files_reports_scan_limit(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("a\n", encoding="utf-8")
    (workspace / "b.py").write_text("b\n", encoding="utf-8")
    monkeypatch.setattr(read_tools_module, "FILE_SEARCH_SCAN_LIMIT", 1)

    result = find_files(workspace, pattern="*.py", max_results=10)

    assert result.files == ["a.py"]
    assert result.truncated is True
    assert result.truncation_reason == "scan_limit"
    assert result.scanned_entries == 1


def test_find_files_reports_timeout(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("a\n", encoding="utf-8")
    monkeypatch.setattr(read_tools_module, "FILE_SEARCH_TIMEOUT_SECONDS", 0.0)

    result = find_files(workspace, pattern="*.py", max_results=10)

    assert result.files == []
    assert result.truncated is True
    assert result.truncation_reason == "timeout"
    assert result.scanned_entries == 0


def test_find_files_does_not_spawn_git_subprocess(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    (workspace / "src").mkdir()
    (workspace / "src" / "Main.java").write_text("class Main {}\n", encoding="utf-8")

    def fail_subprocess(*args, **kwargs):
        pytest.fail("File discovery must not spawn git or any other subprocess.")

    monkeypatch.setattr(read_tools_module.subprocess, "run", fail_subprocess)

    result = find_files(workspace, pattern="**/*.java")

    assert result.files == ["src/Main.java"]
    assert result.truncated is False


def test_find_files_observes_cancellation_before_scanning(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Main.java").write_text("class Main {}\n", encoding="utf-8")
    cancellation = CancellationToken()
    cancellation.cancel()

    result = find_files(
        workspace,
        pattern="**/*.java",
        cancellation_token=cancellation,
    )

    assert result.files == []
    assert result.truncated is True
    assert result.truncation_reason == "cancelled"
    assert result.scanned_entries == 0


def test_find_files_reports_missing_directory_as_empty_result(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = find_files(workspace, "src/test/java/com/example")

    assert result.root == "src/test/java/com/example"
    assert result.files == []
    assert result.truncated is False
    assert result.exists is False
    assert result.is_directory is False


def test_find_files_reports_file_target_without_traversing(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "src" / "main.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")

    result = find_files(workspace, "src/main.py")

    assert result.files == []
    assert result.exists is True
    assert result.is_directory is False


def test_find_files_applies_pattern_depth_and_truncation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "a").mkdir(parents=True)
    (workspace / "a" / "one.py").write_text("one\n", encoding="utf-8")
    (workspace / "a" / "two.py").write_text("two\n", encoding="utf-8")
    (workspace / "a" / "nested").mkdir()
    (workspace / "a" / "nested" / "three.py").write_text("three\n", encoding="utf-8")
    (workspace / "a" / "note.txt").write_text("note\n", encoding="utf-8")

    result = find_files(
        workspace,
        pattern="test_*.py",
        max_results=10,
        max_depth=2,
    )
    assert result.files == []

    result = find_files(
        workspace,
        pattern="**/*.py",
        max_results=1,
        max_depth=2,
    )
    assert result.files == ["a/one.py"]
    assert result.truncated is True


def test_read_file_directory_error_suggests_file_search(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)

    with pytest.raises(IsADirectoryError, match=r'search\(kind="files"'):
        read_file(workspace, "src")


def test_read_file_returns_line_slice(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source_file = workspace / "src" / "Service.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    result = read_file(workspace, "src/Service.java", start_line=2, end_line=3)

    assert result.path == "src/Service.java"
    assert result.content == "line 2\nline 3"
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.total_lines == 3
    assert result.returned_lines == 2
    assert result.returned_chars == len("line 2\nline 3")
    assert result.total_chars == len("line 1\nline 2\nline 3\n")
    assert result.full_resource_read is False
    assert result.guidance is None


def test_read_file_hashes_raw_bytes_without_exposing_crlf(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source_file = workspace / "src" / "Service.java"
    source_file.parent.mkdir(parents=True)
    raw_content = b"line 1\r\nline 2\r\n"
    source_file.write_bytes(raw_content)

    result = read_file(workspace, "src/Service.java")

    assert result.content == "line 1\nline 2"
    assert result.total_chars == len("line 1\nline 2\n")
    assert result.content_sha256 == hashlib.sha256(raw_content).hexdigest()


def test_large_full_read_reports_raw_cost_and_search_guidance(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    content = "\n".join(f"line {index}" for index in range(600))
    (workspace / "large.py").write_text(content, encoding="utf-8")

    full = read_file(workspace, "large.py")
    partial = read_file(workspace, "large.py", start_line=10, end_line=30)

    assert full.total_lines == 600
    assert full.returned_lines == 600
    assert full.total_chars == len(content)
    assert full.full_resource_read is True
    assert "text search on the exact resource" in str(full.guidance)
    assert partial.full_resource_read is False
    assert partial.guidance is None


def test_read_file_handles_empty_files(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    empty_file = workspace / "empty.txt"
    empty_file.parent.mkdir(parents=True)
    empty_file.write_text("", encoding="utf-8")

    result = read_file(workspace, "empty.txt")

    assert result.content == ""
    assert result.start_line == 1
    assert result.end_line == 0
    assert result.total_lines == 0


def test_read_file_rejects_start_line_past_end(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source_file = workspace / "src" / "Service.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("line 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="number of lines"):
        read_file(workspace, "src/Service.java", start_line=2)


def test_read_file_rejects_sensitive_and_outside_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")

    with pytest.raises(WorkspaceAccessError, match="sensitive path"):
        read_file(workspace, ".env")

    with pytest.raises(WorkspaceAccessError, match="escapes workspace"):
        read_file(workspace, "../outside.txt")


def test_read_artifact_schema_distinguishes_artifact_path_from_evidence_label(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "run" / "artifacts"
    workspace.mkdir()
    artifacts.mkdir(parents=True)

    registry = ToolRegistry(str(workspace), artifact_dir=str(artifacts))
    schema = registry.schemas(["read"])[0]
    parameters = schema["function"]["parameters"]

    assert "artifact" in parameters["properties"]["source"]["enum"]
    assert "target" in parameters["properties"]
    assert parameters.get("additionalProperties") is False


def test_read_artifact_reads_only_current_run_artifacts(tmp_path) -> None:
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)
    artifact = artifacts / "read_file_call_1.txt"
    artifact.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")

    result = read_artifact(artifacts, "read_file_call_1.txt", start_line=2, end_line=3)

    assert result.path == "read_file_call_1.txt"
    assert result.content == "line 2\nline 3"
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.total_lines == 3

    with pytest.raises(WorkspaceAccessError, match="escapes artifact directory"):
        read_artifact(artifacts, outside)

    with pytest.raises(ValueError, match="not an evidence id"):
        read_artifact(artifacts, "ev_0001_call_1")


def test_compressed_artifact_path_can_be_read_verbatim(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source_file = workspace / "src" / "Large.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("\n".join(f"line {index}" for index in range(900)), encoding="utf-8")
    artifacts = tmp_path / "run" / "artifacts"

    file_result = read_file(workspace, "src/Large.java")
    observation, _ = build_observation(
        tool_call_id="call_1",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "src/Large.java"},
        result=file_result,
        artifact_dir=artifacts,
    )
    artifact_result = read_artifact(artifacts, observation.artifact_path or "")

    assert observation.artifact_path == "read_call_1.txt"
    assert artifact_result.path == "read_call_1.txt"
    assert artifact_result.content.splitlines()[0] == "line 0"
    assert artifact_result.content.splitlines()[-1] == "line 899"
    assert '"content"' not in artifact_result.content


def test_read_artifact_rejects_paths_outside_current_artifact_root(tmp_path) -> None:
    artifacts = tmp_path / "run_20260702_001" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "workspace_read_call_1.txt").write_text("artifact\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Path is not an artifact file"):
        read_artifact(
            artifacts,
            "runs/run_20260702_001/artifacts/workspace_read_call_1.txt",
        )


def test_search_text_finds_matches_and_skips_sensitive_files(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "CouponService.java").write_text(
        "class CouponService {\n  String applyCoupon() { return \"ok\"; }\n}\n",
        encoding="utf-8",
    )
    (workspace / ".env").write_text("CouponService_SECRET=1\n", encoding="utf-8")

    result = search_text(workspace, "CouponService")

    assert [(match.path, match.line) for match in result.matches] == [
        ("src/CouponService.java", 1)
    ]
    assert result.truncated is False


def test_search_text_rejects_sensitive_search_roots(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "config").write_text("needle\n", encoding="utf-8")

    with pytest.raises(WorkspaceAccessError, match="sensitive path"):
        search_text(workspace, "needle", ".git")


def test_search_text_filesystem_walker_prunes_standard_ignored_directories(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "src" / "Main.java"
    source.parent.mkdir()
    source.write_text("class Main { String value = \"needle\"; }\n", encoding="utf-8")
    for directory in ("node_modules", "target", "build"):
        ignored = workspace / directory / "Ignored.java"
        ignored.parent.mkdir()
        ignored.write_text("needle\n", encoding="utf-8")

    result = search_text(workspace, "needle", file_glob="*.java", max_depth=8)

    assert [match.path for match in result.matches] == ["src/Main.java"]
    assert result.truncated is False
    assert result.truncation_reason is None


def test_search_text_supports_standard_double_star_file_glob(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    direct = workspace / "src" / "main" / "java" / "App.java"
    nested = workspace / "src" / "main" / "java" / "com" / "acme" / "Nested.java"
    test_file = workspace / "src" / "test" / "java" / "AppTest.java"
    direct.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    for path in (direct, nested, test_file):
        path.write_text("needle\n", encoding="utf-8")

    result = search_text(
        workspace,
        "needle",
        file_glob="src/main/java/**/*.java",
        max_depth=12,
    )

    assert [match.path for match in result.matches] == [
        "src/main/java/App.java",
        "src/main/java/com/acme/Nested.java",
    ]


def test_search_text_fallback_prunes_generated_directories(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "Main.java"
    source.parent.mkdir(parents=True)
    source.write_text("needle\n", encoding="utf-8")
    for directory in ("node_modules", "target", "build"):
        ignored = workspace / directory / "deep" / "Ignored.java"
        ignored.parent.mkdir(parents=True)
        ignored.write_text("needle\n", encoding="utf-8")

    result = search_text(workspace, "needle", file_glob="*.java", max_depth=8)

    assert [match.path for match in result.matches] == ["src/Main.java"]
    assert result.truncated is False
    assert result.scanned_entries < 10


def test_search_text_applies_file_glob_before_opening(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Main.java").write_text("needle\n", encoding="utf-8")
    (workspace / "notes.txt").write_text("needle\n", encoding="utf-8")
    opened: list[str] = []
    original_open = Path.open

    def tracked_open(path: Path, *args, **kwargs):
        opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    result = search_text(workspace, "needle", file_glob="*.java")

    assert [match.path for match in result.matches] == ["Main.java"]
    assert opened == ["Main.java"]


def test_search_text_reports_scan_limit(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.java").write_text("no match\n", encoding="utf-8")
    (workspace / "b.java").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(read_tools_module, "TEXT_SEARCH_SCAN_LIMIT", 1)

    result = search_text(workspace, "needle", file_glob="*.java")

    assert result.matches == []
    assert result.truncated is True
    assert result.truncation_reason == "scan_limit"
    assert result.scanned_entries == 1


def test_search_text_reports_timeout(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Main.java").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(read_tools_module, "TEXT_SEARCH_TIMEOUT_SECONDS", 0.0)

    result = search_text(workspace, "needle", file_glob="*.java")

    assert result.matches == []
    assert result.truncated is True
    assert result.truncation_reason == "timeout"
    assert result.scanned_entries == 0


def test_tool_registry_file_search_observes_cancellation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Main.java").write_text("class Main {}\n", encoding="utf-8")
    cancellation = CancellationToken()
    cancellation.cancel()
    registry = ToolRegistry(str(workspace), cancellation_token=cancellation)

    result = registry.execute_admitted(
        registry.admit(
            "search",
            {
                "kind": "files",
                "query": "**/*.java",
                "path": ".",
            },
        )
    )

    assert result.files == []
    assert result.truncated is True
    assert result.truncation_reason == "cancelled"
    assert result.scanned_entries == 0


def test_tool_registry_text_search_observes_cancellation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Main.java").write_text("needle\n", encoding="utf-8")
    cancellation = CancellationToken()
    cancellation.cancel()
    registry = ToolRegistry(str(workspace), cancellation_token=cancellation)

    result = registry.execute_admitted(
        registry.admit(
            "search",
            {
                "kind": "text",
                "query": "needle",
                "path": ".",
                "file_glob": "*.java",
            },
        )
    )

    assert result.matches == []
    assert result.truncated is True
    assert result.truncation_reason == "cancelled"
    assert result.scanned_entries == 0


def test_tool_registry_searches_artifact_files_and_text(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    log_file = artifacts / "logs" / "build.txt"
    log_file.parent.mkdir(parents=True)
    log_file.write_text("setup ok\nERROR missing symbol\ndone\n", encoding="utf-8")
    registry = ToolRegistry(str(workspace), artifact_dir=str(artifacts))

    files = registry.execute_admitted(
        registry.admit(
            "search",
            {
                "source": "artifact",
                "kind": "files",
                "query": "**/*.txt",
                "path": ".",
            },
        )
    )
    matches = registry.execute_admitted(
        registry.admit(
            "search",
            {
                "source": "artifact",
                "kind": "text",
                "query": "ERROR",
                "path": "logs/build.txt",
            },
        )
    )

    assert files.files == ["logs/build.txt"]
    assert [(match.path, match.line, match.text) for match in matches.matches] == [
        ("logs/build.txt", 2, "ERROR missing symbol")
    ]


def test_tool_registry_artifact_search_requires_artifact_root(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(str(workspace))

    result = registry.execute_admitted(
        registry.admit(
            "search",
            {
                "source": "artifact",
                "kind": "text",
                "query": "ERROR",
                "path": "build.txt",
            },
        )
    )

    assert result == {
        "status": "unavailable",
        "reason": "artifacts_disabled",
    }


def test_search_text_observes_cancellation_before_scanning(
    monkeypatch,
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    cancellation = CancellationToken()
    cancellation.cancel()

    def fail_subprocess(*args, **kwargs):
        pytest.fail("Cancelled text search must not start any subprocess")

    monkeypatch.setattr(read_tools_module.subprocess, "run", fail_subprocess)

    result = search_text(
        workspace,
        "needle",
        file_glob="*.java",
        cancellation_token=cancellation,
    )

    assert result.matches == []
    assert result.truncation_reason == "cancelled"
    assert result.scanned_entries == 0


def test_inspect_git_diff_returns_workspace_diff(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "mini-code@example.test"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Mini Code"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    tracked_file = workspace / "README.md"
    tracked_file.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    tracked_file.write_text("new\n", encoding="utf-8")

    result = inspect_git_diff(workspace)

    assert result.returncode == 0
    assert "-old" in result.diff
    assert "+new" in result.diff


def test_inspect_git_diff_rejects_parent_repository_leakage(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    repository = tmp_path / "repository"
    workspace = repository / "nested-workspace"
    workspace.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "mini-code@example.test"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Mini Code"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    tracked_file = repository / "README.md"
    tracked_file.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    tracked_file.write_text("new\n", encoding="utf-8")

    result = inspect_git_diff(workspace)

    assert result.returncode == 2
    assert result.diff == ""
    assert "refusing to inspect parent repository diff" in result.stderr


def test_inspect_git_diff_reports_untracked_files(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    new_file = workspace / "src" / "test" / "java" / "ExampleTest.java"
    new_file.parent.mkdir(parents=True)
    new_file.write_text("class ExampleTest {}\n", encoding="utf-8")

    result = inspect_git_diff(workspace)

    assert result.returncode == 0
    assert result.diff == ""
    assert "?? src/test/java/ExampleTest.java" in result.status_short
    assert result.untracked_files == ["src/test/java/ExampleTest.java"]
