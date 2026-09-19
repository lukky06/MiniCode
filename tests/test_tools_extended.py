import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import Any

import pytest

import minicode_harness.tools.registry as registry_module
from minicode_harness.policy import RiskLevel
from minicode_harness.tools import (
    DockerCommandExecutor,
    StaleWriteError,
    ToolRegistry,
    compact_tool_schema_for_provider,
    edit_file,
    find_files,
    preview_edit_file,
    preview_write_file,
    read_file,
    search_text,
    write_file,
)
from minicode_harness.context.token import estimate_tokens
from minicode_harness.tools.registry import ToolDefinition


def test_edit_file_replaces_one_unique_block(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    result = edit_file(workspace, "app.py", "VALUE = 1", "VALUE = 2")

    assert result.changed
    assert result.replacements == 1
    assert source.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_edit_file_preview_contains_unique_match_and_diff(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    preview = preview_edit_file(workspace, "app.py", "VALUE = 1", "VALUE = 2")

    assert preview.matches == 1
    assert "-VALUE = 1" in preview.diff
    assert "+VALUE = 2" in preview.diff
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_edit_file_rejects_ambiguous_match_without_writing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("x = 1\nx = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="matched 2 locations"):
        edit_file(workspace, "app.py", "x = 1", "x = 2")

    assert source.read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_registry_serializes_concurrent_edits_to_same_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("FIRST = 1\nSECOND = 1\n", encoding="utf-8")
    registry = ToolRegistry(str(workspace), enable_write=True)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    call_lock = Lock()
    call_count = 0

    def tracked_edit(
        workspace_path: Path | str,
        path: Path | str,
        old_text: str,
        new_text: str,
    ) -> Any:
        nonlocal call_count
        with call_lock:
            call_count += 1
            call_number = call_count
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        return edit_file(workspace_path, path, old_text, new_text)

    monkeypatch.setattr(registry_module, "edit_file", tracked_edit)
    first_admission = registry.admit(
        "edit",
        {"path": "app.py", "old_text": "FIRST = 1", "new_text": "FIRST = 2"},
    )
    second_admission = registry.admit(
        "edit",
        {"path": "app.py", "old_text": "SECOND = 1", "new_text": "SECOND = 2"},
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(registry.execute_admitted, first_admission)
        assert first_entered.wait(timeout=1)
        second = pool.submit(registry.execute_admitted, second_admission)
        try:
            assert not second_entered.wait(timeout=0.1)
        finally:
            release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert second_entered.is_set()
    assert source.read_text(encoding="utf-8") == "FIRST = 2\nSECOND = 2\n"


def test_registry_keeps_edits_to_different_files_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first_source = workspace / "first.py"
    second_source = workspace / "second.py"
    first_source.write_text("VALUE = 1\n", encoding="utf-8")
    second_source.write_text("VALUE = 1\n", encoding="utf-8")
    registry = ToolRegistry(str(workspace), enable_write=True)
    barrier = Barrier(2)

    def synchronized_edit(
        workspace_path: Path | str,
        path: Path | str,
        old_text: str,
        new_text: str,
    ) -> Any:
        barrier.wait(timeout=1)
        return edit_file(workspace_path, path, old_text, new_text)

    monkeypatch.setattr(registry_module, "edit_file", synchronized_edit)
    admissions = [
        registry.admit(
            "edit",
            {"path": path, "old_text": "VALUE = 1", "new_text": "VALUE = 2"},
        )
        for path in ("first.py", "second.py")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(registry.execute_admitted, admission)
            for admission in admissions
        ]
        for future in futures:
            future.result(timeout=2)

    assert first_source.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert second_source.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_find_files_and_regex_search_are_bounded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "a.py").write_text("Value = 10\n", encoding="utf-8")
    (workspace / "src" / "b.txt").write_text("value = 20\n", encoding="utf-8")

    found = find_files(workspace, pattern="**/*.py", max_results=10)
    searched = search_text(
        workspace,
        r"value\s*=\s*\d+",
        "src",
        use_regex=True,
        case_sensitive=False,
        file_glob="*.py",
    )

    assert found.files == ["src/a.py"]
    assert [match.path for match in searched.matches] == ["src/a.py"]


def test_write_file_preview_requires_explicit_overwrite_and_contains_diff(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=true"):
        preview_write_file(workspace, "app.py", "VALUE = 2\n")

    preview = preview_write_file(
        workspace,
        "app.py",
        "VALUE = 2\n",
        overwrite=True,
    )

    assert preview.created is False
    assert preview.overwrite is True
    assert preview.old_bytes == len("VALUE = 1\n".encode("utf-8"))
    assert preview.new_bytes == len("VALUE = 2\n".encode("utf-8"))
    assert preview.changed_lines == 2
    assert "-VALUE = 1" in preview.diff
    assert "+VALUE = 2" in preview.diff
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_read_hash_guards_atomic_whole_file_overwrite(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    read = read_file(workspace, "app.py")
    assert read.content_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()

    result = write_file(
        workspace,
        "app.py",
        "VALUE = 2\n",
        overwrite=True,
        expected_sha256=read.content_sha256,
    )

    assert result.overwritten is True
    assert source.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_whole_file_overwrite_rejects_changed_read_version(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    read = read_file(workspace, "app.py")
    source.write_text("VALUE = external\n", encoding="utf-8")

    with pytest.raises(StaleWriteError, match="changed after it was read"):
        write_file(
            workspace,
            "app.py",
            "VALUE = agent\n",
            overwrite=True,
            expected_sha256=read.content_sha256,
        )

    assert source.read_text(encoding="utf-8") == "VALUE = external\n"


def test_registry_exposes_precise_edit_with_medium_risk(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(str(workspace), enable_write=True)
    names = {schema["function"]["name"] for schema in registry.schemas()}

    assert {"read", "search", "edit"}.issubset(names)
    assert "find_files" not in names
    assert "edit_file" not in names
    assert registry.risk_level("edit") == RiskLevel.MEDIUM

    source = workspace / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    preview = registry.preview_admitted(
        registry.admit(
            "edit",
            {"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"},
        )
    )
    assert preview["matches"] == 1
    assert "-VALUE = 1" in preview["diff"]


def test_registry_keeps_worktree_schema_stable_and_returns_unavailable_without_handler(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plain = ToolRegistry(str(workspace), enable_write=True)
    enabled = ToolRegistry(
        str(workspace),
        enable_write=True,
        worktree_worker_handler=lambda task: {"task": task},
    )

    plain_names = {schema["function"]["name"] for schema in plain.schemas()}
    enabled_names = {schema["function"]["name"] for schema in enabled.schemas()}

    assert "delegate_worktree" in plain_names
    assert "delegate_worktree" in enabled_names
    assert plain.execute_admitted(
        plain.admit("delegate_worktree", {"task": "修复测试"})
    ) == {
        "status": "unavailable",
        "reason": "worktree_workers_disabled",
    }
    assert enabled.risk_level("delegate_worktree") == RiskLevel.MEDIUM


def test_compact_tool_schema_for_provider_preserves_validation_contract() -> None:
    schema = {
        "title": "ReadFileArgs",
        "description": "Arguments for read_file.",
        "type": "object",
        "properties": {
            "path": {
                "title": "Path",
                "description": "Workspace path.",
                "type": "string",
            },
            "start_line": {
                "title": "Start Line",
                "default": None,
                "anyOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "null"},
                ],
            },
            "overwrite": {
                "title": "Overwrite",
                "default": False,
                "type": "boolean",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    original = deepcopy(schema)

    compacted = compact_tool_schema_for_provider(schema)

    assert schema == original
    assert "title" not in compacted
    assert "description" not in compacted
    assert compacted["properties"]["path"]["description"] == "Workspace path."
    assert "title" not in compacted["properties"]["path"]
    assert "default" not in compacted["properties"]["start_line"]
    assert compacted["properties"]["start_line"]["anyOf"] == [
        {"type": "integer", "minimum": 1},
        {"type": "null"},
    ]
    assert compacted["properties"]["overwrite"]["default"] is False
    assert compacted["required"] == ["path"]
    assert compacted["additionalProperties"] is False


def test_normal_write_registry_exposes_exact_stable_eleven_tool_surface(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(
        str(workspace),
        enable_write=True,
        task_create_handler=lambda tasks: {"created": tasks},
        task_update_handler=lambda updates: {"updated": updates},
        task_list_handler=lambda: {"tasks": []},
    )

    names = [schema["function"]["name"] for schema in registry.schemas()]

    assert set(names) == {
        "read",
        "search",
        "edit",
        "write",
        "apply_patch",
        "run_command",
        "task",
        "delegate_task",
        "delegate_worktree",
        "runtime_task_status",
        "runtime_task_stop",
    }
    assert not {
        "read_file",
        "read_artifact",
        "read_memory",
        "load_skill",
        "inspect_git_diff",
        "find_files",
        "search_text",
        "edit_file",
        "write_file",
        "task_create",
        "task_update",
        "task_list",
    }.intersection(names)
    search_schema = next(
        schema for schema in registry.schemas() if schema["function"]["name"] == "search"
    )
    source_schema = search_schema["function"]["parameters"]["properties"]["source"]
    assert source_schema["enum"] == ["workspace", "artifact"]
    assert source_schema["default"] == "workspace"


def test_read_and_search_schemas_explain_artifact_followup(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(
        str(workspace),
        enable_write=False,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    schemas = {
        schema["function"]["name"]: schema["function"]
        for schema in registry.schemas()
    }

    read_schema = schemas["read"]
    read_properties = read_schema["parameters"]["properties"]
    search_schema = schemas["search"]
    search_properties = search_schema["parameters"]["properties"]

    assert "artifact_path" in read_properties["target"]["description"]
    assert "artifact_path" in search_properties["path"]["description"]


def test_registry_compacts_provider_schemas_by_at_least_ten_percent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(
        str(workspace),
        enable_write=True,
        artifact_dir=str(tmp_path / "artifacts"),
        subagent_handler=lambda task: task,
        memory_topic_reader=lambda topic: {"topic": topic},
    )
    raw_schemas: list[dict[str, Any]] = []
    for tool in registry._tools.values():
        parameters = tool.parameters
        if parameters is None:
            parameters = (
                tool.args_model.model_json_schema()
                if tool.args_model is not None
                else {"type": "object", "properties": {}}
            )
        raw_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )

    compacted_schemas = registry.schemas()
    raw_tokens = estimate_tokens(
        json.dumps(raw_schemas, ensure_ascii=False, separators=(",", ":"))
    )
    compacted_tokens = estimate_tokens(
        json.dumps(compacted_schemas, ensure_ascii=False, separators=(",", ":"))
    )

    assert len(compacted_schemas) == 11
    assert compacted_tokens <= raw_tokens * 0.90
    assert compacted_tokens <= 1_700
    for schema in compacted_schemas:
        parameters = schema["function"]["parameters"]
        assert "description" not in parameters
        _assert_provider_schema_has_no_display_only_fields(parameters)


def _assert_provider_schema_has_no_display_only_fields(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _assert_provider_schema_has_no_display_only_fields(item)
        return
    if not isinstance(value, dict):
        return
    assert "title" not in value
    assert not ("default" in value and value["default"] is None)
    for item in value.values():
        _assert_provider_schema_has_no_display_only_fields(item)


def test_admission_validates_once_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    registry = ToolRegistry(str(workspace))
    calls = 0
    original_validate = ToolDefinition.validate

    def counting_validate(self, arguments):
        nonlocal calls
        calls += 1
        return original_validate(self, arguments)

    monkeypatch.setattr(ToolDefinition, "validate", counting_validate)

    admission = registry.admit(
        "read",
        {"source": "workspace", "target": "README.md"},
    )
    result = registry.execute_admitted(admission)

    assert calls == 1
    assert result.path == "README.md"


def test_request_user_input_validates_bounded_choices(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-user-input"
    workspace.mkdir()
    calls = []
    registry = ToolRegistry(
        str(workspace),
        request_user_input_handler=lambda question, options: calls.append((question, options)) or {
            "selected_index": 1,
            "selected_label": options[1]["label"],
        },
    )

    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert "request_user_input" in names
    result = registry.execute_admitted(
        registry.admit(
            "request_user_input",
            {
                "question": "Which validation strategy should the plan use?",
                "options": [
                    {"label": "strict", "description": "Reject stale callers"},
                    {"label": "current", "description": "Use the current contract"},
                ],
            },
        )
    )
    assert result["selected_label"] == "current"
    assert calls[0][0].startswith("Which validation")

    with pytest.raises(Exception):
        registry.admit(
            "request_user_input",
            {"question": "Too few", "options": [{"label": "only"}]},
        )


def test_docker_command_schema_tells_model_about_linux_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry(
        str(workspace),
        enable_write=True,
        command_executor=DockerCommandExecutor(image="python:3.11-slim"),
    )

    run_command = next(
        schema["function"]
        for schema in registry.schemas()
        if schema["function"]["name"] == "run_command"
    )

    assert "Linux Docker container" in run_command["description"]
    assert "/workspace" in run_command["description"]


def test_registry_can_disable_command_without_disabling_write_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    default_registry = ToolRegistry(str(workspace), enable_write=True)
    disabled_registry = ToolRegistry(
        str(workspace),
        enable_write=True,
        enable_command=False,
    )
    default_names = {
        schema["function"]["name"] for schema in default_registry.schemas()
    }
    disabled_names = {
        schema["function"]["name"] for schema in disabled_registry.schemas()
    }

    assert "run_command" in default_names
    assert {"edit", "apply_patch", "write"}.issubset(disabled_names)
    assert "run_command" not in disabled_names
