import json

from minicode_harness.context import (
    COMPACTED_EXECUTION_HEADING,
    compact_execution_history,
    compact_execution_history_with_details,
    group_messages,
)


def _tool_group(name, arguments, result, *, call_id="call_1"):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]


def test_compaction_keeps_workspace_reads_and_discards_searches_and_free_form_messages():
    messages = [
        {"role": "user", "content": "必须只修改一个文件"},
        {"role": "assistant", "content": "I think the bug is here."},
        *_tool_group(
            "read",
            {"source": "workspace", "target": "src/a.py"},
            {"status": "ok", "content": "secret code"},
        ),
        *_tool_group(
            "search",
            {"source": "workspace", "kind": "text", "path": ".", "query": "value"},
            {"status": "ok", "matches": ["src/a.py:1"]},
            call_id="call_2",
        ),
    ]

    rendered = compact_execution_history(group_messages(messages))

    assert "- read | src/a.py" in rendered
    assert "secret code" not in rendered
    assert "- search |" not in rendered
    assert "value" not in rendered
    assert "I think the bug is here" not in rendered


def test_compaction_keeps_write_state_without_content_or_patch():
    groups = group_messages(
        _tool_group(
            "apply_patch",
            {"path": "src/a.py", "patch": "very private patch body"},
            {"status": "success", "path": "src/a.py"},
        )
    )

    rendered = compact_execution_history(groups)

    assert rendered.startswith(COMPACTED_EXECUTION_HEADING)
    assert "- modified | src/a.py" in rendered
    assert "apply_patch" not in rendered
    assert "very private patch body" not in rendered


def test_compaction_keeps_failed_write_result():
    groups = group_messages(
        _tool_group(
            "write",
            {"path": "src/a.py"},
            {
                "status": "failed",
                "message": "permission denied",
            },
        )
    )

    rendered = compact_execution_history(groups)

    assert "write | src/a.py" in rendered
    assert "status: failed" in rendered
    assert "permission denied" in rendered


def test_compaction_keeps_command_argv_returncode_preview_and_artifact():
    groups = group_messages(
        _tool_group(
            "run_command",
            {"argv": ["python", "-m", "pytest", "tests/test_a.py", "-q"]},
            {
                "status": "failed",
                "returncode": 1,
                "timed_out": False,
                "message": "1 failed, 3 passed",
                "artifact_path": "runs/artifacts/pytest.txt",
            },
        )
    )

    rendered = compact_execution_history(groups)

    assert "python -m pytest tests/test_a.py -q" in rendered
    assert "status: failed" in rendered
    assert "returncode: 1" in rendered
    assert "timed_out: false" in rendered
    assert "result: 1 failed, 3 passed" in rendered
    assert "artifact: runs/artifacts/pytest.txt" in rendered


def test_failed_command_preview_keeps_diagnostic_lines_from_long_output():
    groups = group_messages(
        _tool_group(
            "run_command",
            {"argv": ["python", "-m", "unittest", "tests.test_cache"]},
            {
                "status": "command_failed",
                "returncode": 1,
                "stdout": "\n".join(
                    [
                        "setup detail " + ("x" * 120),
                        "more unrelated output " + ("y" * 120),
                        "assert cache.deleted_keys == ['ticket:tenant-a:ticket-1']",
                        (
                            "AssertionError: assert ['ticket:ticket-1'] "
                            "== ['ticket:tenant-a:ticket-1']"
                        ),
                        "FAILED tests.test_cache.CacheTests.test_tenant_key",
                        "1 failed",
                    ]
                ),
            },
        )
    )

    rendered = compact_execution_history(groups)

    assert "AssertionError" in rendered
    assert "ticket:ticket-1" in rendered
    assert "ticket:tenant-a:ticket-1" in rendered
    assert "more unrelated output" not in rendered


def test_compaction_keeps_only_path_for_successful_read_with_error_words_in_content():
    groups = group_messages(
        _tool_group(
            "read",
            {"source": "workspace", "target": "src/domain.py"},
            {
                "path": "src/domain.py",
                "content": (
                    "raise ValueError('invalid input')\n"
                    "failure_timeout = 'documented domain term'\n"
                ),
            },
        )
    )

    rendered = compact_execution_history(groups)

    assert "- read | src/domain.py" in rendered
    assert "ValueError" not in rendered
    assert "failure_timeout" not in rendered


def test_compaction_drops_failed_read():
    groups = group_messages(
        _tool_group(
            "read",
            {"source": "workspace", "target": "src/missing.py"},
            {
                "message": "file is missing",
                "error_type": "FileNotFoundError",
            },
        )
    )

    assert compact_execution_history(groups) == ""


def test_compaction_drops_runtime_reuse_results():
    groups = group_messages(
        _tool_group(
            "run_command",
            {"argv": ["python", "-V"]},
            {"status": "duplicate_reused", "message": "use the previous result"},
        )
    )

    assert compact_execution_history(groups) == ""


def test_compaction_carries_forward_existing_execution_record():
    existing = {
        "role": "assistant",
        "content": (
            f"{COMPACTED_EXECUTION_HEADING}\n\n"
            "- modified | src/a.py"
        ),
    }
    new_command = _tool_group(
        "run_command",
        {"argv": ["python", "-m", "pytest"]},
        {"status": "passed", "returncode": 0},
    )

    rendered = compact_execution_history([[existing], new_command])

    assert "modified | src/a.py" in rendered
    assert "write_file | src/a.py" not in rendered
    assert "run_command | python -m pytest" in rendered


def test_compaction_drops_failed_non_write_non_command_tool():
    groups = group_messages(
        _tool_group(
            "mcp__example__lookup",
            {"name": "record"},
            {"status": "failed", "error": "remote unavailable"},
        )
    )

    assert compact_execution_history(groups) == ""


def test_compaction_reports_omitted_execution_entries():
    messages = []
    for index in range(70):
        messages.extend(
            _tool_group(
                "run_command",
                {"argv": ["python", "-m", "pytest", f"tests/test_{index}.py"]},
                {"status": "passed", "returncode": 0},
                call_id=f"call_{index}",
            )
        )

    rendered, omitted = compact_execution_history_with_details(group_messages(messages))

    assert rendered.startswith(COMPACTED_EXECUTION_HEADING)
    assert omitted == 6
    assert "tests/test_0.py" not in rendered
    assert "tests/test_69.py" in rendered


def test_compaction_drops_mcp_calls_regardless_of_effect_metadata():
    read_groups = group_messages(
        _tool_group(
            "mcp__tickets__get",
            {"id": "T-1"},
            {"status": "success", "id": "T-1", "body": "large recoverable payload"},
        )
    )
    create_groups = group_messages(
        _tool_group(
            "mcp__tickets__create",
            {"title": "incident"},
            {"status": "success", "ticket_id": "T-1024", "body": "private body"},
        )
    )
    effects = {
        "mcp__tickets__get": {"read_only": True, "destructive": False},
        "mcp__tickets__create": {"read_only": False, "destructive": False},
    }

    assert compact_execution_history(read_groups, tool_effects=effects) == ""
    assert compact_execution_history(create_groups, tool_effects=effects) == ""


def test_compaction_drops_failed_readonly_and_successful_unknown_tools():
    effects = {"mcp__docs__read": {"read_only": True, "destructive": False}}
    failed_read = group_messages(
        _tool_group(
            "mcp__docs__read",
            {"id": "D-1"},
            {"status": "failed", "message": "remote unavailable"},
        )
    )
    unknown_write = group_messages(
        _tool_group(
            "mcp__unknown__send",
            {"name": "notice"},
            {"status": "success", "id": "M-7"},
        )
    )

    assert compact_execution_history(failed_read, tool_effects=effects) == ""
    assert compact_execution_history(unknown_write, tool_effects=effects) == ""


def test_compaction_merges_writes_by_file_and_drops_noops():
    messages = [
        *_tool_group(
            "edit",
            {"path": "src/a.py"},
            {"path": "src/a.py", "changed": True},
            call_id="edit_a_1",
        ),
        *_tool_group(
            "apply_patch",
            {"path": "src/a.py"},
            {"files": ["src/a.py"]},
            call_id="patch_a",
        ),
        *_tool_group(
            "write",
            {"path": "src/b.py"},
            {"path": "src/b.py", "changed": True},
            call_id="write_b",
        ),
        *_tool_group(
            "edit",
            {"path": "src/noop.py"},
            {"path": "src/noop.py", "changed": False},
            call_id="noop",
        ),
    ]

    rendered = compact_execution_history(group_messages(messages))

    assert rendered.count("- modified | src/a.py") == 1
    assert rendered.count("- modified | src/b.py") == 1
    assert "src/noop.py" not in rendered
    assert "edit_file" not in rendered
    assert "apply_patch" not in rendered
    assert "write_file" not in rendered


def test_compaction_keeps_only_latest_result_for_same_command():
    command = {"argv": ["python", "-m", "pytest", "tests/test_a.py", "-q"]}
    messages = [
        *_tool_group(
            "run_command",
            command,
            {
                "returncode": 1,
                "stderr": "1 failed",
                "artifact_path": "run_command_failed.txt",
            },
            call_id="failed",
        ),
        *_tool_group(
            "run_command",
            command,
            {
                "returncode": 0,
                "stdout": "3 passed in 0.20s",
                "artifact_path": "run_command_passed.txt",
            },
            call_id="passed",
        ),
    ]

    rendered = compact_execution_history(group_messages(messages))

    assert rendered.count("run_command | python -m pytest tests/test_a.py -q") == 1
    assert "status: passed" in rendered
    assert "returncode: 0" in rendered
    assert "3 passed" in rendered
    assert "run_command_passed.txt" in rendered
    assert "run_command_failed.txt" not in rendered
    assert "1 failed" not in rendered


def test_compaction_keeps_only_workspace_path_for_large_read():
    groups = group_messages(
        _tool_group(
            "read",
            {"source": "workspace", "target": "src/large.py"},
            {
                "summary": "Read file src/large.py lines 1-1200 of 1200.",
                "artifact_path": "read_file_large.txt",
                "output_preview": {"path": "src/large.py"},
            },
        )
    )

    rendered = compact_execution_history(groups)

    assert "- read | src/large.py" in rendered
    assert "read_file_large.txt" not in rendered
    assert "output_preview" not in rendered


def test_compaction_prioritizes_modifications_and_verification_over_failed_reads():
    messages = [
        *_tool_group(
            "edit",
            {"path": "src/a.py"},
            {"path": "src/a.py", "changed": True},
            call_id="edit_a",
        ),
        *_tool_group(
            "edit",
            {"path": "tests/test_a.py"},
            {"path": "tests/test_a.py", "changed": True},
            call_id="edit_test",
        ),
        *_tool_group(
            "run_command",
            {"argv": ["python", "-m", "pytest", "tests/test_a.py", "-q"]},
            {"returncode": 0, "stdout": "1 passed in 0.10s"},
            call_id="verify",
        ),
    ]
    for index in range(20):
        messages.extend(
            _tool_group(
                "read",
                {"source": "workspace", "target": f"src/background_{index}.py"},
                {"status": "ok", "path": f"src/background_{index}.py", "content": "reloadable"},
                call_id=f"background_{index}",
            )
        )

    rendered, omitted = compact_execution_history_with_details(
        group_messages(messages),
        max_chars=750,
    )

    assert "modified | src/a.py" in rendered
    assert "modified | tests/test_a.py" in rendered
    assert "python -m pytest tests/test_a.py -q" in rendered
    assert "returncode: 0" in rendered
    assert "1 passed" in rendered
    assert omitted > 0


def test_repeated_compaction_retains_early_modifications_and_verification():
    initial = group_messages(
        [
            *_tool_group(
                "edit",
                {"path": "src/a.py"},
                {"path": "src/a.py", "changed": True},
                call_id="edit_a",
            ),
            *_tool_group(
                "edit",
                {"path": "tests/test_a.py"},
                {"path": "tests/test_a.py", "changed": True},
                call_id="edit_test",
            ),
            *_tool_group(
                "run_command",
                {"argv": ["python", "-m", "pytest", "tests/test_a.py", "-q"]},
                {"returncode": 0, "stdout": "1 passed in 0.10s"},
                call_id="verify",
            ),
        ]
    )
    rendered = compact_execution_history(initial)

    for index in range(25):
        existing = {"role": "assistant", "content": rendered}
        failed_read = _tool_group(
            "read",
            {"source": "workspace", "target": f"src/missing_{index}.py"},
            {"status": "failed", "message": "file is missing"},
            call_id=f"missing_{index}",
        )
        rendered, _ = compact_execution_history_with_details(
            [[existing], failed_read],
            max_chars=750,
        )

    assert "modified | src/a.py" in rendered
    assert "modified | tests/test_a.py" in rendered
    assert "python -m pytest tests/test_a.py -q" in rendered
    assert "returncode: 0" in rendered
    assert "1 passed" in rendered


def test_fifty_turn_compaction_retains_current_changes_and_latest_verification():
    messages = []
    for index in range(6):
        messages.extend(
            _tool_group(
                "read_file",
                {"path": f"src/background_{index}.py"},
                {"path": f"src/background_{index}.py", "content": "reloadable"},
                call_id=f"background_{index}",
            )
        )
    messages.extend(
        _tool_group(
            "edit",
            {"path": "src/service.py"},
            {"path": "src/service.py", "changed": True},
            call_id="turn_7_service",
        )
    )
    messages.extend(
        _tool_group(
            "edit",
            {"path": "tests/test_service.py"},
            {"path": "tests/test_service.py", "changed": True},
            call_id="turn_7_test",
        )
    )
    messages.extend(
        _tool_group(
            "run_command",
            {"argv": ["python", "-m", "pytest", "tests/test_service.py", "-q"]},
            {
                "returncode": 0,
                "stdout": "2 passed in 0.12s",
                "artifact_path": "run_command_turn_7.txt",
            },
            call_id="turn_7_verify",
        )
    )
    rendered = compact_execution_history(group_messages(messages))

    for turn in range(8, 51):
        existing = {"role": "assistant", "content": rendered}
        read_result = (
            {"status": "failed", "message": "optional path is missing"}
            if turn % 3 == 0
            else {"path": f"src/analysis_{turn}.py", "content": "reloadable"}
        )
        next_group = _tool_group(
            "read",
            {"source": "workspace", "target": f"src/analysis_{turn}.py"},
            read_result,
            call_id=f"turn_{turn}_read",
        )
        rendered, _ = compact_execution_history_with_details(
            [[existing], next_group],
            max_chars=900,
        )

    assert "modified | src/service.py" in rendered
    assert "modified | tests/test_service.py" in rendered
    assert "python -m pytest tests/test_service.py -q" in rendered
    assert "returncode: 0" in rendered
    assert "2 passed" in rendered
    assert "artifact: run_command_turn_7.txt" in rendered
    assert len(rendered) <= 900 + len(COMPACTED_EXECUTION_HEADING) + 2
