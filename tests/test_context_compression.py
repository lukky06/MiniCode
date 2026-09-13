import json

from minicode_harness.context import (
    ContextPreparer,
    TokenBudget,
    build_observation,
    validate_message_protocol,
)
from minicode_harness.context.limits import (
    COMMAND_OUTPUT_CHAR_LIMIT,
    FAILED_COMMAND_OUTPUT_CHAR_LIMIT,
    FAILED_COMMAND_PREVIEW_CHAR_LIMIT,
    FILE_SEARCH_RESULT_LIMIT,
    OBSERVATION_TOKEN_LIMIT,
    READ_FILE_LINE_LIMIT,
    SEARCH_MATCH_LIMIT,
)
from minicode_harness.tools import FileReadResult


def _lines(count: int) -> str:
    return "\n".join(f"line {index}" for index in range(1, count + 1))


def test_partial_read_uses_returned_line_count_not_source_total(tmp_path) -> None:
    result = FileReadResult(
        path="large.py",
        content=_lines(20),
        start_line=100,
        end_line=119,
        total_lines=2000,
    )

    observation, events = build_observation(
        tool_call_id="partial",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "large.py"},
        result=result,
        artifact_dir=tmp_path,
    )

    assert observation.is_truncated is False
    assert observation.artifact_path is None
    assert observation.lossiness == "none"
    assert events == []


def test_read_limit_constants_match_bounded_expansion() -> None:
    assert OBSERVATION_TOKEN_LIMIT == 12_000
    assert READ_FILE_LINE_LIMIT == 800
    assert COMMAND_OUTPUT_CHAR_LIMIT == 20_000


def test_800_returned_lines_remain_inline(tmp_path) -> None:
    result = FileReadResult(
        path="large.py",
        content=_lines(800),
        start_line=1,
        end_line=800,
        total_lines=2000,
    )

    observation, events = build_observation(
        tool_call_id="boundary",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "large.py"},
        result=result,
        artifact_dir=tmp_path,
    )

    assert observation.is_truncated is False
    assert observation.artifact_path is None
    assert events == []


def test_801_returned_lines_are_compressed(tmp_path) -> None:
    result = FileReadResult(
        path="large.py",
        content=_lines(801),
        start_line=1,
        end_line=801,
        total_lines=2000,
    )

    observation, events = build_observation(
        tool_call_id="large",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "large.py"},
        result=result,
        artifact_dir=tmp_path,
    )

    assert observation.is_truncated is True
    assert observation.artifact_path == "read_large.txt"
    assert events[0].details["reasons"] == ["workspace_read_over_800_returned_lines"]


def test_complete_801_line_file_read_is_compressed(tmp_path) -> None:
    result = FileReadResult(
        path="large.py",
        content=_lines(801),
        start_line=1,
        end_line=801,
        total_lines=801,
    )

    observation, _ = build_observation(
        tool_call_id="complete",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "large.py"},
        result=result,
        artifact_dir=tmp_path,
    )

    assert observation.is_truncated is True


def test_returned_line_count_does_not_require_range_metadata(tmp_path) -> None:
    observation, events = build_observation(
        tool_call_id="missing_range",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "large.py"},
        result={
            "path": "large.py",
            "content": _lines(801),
            "total_lines": 2000,
        },
        artifact_dir=tmp_path,
    )

    assert observation.is_truncated is True
    assert events[0].details["reasons"] == ["workspace_read_over_800_returned_lines"]


def test_file_search_over_50_results_is_compressed_to_an_artifact(tmp_path) -> None:
    files = [f"src/package_{index}/Module{index}.java" for index in range(FILE_SEARCH_RESULT_LIMIT + 1)]
    details = [
        {"path": path, "size_bytes": 1000 + index, "estimated_lines": 20 + index}
        for index, path in enumerate(files)
    ]

    observation, events = build_observation(
        tool_call_id="large_files",
        tool_name="search",
        tool_arguments={"kind": "files", "query": "**/*.java", "path": "src"},
        result={
            "root": "src",
            "pattern": "**/*.java",
            "files": files,
            "file_details": details,
            "truncated": False,
            "scanned_entries": len(files),
            "exists": True,
            "is_directory": True,
        },
        artifact_dir=tmp_path,
    )

    assert observation.is_truncated is True
    assert observation.artifact_path == "search_large_files.txt"
    assert events[0].details["reasons"] == [
        f"file_search_over_{FILE_SEARCH_RESULT_LIMIT}_files"
    ]
    compact = json.loads(observation.content)["output_preview"]
    assert compact["match_count"] == FILE_SEARCH_RESULT_LIMIT + 1
    assert len(compact["files"]) == 20
    assert compact["omitted_files"] == FILE_SEARCH_RESULT_LIMIT + 1 - 20
    assert "file_details" not in compact
    assert (tmp_path / observation.artifact_path).exists()


def test_search_over_50_matches_is_compressed_to_an_artifact(tmp_path) -> None:
    matches = [
        {
            "path": f"src/module_{index % 14}.py",
            "line": index + 1,
            "text": f"match {index}",
        }
        for index in range(SEARCH_MATCH_LIMIT + 1)
    ]

    observation, events = build_observation(
        tool_call_id="large_search",
        tool_name="search",
        tool_arguments={"source": "workspace", "kind": "text", "query": "target"},
        result={
            "query": "target",
            "matches": matches,
            "truncated": False,
        },
        artifact_dir=tmp_path,
    )

    assert observation.is_truncated is True
    assert observation.artifact_path == "search_large_search.txt"
    assert events[0].reason == "tool_output"
    assert events[0].details["reasons"] == ["text_search_over_50_matches"]
    assert (tmp_path / observation.artifact_path).exists()


def test_command_output_uses_20k_character_limit(tmp_path) -> None:
    small, small_events = build_observation(
        tool_call_id="small_command",
        tool_name="run_command",
        result={
            "command": "pytest -q",
            "returncode": 0,
            "stdout": "x" * 18_000,
            "stderr": "",
        },
        artifact_dir=tmp_path,
    )
    large, large_events = build_observation(
        tool_call_id="large_command",
        tool_name="run_command",
        result={
            "command": "pytest -q",
            "returncode": 0,
            "stdout": "x" * 21_000,
            "stderr": "",
        },
        artifact_dir=tmp_path,
    )

    assert small.is_truncated is False
    assert small_events == []
    assert large.is_truncated is True
    assert "run_command_over_20000_chars" in large_events[0].details["reasons"]


def test_failed_command_output_is_compressed_before_the_20k_success_limit(
    tmp_path,
) -> None:
    stdout = (
        "================================== FAILURES ===================================\n"
        "________________________ test_search_filters_by_status ________________________\n"
        "E   TypeError: Ticket.__init__() got an unexpected keyword argument 'ticket_id'\n"
        + "\n".join(
            f"FAILED diagnostic context {index}: repeated fixture failure details"
            for index in range(220)
        )
        + "\n=========================== short test summary info ===========================\n"
        "FAILED tests/application/test_search_tickets.py::test_search_filters_by_status\n"
        "10 failed, 1 passed in 0.42s\n"
    )
    result = {
        "argv": ["python", "-m", "pytest", "-q", "tests/application/test_search_tickets.py"],
        "command": "python -m pytest -q tests/application/test_search_tickets.py",
        "returncode": 1,
        "stdout": stdout,
        "stderr": "",
        "duration_seconds": 0.42,
        "timed_out": False,
    }
    raw_text = json.dumps(result, ensure_ascii=False, indent=2)

    observation, events = build_observation(
        tool_call_id="failed_command",
        tool_name="run_command",
        result=result,
        artifact_dir=tmp_path,
    )

    assert FAILED_COMMAND_OUTPUT_CHAR_LIMIT < len(raw_text) < COMMAND_OUTPUT_CHAR_LIMIT
    assert observation.is_truncated is True
    assert observation.artifact_path == "run_command_failed_command.txt"
    assert events[0].details["reasons"] == [
        f"failed_run_command_over_{FAILED_COMMAND_OUTPUT_CHAR_LIMIT}_chars"
    ]
    compact = json.loads(observation.content)["output_preview"]
    assert compact["returncode"] == 1
    assert "TypeError" in compact["stdout_preview"]
    assert "ticket_id" in compact["stdout_preview"]
    assert "10 failed, 1 passed" in compact["stdout_preview"]
    assert len(compact["stdout_preview"]) <= FAILED_COMMAND_PREVIEW_CHAR_LIMIT
    assert observation.token_estimate < 1_500
    artifact = tmp_path / observation.artifact_path
    assert artifact.exists()
    assert "diagnostic context 219" in artifact.read_text(encoding="utf-8")


def test_failed_command_entry_compression_prevents_minimum_prompt_overflow(
    tmp_path,
) -> None:
    stdout = (
        "E   TypeError: Ticket.__init__() got an unexpected keyword argument 'ticket_id'\n"
        + "\n".join(
            f"FAILED repeated failure detail {index} with enough context to consume budget"
            for index in range(180)
        )
        + "\n10 failed, 1 passed in 0.42s\n"
    )
    result = {
        "argv": ["python", "-m", "pytest", "-q", "tests/application/test_search_tickets.py"],
        "command": "python -m pytest -q tests/application/test_search_tickets.py",
        "returncode": 1,
        "stdout": stdout,
        "stderr": "",
    }
    raw_text = json.dumps(result, ensure_ascii=False, indent=2)
    observation, _ = build_observation(
        tool_call_id="latest_failure",
        tool_name="run_command",
        result=result,
        artifact_dir=tmp_path,
    )
    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "latest_failure",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"argv": result["argv"]}),
                },
            }
        ],
    }
    base_messages = [
        {"role": "user", "content": "active task " + "u" * 2_000},
        assistant_tool_call,
    ]
    system_messages = [{"role": "system", "content": "s" * 5_000}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "t" * 8_000,
                "parameters": {"type": "object"},
            },
        }
    ]
    preparer = ContextPreparer(
        TokenBudget(
            context_budget=7_500,
            reserved_output=1_000,
            soft_limit=0.8,
            hard_limit=0.95,
        )
    )

    raw_prepared = preparer.prepare(
        system_messages=system_messages,
        messages=[
            *base_messages,
            {
                "role": "tool",
                "tool_call_id": "latest_failure",
                "content": raw_text,
            },
        ],
        tools=tools,
    )

    assert raw_prepared.token_estimate <= preparer.budget.hard_token_limit
    assert raw_prepared.compression_events

    prepared = preparer.prepare(
        system_messages=system_messages,
        messages=[
            *base_messages,
            {
                "role": "tool",
                "tool_call_id": "latest_failure",
                "content": observation.content,
            },
        ],
        tools=tools,
    )

    assert prepared.token_estimate <= preparer.budget.hard_token_limit
    assert validate_message_protocol(prepared.request.messages) == []
    latest_result = prepared.request.messages[-1]
    assert latest_result["role"] == "tool"
    assert "TypeError" in latest_result["content"]
    assert "ticket_id" in latest_result["content"]
    assert "10 failed, 1 passed" in latest_result["content"]


def test_observation_token_limit_remains_a_hard_cap(tmp_path) -> None:
    below, below_events = build_observation(
        tool_call_id="below_token_limit",
        tool_name="custom_tool",
        result={"content": "x" * 47_000},
        artifact_dir=tmp_path,
    )
    above, above_events = build_observation(
        tool_call_id="above_token_limit",
        tool_name="custom_tool",
        result={"content": "x" * 49_000},
        artifact_dir=tmp_path,
    )

    assert below.is_truncated is False
    assert below_events == []
    assert above.is_truncated is True
    assert "observation_over_12000_tokens" in above_events[0].details["reasons"]


def test_large_read_artifact_reuses_original_path_without_recursive_copy(tmp_path) -> None:
    original = tmp_path / "read_file_large.txt"
    original.write_text(_lines(801), encoding="utf-8")

    observation, events = build_observation(
        tool_call_id="read_again",
        tool_name="read",
        result=FileReadResult(
            path="read_file_large.txt",
            content=original.read_text(encoding="utf-8"),
            start_line=1,
            end_line=801,
            total_lines=801,
        ),
        artifact_dir=tmp_path,
        tool_arguments={"source": "artifact", "target": "read_file_large.txt"},
    )

    assert observation.is_truncated is True
    assert observation.artifact_path == "read_file_large.txt"
    assert events[0].details["artifact_path"] == "read_file_large.txt"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["read_file_large.txt"]
    assert "read_read_again.txt" not in observation.content
