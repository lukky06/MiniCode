import json

from minicode_harness.context import build_observation
from minicode_harness.tools import FileReadResult, read_artifact


def _source_lines(count: int) -> list[str]:
    return [f"source line {index}" for index in range(1, count + 1)]


def test_read_file_artifact_contains_raw_source_with_source_line_numbers(tmp_path) -> None:
    lines = _source_lines(801)
    result = FileReadResult(
        path="src/large.py",
        content="\n".join(lines),
        start_line=1,
        end_line=801,
        total_lines=801,
    )

    observation, _ = build_observation(
        tool_call_id="source",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "src/large.py"},
        result=result,
        artifact_dir=tmp_path,
    )
    artifact_path = tmp_path / (observation.artifact_path or "")
    artifact_text = artifact_path.read_text(encoding="utf-8")
    selected = read_artifact(
        tmp_path,
        observation.artifact_path or "",
        start_line=180,
        end_line=220,
    )

    assert artifact_text.splitlines()[0] == "source line 1"
    assert '"content"' not in artifact_text
    assert selected.content.splitlines() == lines[179:220]
    assert selected.start_line == 180
    assert selected.end_line == 220


def test_read_artifact_output_uses_raw_text_when_recompressed(tmp_path) -> None:
    source = "\n".join(_source_lines(801))
    source_path = tmp_path / "source.txt"
    source_path.write_text(source, encoding="utf-8", newline="")
    result = read_artifact(tmp_path, "source.txt")

    observation, _ = build_observation(
        tool_call_id="artifact",
        tool_name="read",
        tool_arguments={"source": "artifact", "target": "source.txt"},
        result=result,
        artifact_dir=tmp_path,
    )

    stored = (tmp_path / (observation.artifact_path or "")).read_text(encoding="utf-8")
    assert stored == source


def test_text_artifact_preserves_unicode_crlf_and_blank_lines(tmp_path) -> None:
    lines = ["α", "", "β", *[f"行 {index}" for index in range(798)]]
    content = "\r\n".join(lines)

    observation, _ = build_observation(
        tool_call_id="unicode",
        tool_name="read",
        tool_arguments={"source": "workspace", "target": "unicode.txt"},
        result={
            "path": "unicode.txt",
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "total_lines": len(lines),
        },
        artifact_dir=tmp_path,
    )

    stored = (tmp_path / (observation.artifact_path or "")).read_bytes().decode("utf-8")
    assert stored == content
    assert stored.startswith("α\r\n\r\nβ")


def test_non_read_tool_artifact_keeps_structured_payload(tmp_path) -> None:
    observation, _ = build_observation(
        tool_call_id="command",
        tool_name="run_command",
        result={
            "command": "python -m pytest -q",
            "returncode": 1,
            "stdout": "x" * 21_000,
            "stderr": "",
        },
        artifact_dir=tmp_path,
    )

    stored = (tmp_path / (observation.artifact_path or "")).read_text(encoding="utf-8")
    payload = json.loads(stored)
    assert payload["command"] == "python -m pytest -q"
    assert payload["returncode"] == 1
