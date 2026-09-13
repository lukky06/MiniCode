from pathlib import Path

from minicode_harness.context import ProjectContextCache
from minicode_harness.models import NormalizedToolCall
from minicode_harness.tools import FileReadResult


def _read_result(path: str, content: str) -> FileReadResult:
    lines = content.splitlines()
    return FileReadResult(
        path=path,
        content=content,
        start_line=1,
        end_line=max(1, len(lines)),
        total_lines=max(1, len(lines)),
    )


def _workspace_read_call(path: str) -> NormalizedToolCall:
    return NormalizedToolCall(
        id="call_1",
        name="read",
        arguments={"source": "workspace", "target": path},
    )


def test_project_context_cache_reuses_fresh_workspace_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "A.java").write_text("class A {}\n", encoding="utf-8")
    cache = ProjectContextCache(tmp_path / "data")
    result = _read_result("A.java", "class A {}\n")

    cache.save_workspace_read_result(
        workspace=workspace,
        run_id="run_1",
        step=1,
        tool_call=_workspace_read_call("A.java"),
        result=result,
    )
    cached = cache.lookup_workspace_read(
        workspace=workspace,
        arguments={"source": "workspace", "target": "A.java"},
    )

    assert cached is not None
    assert cached.tool_name == "read"
    assert cached.run_id == "run_1"
    assert cached.result_payload["content"] == "class A {}\n"


def test_project_context_cache_invalidates_changed_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "A.java"
    source.write_text("class A {}\n", encoding="utf-8")
    cache = ProjectContextCache(tmp_path / "data")
    cache.save_workspace_read_result(
        workspace=workspace,
        run_id="run_1",
        step=1,
        tool_call=_workspace_read_call("A.java"),
        result=_read_result("A.java", "class A {}\n"),
    )

    source.write_text("class A { int value; }\n", encoding="utf-8")

    assert cache.lookup_workspace_read(
        workspace=workspace,
        arguments={"source": "workspace", "target": "A.java"},
    ) is None


def test_project_context_cache_has_no_fact_or_source_index_api(tmp_path: Path) -> None:
    cache = ProjectContextCache(tmp_path / "data")

    assert not hasattr(cache, "upsert_evidence_facts")
    assert not hasattr(cache, "load_fact_profile")
    assert not hasattr(cache, "render_fact_profile")
    assert not hasattr(cache, "save_source_index")
    assert not hasattr(cache, "lookup_source_index")
