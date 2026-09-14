from typing import Any

import minicode_harness.tools.registry as registry_module
from minicode_harness.context import ContextPreparer, InspectedFile, RunState, TokenBudget
from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.storage import HarnessDataStore as ProjectMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.state import ApprovalDecision, CheckpointStore, StaticApprovalClient
from minicode_harness.tools import SearchResult
from minicode_harness.trace import TraceWriter


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def call_request(self, request) -> ModelResponse:
        self.calls.append((request.as_chat_messages(), request.tools))
        return self.responses.pop(0)


def _read_call(call_id: str, start: int | None = None, end: int | None = None) -> ModelResponse:
    arguments: dict[str, Any] = {"source": "workspace", "target": "source.py"}
    if start is not None:
        arguments["start_line"] = start
    if end is not None:
        arguments["end_line"] = end
    return ModelResponse(
        tool_calls=[
            NormalizedToolCall(
                id=call_id,
                name="read",
                arguments=arguments,
            )
        ]
    )


def _search_call(call_id: str, **overrides: Any) -> ModelResponse:
    arguments: dict[str, Any] = {
        "source": "workspace",
        "kind": "text",
        "query": "needle",
        "path": ".",
        "limit": 50,
        "use_regex": False,
        "case_sensitive": True,
        "file_glob": "*.py",
        "max_depth": 12,
    }
    arguments.update(overrides)
    return ModelResponse(
        tool_calls=[
            NormalizedToolCall(
                id=call_id,
                name="search",
                arguments=arguments,
            )
        ]
    )


def _make_workspace(tmp_path) -> Any:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("source.py").write_text(
        "\n".join(f"line {index} needle" for index in range(1, 301)),
        encoding="utf-8",
    )
    return workspace


def _compacting_preparer() -> ContextPreparer:
    return ContextPreparer(
        TokenBudget(
            context_budget=4_000,
            reserved_output=0,
            soft_limit=0.80,
            hard_limit=0.95,
        )
    )


def test_read_is_allowed_again_after_source_result_leaves_visible_history(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.read_file
    calls = 0

    def counting_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "read_file", counting_read)
    responses = [_read_call("read_original", 1, 200)]
    responses.extend(
        _search_call(f"search_{index}", query=f"line {index}")
        for index in range(1, 7)
    )
    responses.extend([_read_call("read_again", 1, 200), ModelResponse(final_text="done")])
    loop = AgentLoop(
        task="Inspect source.py",
        workspace=workspace,
        model_client=ScriptedModelClient(responses),
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        context_preparer=_compacting_preparer(),
        no_skills=True,
    )

    loop.run()

    # Once the original Tool Result leaves the visible history, Run-local reuse
    # no longer applies and the file is read again from the current workspace.
    assert calls == 2
    repeated = next(item for item in loop.observations if item.tool_call_id == "read_again")
    assert repeated.metadata.get("status") != "duplicate_reused"
    assert "line 1 needle" in repeated.content


def test_search_executes_again_after_source_result_leaves_visible_history(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.search_text
    calls = 0

    def counting_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "search_text", counting_search)
    responses = [_search_call("search_original")]
    responses.extend(
        _read_call(f"read_{index}", index * 20 + 1, index * 20 + 20)
        for index in range(7)
    )
    responses.extend([_search_call("search_again"), ModelResponse(final_text="done")])
    loop = AgentLoop(
        task="Search source.py",
        workspace=workspace,
        model_client=ScriptedModelClient(responses),
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        context_preparer=_compacting_preparer(),
        no_skills=True,
    )

    loop.run()

    assert calls == 2
    repeated = next(item for item in loop.observations if item.tool_call_id == "search_again")
    assert repeated.metadata.get("status") != "duplicate_reused"


def test_contained_read_range_is_reused_without_second_file_access(tmp_path, monkeypatch) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.read_file
    calls = 0

    def counting_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "read_file", counting_read)
    client = ScriptedModelClient(
        [_read_call("read_big", 1, 200), _read_call("read_small", 100, 150), ModelResponse(final_text="done")]
    )
    loop = AgentLoop(
        task="Inspect and fix source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    )

    result = loop.run()

    assert result.status == "completed"
    assert calls == 1
    duplicate = next(item for item in loop.observations if item.tool_call_id == "read_small")
    assert duplicate.metadata["status"] == "duplicate_reused"
    assert duplicate.metadata["requested_range"] == [100, 150]
    assert duplicate.metadata["covered_by"] == [1, 200]


def test_partially_overlapping_read_executes_and_records_overlap(tmp_path, monkeypatch) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.read_file
    calls = 0

    def counting_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "read_file", counting_read)
    client = ScriptedModelClient(
        [_read_call("read_first", 1, 100), _read_call("read_overlap", 80, 150), ModelResponse(final_text="done")]
    )
    loop = AgentLoop(
        task="Inspect source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    )

    loop.run()

    assert calls == 2
    overlap = next(item for item in loop.observations if item.tool_call_id == "read_overlap")
    assert overlap.metadata["overlap_detected"] is True


def test_workspace_write_invalidates_read_coverage(tmp_path, monkeypatch) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.read_file
    calls = 0

    def counting_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "read_file", counting_read)
    replacement = "\n".join(f"changed {index}" for index in range(1, 301))
    client = ScriptedModelClient(
        [
            _read_call("read_before"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": replacement,
                            "overwrite": True,
                        },
                    )
                ]
            ),
            _read_call("read_after", 1, 20),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Modify source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    )

    loop.run()

    assert calls == 2
    after = next(item for item in loop.observations if item.tool_call_id == "read_after")
    assert after.metadata.get("status") != "duplicate_reused"


def test_existing_whole_file_overwrite_requires_full_read_first(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="unsafe_overwrite",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "agent replacement\n",
                            "overwrite": True,
                        },
                    )
                ]
            ),
            ModelResponse(final_text="Need to read the file before replacing it."),
        ]
    )
    approvals = StaticApprovalClient(ApprovalDecision.APPROVE)
    loop = AgentLoop(
        task="Replace source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "read-required" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=approvals,
    )

    result = loop.run()

    assert result.status == "completed"
    observation = next(
        item for item in loop.observations if item.tool_call_id == "unsafe_overwrite"
    )
    assert observation.metadata["status"] == "stale_write"
    assert observation.metadata["reason"] == "read_required_before_overwrite"
    assert approvals.requests == []
    assert "line 1 needle" in workspace.joinpath("source.py").read_text(encoding="utf-8")


def test_existing_whole_file_overwrite_rejects_external_change_after_read(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)

    class MutatingModelClient(ScriptedModelClient):
        def call_request(self, request):
            if len(self.calls) == 1:
                workspace.joinpath("source.py").write_text("external change\n", encoding="utf-8")
            return super().call_request(request)

    client = MutatingModelClient(
        [
            _read_call("fresh_read"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="stale_overwrite",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "agent replacement\n",
                            "overwrite": True,
                        },
                    )
                ]
            ),
            ModelResponse(final_text="The file changed, so I did not overwrite it."),
        ]
    )
    loop = AgentLoop(
        task="Replace source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "external-change" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    )

    result = loop.run()

    assert result.status == "completed"
    observation = next(
        item for item in loop.observations if item.tool_call_id == "stale_overwrite"
    )
    assert observation.metadata["status"] == "stale_write"
    assert observation.metadata["reason"] == "file_changed_since_read"
    assert workspace.joinpath("source.py").read_text(encoding="utf-8") == "external change\n"


def test_identical_search_is_reused_but_limit_or_depth_changes_are_not(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.search_text
    calls = 0

    def counting_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "search_text", counting_search)
    client = ScriptedModelClient(
        [
            _search_call("search_first"),
            _search_call("search_duplicate"),
            _search_call("search_different", limit=10),
            _search_call("search_depth", max_depth=2),
            _search_call("search_case", case_sensitive=False),
            _search_call("search_regex", use_regex=True),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Find needle",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    )

    loop.run()

    assert calls == 5
    duplicate = next(item for item in loop.observations if item.tool_call_id == "search_duplicate")
    assert duplicate.metadata["status"] == "duplicate_reused"


def test_incomplete_search_result_is_not_reused(tmp_path, monkeypatch) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.search_text
    calls = 0

    def first_timeout_then_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SearchResult(
                query="needle",
                truncated=True,
                truncation_reason="timeout",
                scanned_entries=1,
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "search_text", first_timeout_then_search)
    client = ScriptedModelClient(
        [
            _search_call("search_timeout"),
            _search_call("search_retry"),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Find needle",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
    )

    loop.run()

    assert calls == 2
    timed_out = next(item for item in loop.observations if item.tool_call_id == "search_timeout")
    retried = next(item for item in loop.observations if item.tool_call_id == "search_retry")
    assert timed_out.metadata["truncation_reason"] == "timeout"
    assert timed_out.metadata["scanned_entries"] == 1
    assert retried.metadata.get("status") != "duplicate_reused"


def test_workspace_write_invalidates_exact_search_reuse(tmp_path, monkeypatch) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.search_text
    calls = 0

    def counting_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "search_text", counting_search)
    client = ScriptedModelClient(
        [
            _search_call("search_before"),
            _read_call("read_for_write"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write_for_search",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "changed\n",
                            "overwrite": True,
                        },
                    )
                ]
            ),
            _search_call("search_after"),
            ModelResponse(final_text="done"),
        ]
    )
    loop = AgentLoop(
        task="Modify and search source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    )

    loop.run()

    assert calls == 2
    after = next(item for item in loop.observations if item.tool_call_id == "search_after")
    assert after.metadata.get("status") != "duplicate_reused"


def test_checkpoint_run_state_rebuilds_read_coverage(tmp_path, monkeypatch) -> None:
    workspace = _make_workspace(tmp_path)
    original = registry_module.read_file
    calls = 0

    def counting_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "read_file", counting_read)
    state = RunState(
        inspected_files=[
            InspectedFile(
                path="source.py",
                summary="Read source.py lines 1-200 of 300.",
                last_tool_call_id="prior_read",
                last_step=1,
                line_start=1,
                line_end=200,
                total_lines=300,
                content_status="partial_content_available_in_context",
                workspace_generation=0,
            )
        ]
    )
    history = [
        {"role": "user", "content": "Inspect source.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "prior_read",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"source":"workspace","target":"source.py","start_line":1,"end_line":200}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "prior_read",
            "content": "tool: read\nstatus: ok\nresource: source.py\nrange: 1-200 of 300",
        },
    ]
    client = ScriptedModelClient([_read_call("resumed_read", 50, 80), ModelResponse(final_text="done")])
    loop = AgentLoop(
        task="Inspect source.py",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        config=AgentLoopConfig(start_step=1),
        initial_run_state=state,
        initial_message_history=history,
    )

    loop.run()

    assert calls == 0
    resumed = next(item for item in loop.observations if item.tool_call_id == "resumed_read")
    assert resumed.metadata["status"] == "duplicate_reused"


def test_resume_uses_latest_workspace_generation_after_write_without_reread(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = _make_workspace(tmp_path)
    first_client = ScriptedModelClient(
        [
            _read_call("read_zero"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write_one",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "first\n" * 30,
                            "overwrite": True,
                        },
                    )
                ]
            ),
            _read_call("read_one"),
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="write_two",
                        name="write",
                        arguments={
                            "path": "source.py",
                            "content": "second\n" * 30,
                            "overwrite": True,
                        },
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]
    )
    trace_path = tmp_path / "first_run" / "trace.jsonl"
    AgentLoop(
        task="Modify source.py twice",
        workspace=workspace,
        model_client=first_client,
        trace_writer=TraceWriter(trace_path),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        enable_write=True,
        approval_client=StaticApprovalClient(ApprovalDecision.APPROVE),
    ).run()
    checkpoint_store = CheckpointStore(trace_path.parent / "checkpoints")
    checkpoint = checkpoint_store.load_latest()
    assert checkpoint is not None
    assert max(
        int(item.metadata.get("workspace_generation") or 0)
        for item in checkpoint.recent_observations
    ) == 2

    original = registry_module.read_file
    calls = 0

    def counting_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_module, "read_file", counting_read)
    resumed_client = ScriptedModelClient(
        [_read_call("read_after_resume", 1, 20), ModelResponse(final_text="done")]
    )
    resumed_loop = AgentLoop(
        task=checkpoint.task,
        workspace=workspace,
        model_client=resumed_client,
        trace_writer=TraceWriter(tmp_path / "resumed_run" / "trace.jsonl"),
        memory_store=ProjectMemoryStore(tmp_path / "memory"),
        no_skills=True,
        config=AgentLoopConfig(start_step=checkpoint.step),
        initial_observations=list(checkpoint.recent_observations),
        initial_modified_files=list(checkpoint.modified_files),
        initial_run_state=checkpoint.run_state,
        initial_message_history=checkpoint_store.load_history(checkpoint),
    )

    resumed_loop.run()

    assert resumed_loop.workspace_generation == 2
    assert calls == 1
    observation = next(
        item for item in resumed_loop.observations if item.tool_call_id == "read_after_resume"
    )
    assert observation.metadata.get("status") != "duplicate_reused"
