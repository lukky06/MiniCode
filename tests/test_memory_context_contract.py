from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minicode_harness.context import ContextBuilder, ContextPreparer, TokenBudget
from minicode_harness.loop import AgentLoop
from minicode_harness.memory import MemoryApprovalCandidate, RepositoryMemoryStore
from minicode_harness.trace import TraceWriter


def _repository_memory(tmp_path: Path, workspace: Path) -> RepositoryMemoryStore:
    store = RepositoryMemoryStore(workspace, data_dir=tmp_path / "data")
    store.topic_store.add_entry(
        topic="decisions",
        entry_type="decision",
        summary="Keep canonical messages as the only conversation history.",
        evidence_ids=["candidate_decision"],
    )
    store.refresh_index()
    return store


def test_memory_index_enters_only_system_prompt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = _repository_memory(tmp_path, workspace)
    messages = [
        {"role": "user", "content": "inspect the project"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"source":"workspace","target":"a.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file content"},
    ]
    builder = ContextBuilder(budget=TokenBudget(context_budget=32000, reserved_output=4000))
    built = builder.build(
        available_skills=[],
        workspace=workspace,
        long_term_context=memory.render_index(),
    )
    prepared = ContextPreparer(builder.budget).prepare(
        system_messages=built.messages,
        messages=messages,
        tools=[],
    )

    assert 'read(source="memory", target="decisions")' in prepared.request.system
    assert prepared.request.messages == messages


def test_pending_candidate_never_enters_memory_index(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = _repository_memory(tmp_path, workspace)
    appended = memory.workflow_store.append_review_record(
        source_run_id="run_1",
        user_text="Remember the focused test workflow.",
        assistant_text="Acknowledged.",
    )
    assert appended.record is not None
    candidate = MemoryApprovalCandidate.create(
        topic="build-and-test",
        text="This candidate must stay outside the prompt.",
        reason="Pending review candidate.",
        source_review_seqs=[appended.record.seq],
    )
    memory.workflow_store.save_review_result(
        reviewed_seqs=[appended.record.seq],
        candidates=[candidate],
    )

    assert "This candidate must stay outside the prompt" not in memory.render_index()


def test_agent_loop_holds_one_memory_snapshot_for_the_run(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = _repository_memory(tmp_path, workspace)
    initial = memory.render_index()
    model = SimpleNamespace(
        model="fake-model",
        capabilities=SimpleNamespace(
            context_window=32000,
            reserved_output_tokens=4000,
            max_output_tokens=4096,
        ),
    )
    loop = AgentLoop(
        task="inspect memory",
        workspace=workspace,
        model_client=model,
        trace_writer=TraceWriter(tmp_path / "run" / "trace.jsonl"),
        data_dir=tmp_path / "data",
        repository_memory=memory,
        long_term_context=initial,
        enable_write=False,
    )

    memory.topic_store.add_entry(
        topic="environment",
        entry_type="environment",
        summary="Python 3.12 is available.",
        evidence_ids=["candidate_environment"],
    )
    memory.refresh_index()

    assert loop.long_term_context == initial
    assert "Python 3.12" not in loop.long_term_context
