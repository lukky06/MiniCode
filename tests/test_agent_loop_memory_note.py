from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minicode_harness.loop import AgentLoop, AgentLoopConfig
from minicode_harness.memory import MemorySnapshotStore, RepositoryMemoryStore
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.trace import TraceWriter


def test_agent_loop_limits_successful_memory_topic_reads_per_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    repository = RepositoryMemoryStore(workspace, data_dir=data_dir)
    topic_types = {
        "decisions": "decision",
        "build-and-test": "procedure",
        "instructions": "user_instruction",
    }
    for topic, entry_type in topic_types.items():
        repository.topic_store.add_entry(
            topic=topic,
            entry_type=entry_type,
            summary=f"requirement for {topic}",
            evidence_ids=[f"topic:{topic}"],
        )
    repository.refresh_index()
    run_path = tmp_path / "run_topic_limit"
    run_path.mkdir()
    source = repository.capture_snapshot_source()
    snapshot_store = MemorySnapshotStore(run_path)
    snapshot = snapshot_store.save(
        repository_id=repository.repository_id,
        rendered_index=source.rendered_index,
        topic_payloads=source.topic_payloads,
    )

    class TopicClient(ModelClient):
        def __init__(self) -> None:
            self.responses = [
                ModelResponse(
                    tool_calls=[
                        NormalizedToolCall(
                            id="read_build",
                            name="read",
                            arguments={"source": "memory", "target": "build-and-test"},
                        ),
                        NormalizedToolCall(
                            id="read_decisions",
                            name="read",
                            arguments={"source": "memory", "target": "decisions"},
                        ),
                        NormalizedToolCall(
                            id="read_instructions",
                            name="read",
                            arguments={"source": "memory", "target": "instructions"},
                        ),
                    ]
                ),
                ModelResponse(final_text="done"),
            ]

        def call_request(self, request) -> ModelResponse:
            return self.responses.pop(0)

    result = AgentLoop(
        task="修复缓存并运行聚焦测试。",
        workspace=workspace,
        model_client=TopicClient(),
        trace_writer=TraceWriter(run_path / "trace.jsonl"),
        config=AgentLoopConfig(max_memory_topic_reads=2),
        data_dir=data_dir,
        repository_memory=repository,
        memory_snapshot_hash=snapshot.index_hash,
        memory_snapshot_path=snapshot_store.checkpoint_path,
        no_skills=True,
        run_id="run_topic_limit",
    ).run()

    assert result.status == "completed"
    events = [
        json.loads(line)
        for line in (run_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    statuses = [
        event["status"]
        for event in events
        if event["type"] == "tool_result" and event.get("tool") == "read"
    ]
    assert statuses == ["ok", "ok", "memory_topic_read_limit"]
