from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import minicode_harness.cli as cli_module
from minicode_harness.memory import (
    MemoryResolvedFailure,
    MemoryVerificationEvidence,
    RepositoryMemoryPublisher,
    RepositoryMemoryStore,
)
from minicode_harness.models import ModelResponse


runner = CliRunner()


def test_memory_v2_remember_publishes_topic_without_model_call(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "minicode-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))

    first = runner.invoke(
        cli_module.app,
        [
            "memory",
            "remember",
            "Do not run the full test suite unless explicitly requested.",
            "--workspace",
            str(workspace),
        ],
    )
    duplicate = runner.invoke(
        cli_module.app,
        [
            "memory",
            "remember",
            "Do not run the full test suite unless explicitly requested.",
            "--workspace",
            str(workspace),
        ],
    )

    assert first.exit_code == 0, first.output
    assert "Memory saved" in first.output
    assert "topic=instructions" in first.output
    assert duplicate.exit_code == 0
    assert "already_exists" in duplicate.output
    store = RepositoryMemoryStore(workspace, data_dir=data_dir)
    assert [
        entry.summary for entry in store.topic_store.active_entries("instructions")
    ] == ["Do not run the full test suite unless explicitly requested."]

    listed = runner.invoke(
        cli_module.app,
        ["memory", "list", "--workspace", str(workspace)],
    )
    assert listed.exit_code == 0
    assert "Pending reviews: 0; pending candidates: 0" in listed.output
    assert "instructions" in listed.output


def test_memory_v2_review_candidates_and_approve(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "minicode-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))
    store = RepositoryMemoryStore(workspace, data_dir=data_dir)
    store.workflow_store.append_review_record(
        source_run_id="run_1",
        user_text="以后只运行聚焦测试。",
        assistant_text="已完成。",
        verification=MemoryVerificationEvidence(
            status="passed",
            command="python -m pytest tests/test_memory_v2_cli.py -q",
            returncode=0,
        ),
        resolved_failure=MemoryResolvedFailure(
            failed_command="python -m pytest tests/test_memory_v2_cli.py -q",
            failure_summary="candidate output assertion failed",
            modified_files=["tests/test_memory_v2_cli.py"],
            passed_command="python -m pytest tests/test_memory_v2_cli.py -q",
        ),
    )

    class FakeClient:
        def __init__(self) -> None:
            self.requests = []

        def call_request(self, request):
            self.requests.append(request)
            return ModelResponse(
                final_text=json.dumps(
                    {
                        "candidates": [
                            {
                                "topic": "build-and-test",
                                "text": "只运行聚焦测试。",
                                "reason": "用户明确给出长期测试偏好。",
                                "source_review_seqs": [1],
                            }
                        ]
                    }
                )
            )

    client = FakeClient()
    monkeypatch.setattr(cli_module, "create_model_client", lambda **kwargs: client)

    reviewed = runner.invoke(
        cli_module.app,
        [
            "memory",
            "review",
            "--workspace",
            str(workspace),
            "--provider",
            "deepseek",
            "--model",
            "deepseek-chat",
        ],
    )
    candidates = runner.invoke(
        cli_module.app,
        ["memory", "candidates", "--workspace", str(workspace)],
    )
    candidate_id = store.workflow_store.list_candidates(status="pending")[0].candidate_id
    approved = runner.invoke(
        cli_module.app,
        ["memory", "approve", candidate_id, "--workspace", str(workspace)],
    )

    assert reviewed.exit_code == 0, reviewed.output
    assert "Memory review completed" in reviewed.output
    assert len(client.requests) == 1
    assert candidates.exit_code == 0
    assert candidate_id in candidates.output
    assert "[pending]" in candidates.output
    assert "evidence: review=1; run=run_1; verification=passed" in candidates.output
    assert "verification command: python -m pytest tests/test_memory_v2_cli.py -q" in candidates.output
    assert "failure: python -m pytest tests/test_memory_v2_cli.py -q ->" in candidates.output
    assert "modified files: tests/test_memory_v2_cli.py" in candidates.output
    assert approved.exit_code == 0, approved.output
    assert "status=approved" in approved.output
    assert [
        entry.summary for entry in store.topic_store.active_entries("build-and-test")
    ] == ["只运行聚焦测试。"]


def test_memory_v2_review_auto_publishes_safe_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "minicode-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))
    store = RepositoryMemoryStore(workspace, data_dir=data_dir)
    source_quote = "团队长期约定缓存身份由 tenant_id 与 ticket_id 共同构成。"
    store.workflow_store.append_review_record(
        source_run_id="run_auto",
        user_text=source_quote,
        assistant_text="已确认。",
    )

    class FakeClient:
        def call_request(self, request):
            return ModelResponse(
                final_text=json.dumps(
                    {
                        "candidates": [
                            {
                                "topic": "decisions",
                                "text": source_quote,
                                "reason": "明确的长期仓库约定。",
                                "source_review_seqs": [1],
                                "auto_publish": True,
                                "source_quotes": [source_quote],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr(
        cli_module,
        "create_model_client",
        lambda **kwargs: FakeClient(),
    )

    reviewed = runner.invoke(
        cli_module.app,
        ["memory", "review", "--workspace", str(workspace)],
    )

    assert reviewed.exit_code == 0, reviewed.output
    assert "auto_published=1" in reviewed.output
    assert "pending_candidates=0" in reviewed.output
    assert [
        entry.summary for entry in store.topic_store.active_entries("decisions")
    ] == [source_quote]


def test_memory_v2_list_warns_when_topic_is_near_capacity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "minicode-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))
    store = RepositoryMemoryStore(workspace, data_dir=data_dir)
    publisher = RepositoryMemoryPublisher(store)

    warning = None
    for index in range(20):
        result = publisher.publish(
            topic="decisions",
            text=f"decision-{index}: " + ("x" * 520),
            evidence_ids=[f"test:{index}"],
        )
        warning = result.capacity_warning
        if warning:
            break

    assert warning is not None
    listed = runner.invoke(
        cli_module.app,
        ["memory", "list", "--workspace", str(workspace)],
    )

    assert listed.exit_code == 0, listed.output
    assert "Warning: Memory Topic decisions uses" in listed.output
    assert "review and forget stale entries" in listed.output


def test_memory_v2_remember_then_show_and_forget(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "minicode-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))

    remembered = runner.invoke(
        cli_module.app,
        [
            "memory",
            "remember",
            "Prefer focused tests.",
            "--topic",
            "instructions",
            "--workspace",
            str(workspace),
        ],
    )
    store = RepositoryMemoryStore(workspace, data_dir=data_dir)
    entries = store.topic_store.active_entries("instructions")
    entry_id = entries[0].entry_id

    shown = runner.invoke(
        cli_module.app,
        ["memory", "show", entry_id, "--workspace", str(workspace)],
    )
    forgotten = runner.invoke(
        cli_module.app,
        ["memory", "forget", entry_id, "--workspace", str(workspace)],
    )
    legacy = runner.invoke(
        cli_module.app,
        ["memory", "consolidate", "--workspace", str(workspace)],
    )

    assert remembered.exit_code == 0, remembered.output
    assert shown.exit_code == 0
    assert "Prefer focused tests" in shown.output
    assert forgotten.exit_code == 0
    assert store.topic_store.get_entry("instructions", entry_id).status == "inactive"
    assert "instructions" not in store.index_store.ensure().content
    assert legacy.exit_code == 1
    assert "must be list, remember, review" in legacy.output
