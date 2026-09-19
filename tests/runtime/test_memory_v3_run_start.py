from __future__ import annotations

from types import SimpleNamespace

import minicode_harness.runtime.run_executor as run_executor_module
from minicode_harness.loop import AgentRunResult
from minicode_harness.memory.snapshot import MemorySnapshotStore
from minicode_harness.memory.store import RepositoryMemoryStore
from minicode_harness.models import ModelCapabilities
from minicode_harness.output import NullOutputSink
from minicode_harness.runtime.run_executor import RunExecutionRequest, RunExecutor
from minicode_harness.state import RunStore, StaticApprovalClient


def test_new_run_freezes_v3_memory_before_background_pipeline(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))
    repository = RepositoryMemoryStore(workspace, data_dir=data_dir)
    repository.write_durable_memory(
        "# Memory\n\nSECRET HANDBOOK DETAIL\n",
        "v1\n- Summary only.\n",
    )
    run_store = RunStore(tmp_path / "runs")
    calls: dict[str, object] = {}

    model_client = SimpleNamespace(
        model="fake-model",
        capabilities=ModelCapabilities(
            context_window=32000,
            reserved_output_tokens=4000,
            max_output_tokens=4096,
        ),
    )
    monkeypatch.setattr(
        run_executor_module,
        "create_model_client",
        lambda **kwargs: model_client,
    )

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            calls["loop_kwargs"] = kwargs
            self.observations = []
            self.modified_files = []
            self.run_state = SimpleNamespace(
                inspected_files=[],
                verification=SimpleNamespace(status="not_run"),
            )

        def run(self) -> AgentRunResult:
            return AgentRunResult(
                status="stopped",
                final_text=None,
                steps=1,
                tool_calls=0,
                stop_reason="max_steps",
            )

    monkeypatch.setattr(run_executor_module, "AgentLoop", FakeLoop)

    def fake_start_memory_pipeline(**kwargs) -> None:
        snapshot_store = MemorySnapshotStore(
            kwargs["run_store"].path_for(kwargs["current_run_id"])
        )
        snapshot = snapshot_store.load()
        assert snapshot is not None
        assert snapshot_store.read_summary() == "v1\n- Summary only.\n"
        assert "SECRET HANDBOOK DETAIL" in snapshot_store.read_memory()
        calls["background"] = kwargs

    monkeypatch.setattr(
        run_executor_module,
        "_start_memory_pipeline",
        fake_start_memory_pipeline,
    )

    result = RunExecutor(run_store=run_store).execute(
        RunExecutionRequest(task="inspect", workspace=workspace),
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
    )

    loop_kwargs = calls["loop_kwargs"]
    assert loop_kwargs["long_term_context"] == "v1\n- Summary only.\n"
    assert "SECRET HANDBOOK DETAIL" not in loop_kwargs["long_term_context"]
    assert loop_kwargs["memory_snapshot_hash"]
    background = calls["background"]
    assert background["current_run_id"] == result.run_id
    assert background["workspace"] == workspace.resolve()
    assert background["provider"] == "qwen"
