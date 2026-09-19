from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import minicode_harness.cli as cli_module
from minicode_harness.memory.store import RepositoryMemoryStore
from minicode_harness.models import ModelResponse


runner = CliRunner()


class FakeClient:
    def __init__(self) -> None:
        self.requests = []

    def call_request(self, request):
        self.requests.append(request)
        return ModelResponse(
            final_text=json.dumps(
                {
                    "memory_md": "# Memory\n\nUse focused tests.\n",
                    "memory_summary_md": "v1\n- Focused-test preference.\n",
                }
            )
        )


def _workspace(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))
    return workspace, RepositoryMemoryStore(workspace, data_dir=data_dir)


def test_memory_status_reports_minimal_v3_state(tmp_path: Path, monkeypatch) -> None:
    workspace, _ = _workspace(tmp_path, monkeypatch)

    result = runner.invoke(
        cli_module.app,
        ["memory", "status", "--workspace", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert "schema_version=3" in result.output
    assert "latest_stage1_seq=0" in result.output
    assert "last_phase2_input_seq=0" in result.output
    assert "dirty=false" in result.output


def test_memory_show_displays_summary_and_handbook(tmp_path: Path, monkeypatch) -> None:
    workspace, store = _workspace(tmp_path, monkeypatch)
    store.write_durable_memory(
        "# Memory\n\nDetailed handbook.\n",
        "v1\n- Compact summary.\n",
    )

    result = runner.invoke(
        cli_module.app,
        ["memory", "show", "--workspace", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert "memory_summary.md" in result.output
    assert "Compact summary" in result.output
    assert "MEMORY.md" in result.output
    assert "Detailed handbook" in result.output


def test_memory_consolidate_clean_is_noop_without_model(tmp_path: Path, monkeypatch) -> None:
    workspace, _ = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "create_model_client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should not be created")),
    )

    result = runner.invoke(
        cli_module.app,
        ["memory", "consolidate", "--workspace", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert "NOOP" in result.output


def test_memory_consolidate_dirty_runs_phase2_explicitly(tmp_path: Path, monkeypatch) -> None:
    workspace, store = _workspace(tmp_path, monkeypatch)
    store.write_stage1_memory(
        run_id="run_20260918_001",
        raw_memory="Prefer focused tests.",
        rollout_summary="Focused tests were confirmed.",
        rollout_slug="focused-tests",
    )
    client = FakeClient()
    monkeypatch.setattr(cli_module, "create_model_client", lambda **kwargs: client)

    result = runner.invoke(
        cli_module.app,
        ["memory", "consolidate", "--workspace", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert "completed" in result.output
    assert len(client.requests) == 1
    state = store.load_state()
    assert state.last_phase2_input_seq == state.latest_stage1_seq == 1


def test_memory_consolidate_returns_busy_without_waiting(tmp_path: Path, monkeypatch) -> None:
    workspace, store = _workspace(tmp_path, monkeypatch)
    held = store.try_pipeline_lock()
    assert held is not None
    monkeypatch.setattr(
        cli_module,
        "create_model_client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should not be created")),
    )
    try:
        result = runner.invoke(
            cli_module.app,
            ["memory", "consolidate", "--workspace", str(workspace)],
        )
    finally:
        held.release()

    assert result.exit_code == 0, result.output
    assert "busy" in result.output.lower()
