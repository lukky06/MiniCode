from minicode_harness.output import NullOutputSink
from minicode_harness.runtime.run_executor import RunExecutionRequest, RunExecutor
from minicode_harness.state import RunStore, StaticApprovalClient


class LifecycleSink(NullOutputSink):
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def run_started(self, run_id: str) -> None:
        self.events.append(("run_started", run_id))

    def run_finished(
        self,
        *,
        status: str,
        run_id: str | None = None,
        stop_reason: str | None = None,
    ) -> None:
        self.events.append(("run_finished", status, run_id, stop_reason))


def test_run_executor_exposes_optional_lifecycle_without_model_call(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sink = LifecycleSink()

    result = RunExecutor(run_store=RunStore(tmp_path / "runs")).execute(
        RunExecutionRequest(task="noop", workspace=workspace, dry_run=True),
        output_sink=sink,
        approval_client=StaticApprovalClient(),
    )

    assert sink.events == [
        ("run_started", result.run_id),
        ("run_finished", "created", result.run_id, "dry_run"),
    ]
