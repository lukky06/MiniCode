import json
from types import SimpleNamespace

from minicode_harness.context import SEMANTIC_HISTORY_HEADING
from minicode_harness.context.session_projection import project_canonical_messages
from minicode_harness.loop import AgentRunResult
from minicode_harness.state import ReplSessionMemory
from minicode_harness.memory.repository_id import RepositoryIdentityUnavailable
from minicode_harness.models import ModelCapabilities, ModelResponse
from minicode_harness.output import NullOutputSink
import minicode_harness.runtime.run_executor as run_executor_module
from minicode_harness.runtime.run_executor import (
    RunExecutionRequest,
    RunExecutor,
)
from minicode_harness.state import RunStore, StaticApprovalClient
from minicode_harness.subagent import SubagentResult


def test_run_executor_compacts_active_session_and_preserves_latest_turn(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_memory = ReplSessionMemory(workspace=str(workspace.resolve()))
    original = [
        {"role": "user", "content": "Keep architecture simple." + ("a" * 2_000)},
        {"role": "assistant", "content": "Agreed." + ("b" * 2_000)},
        {"role": "user", "content": "Do not run the full benchmark." + ("c" * 2_000)},
        {"role": "assistant", "content": "Acknowledged." + ("d" * 2_000)},
        {"role": "user", "content": "Latest request stays exact."},
        {"role": "assistant", "content": "Latest answer stays exact."},
    ]
    session_memory.replace_message_history(original)
    requests = []

    class FakeModelClient:
        capabilities = ModelCapabilities(
            context_window=16_000,
            max_output_tokens=2_000,
        )

        def call_request(self, request):
            requests.append(request)
            return ModelResponse(
                final_text=json.dumps(
                    {
                        "items": [
                            {
                                "kind": "constraint",
                                "text": "Keep architecture simple.",
                                "source_turn_ids": ["t0001"],
                            }
                        ]
                    }
                )
            )

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.create_model_client",
        lambda **kwargs: FakeModelClient(),
    )

    result = RunExecutor(session_memory=session_memory).compact_session(
        provider="deepseek",
        model="deepseek-chat",
        focus="保留架构约束和验证边界",
    )

    canonical = session_memory.load_message_history()
    state = session_memory.load_compaction_state()
    projected = project_canonical_messages(canonical, state)
    assert canonical == original
    assert result.changed is True
    assert result.after_tokens < result.before_tokens
    assert result.removed_groups > 0
    assert projected[-2:] == original[-2:]
    assert state.semantic is not None
    assert state.semantic.summary.startswith(SEMANTIC_HISTORY_HEADING)
    payload = json.loads(str(requests[0].messages[0]["content"]))
    assert payload["manual_focus"] == "保留架构约束和验证边界"


def test_run_executor_review_uses_review_skill_and_readonly_subagent(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace-review"
    workspace.mkdir()
    (workspace / "example.py").write_text("VALUE = 2\n", encoding="utf-8")
    run_store = RunStore(tmp_path / "runs-review")
    captured = {}

    monkeypatch.setattr(
        run_executor_module,
        "inspect_git_diff",
        lambda workspace: SimpleNamespace(returncode=0, diff="@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2", stderr=""),
    )
    monkeypatch.setattr(
        run_executor_module,
        "create_model_client",
        lambda **kwargs: SimpleNamespace(
            model="fake-review-model",
            capabilities=SimpleNamespace(
                context_window=32000,
                reserved_output_tokens=4000,
                max_output_tokens=4000,
            ),
        ),
    )
    monkeypatch.setattr(
        run_executor_module.SkillLoader,
        "load",
        lambda self, name: SimpleNamespace(
            content="# Review Skill\nInspect correctness and report P0/P1/P2.",
        ),
    )

    class FakeRunner:
        def __init__(self, **kwargs):
            captured["runner_kwargs"] = kwargs

        def run(self, task):
            captured["task"] = task
            return SubagentResult(
                status="completed",
                summary="P1 example.py: regression\nGate recommendation: FAIL",
            )

    monkeypatch.setattr(run_executor_module, "ReadonlySubagentRunner", FakeRunner)

    result = RunExecutor(run_store=run_store).review_current_diff(
        workspace=workspace,
        provider="deepseek",
        model="deepseek-reasoner",
        focus="recovery",
    )

    assert result.status == "completed"
    assert "P1 example.py" in result.summary
    assert "# Review Skill" in captured["task"]
    assert "recovery" in captured["task"]
    assert captured["runner_kwargs"]["workspace"] == workspace.resolve()
    session = run_store.load_session(result.run_id)
    assert session.no_write is True
    assert session.subagents_enabled is False
    assert session.collaboration_mode == "plan"


def test_run_executor_creates_dry_run_without_model_call(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")

    def fail_model_creation(*args, **kwargs):
        raise AssertionError("dry run must not create a model client")

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.create_model_client",
        fail_model_creation,
    )

    result = RunExecutor(run_store=run_store).execute(
        RunExecutionRequest(task="noop", workspace=workspace, dry_run=True),
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
    )

    assert result.status == "created"
    assert result.stop_reason == "dry_run"
    assert result.dry_run is True
    assert result.run_path == run_store.path_for(result.run_id)
    assert (result.run_path / "run.json").is_file()
    assert (result.run_path / "trace.jsonl").read_text(encoding="utf-8").count("run_started") == 1


def test_run_executor_continues_when_repository_identity_is_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.create_model_client",
        lambda **kwargs: SimpleNamespace(model="fake-model"),
    )

    def fail_repository_memory(*args, **kwargs):
        raise RepositoryIdentityUnavailable("Git repository identity probe timed out.")

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.RepositoryMemoryStore",
        fail_repository_memory,
    )

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.modified_files = []
            self.run_state = SimpleNamespace(
                inspected_files=[],
                verification=SimpleNamespace(status="not_run"),
            )

        def run(self) -> AgentRunResult:
            return AgentRunResult(
                status="completed",
                final_text="still answered",
                steps=1,
                tool_calls=0,
                stop_reason="final_text",
            )

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.AgentLoop",
        FakeLoop,
    )

    result = RunExecutor(run_store=run_store).execute(
        RunExecutionRequest(task="hello", workspace=workspace),
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
    )

    assert result.status == "completed"
    assert result.final_text == "still answered"
    assert captured["repository_memory"] is None
    assert captured["long_term_context"] == ""
    trace = (result.run_path / "trace.jsonl").read_text(encoding="utf-8")
    assert "memory_initialization_skipped" in trace
    assert "RepositoryIdentityUnavailable" in trace


def test_run_executor_owns_recall_loop_persistence_and_finalization(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session_memory = ReplSessionMemory(workspace=str(workspace.resolve()))
    calls: dict[str, object] = {}
    model_client = SimpleNamespace(model="fake-model")

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.create_model_client",
        lambda **kwargs: model_client,
    )

    class FakeOrchestrator:
        def __init__(
            self,
            *,
            trace_writer,
            repository_memory,
            review_model_client,
        ) -> None:
            calls["repository_memory"] = repository_memory
            calls["review_model_client"] = review_model_client

        def recall_snapshot(self):
            calls["recall"] = True
            return SimpleNamespace(
                rendered_index="Repository Memory Index:\n- build-and-test",
                topic_payloads={},
            )

        def finalize_completed_run(self, **kwargs):
            calls["finalize"] = kwargs
            return SimpleNamespace(
                review_status="completed",
                reviewed_turns=8,
                candidate_count=2,
                auto_published_count=1,
                pending_candidates=1,
            )

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.RequestOrchestrator",
        FakeOrchestrator,
    )

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            calls["loop_kwargs"] = kwargs
            self.modified_files = ["src/example.py"]
            self.run_state = SimpleNamespace(
                inspected_files=[],
                verification=SimpleNamespace(status="passed"),
            )

        def run(self) -> AgentRunResult:
            return AgentRunResult(
                status="completed",
                final_text="done",
                steps=3,
                tool_calls=2,
                stop_reason="completed",
            )

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.AgentLoop",
        FakeLoop,
    )

    sink = NullOutputSink()
    approval = StaticApprovalClient()
    result = RunExecutor(
        run_store=run_store,
        session_memory=session_memory,
    ).execute(
        RunExecutionRequest(
            task="change the code",
            workspace=workspace,
            provider="qwen",
            model="fake-model",
            skills=["code-debug"],
        ),
        output_sink=sink,
        approval_client=approval,
    )

    loop_kwargs = calls["loop_kwargs"]
    assert loop_kwargs["output_sink"] is sink
    assert loop_kwargs["approval_client"] is approval
    assert loop_kwargs["session_memory"] is session_memory
    assert loop_kwargs["long_term_context"].startswith("Repository Memory Index")
    assert loop_kwargs["repository_memory"] is calls["repository_memory"]
    assert loop_kwargs["data_dir"] == calls["repository_memory"].data_dir
    assert "memory_store" not in loop_kwargs
    assert result.status == "completed"
    assert result.conversation_session_id == session_memory.session_id
    assert result.final_text == "done"
    assert result.modified_files == ["src/example.py"]
    assert result.verification_status == "passed"
    assert result.memory_review_status == "completed"
    assert result.memory_reviewed_turns == 8
    assert result.memory_candidate_count == 2
    assert result.memory_auto_published_count == 1
    assert result.memory_pending_candidates == 1
    assert [turn.role for turn in session_memory.dialogue] == ["user", "assistant"]
    assert calls["finalize"]["run_id"] == result.run_id
    run_session = json.loads(
        (result.run_path / "run.json").read_text(encoding="utf-8")
    )
    assert run_session["conversation_session_id"] == session_memory.session_id
    assert run_session["status"] == "completed"
    assert run_session["current_step"] == 3
    trace = (result.run_path / "trace.jsonl").read_text(encoding="utf-8")
    assert "run_started" in trace
    assert session_memory.session_id in trace
    assert "run_finished" in trace


def test_run_executor_exposes_stop_summary_without_persisting_it_as_model_text(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_store = RunStore(tmp_path / "runs")
    session_memory = ReplSessionMemory(workspace=str(workspace.resolve()))

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.create_model_client",
        lambda **kwargs: SimpleNamespace(model="fake-model"),
    )

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            self.modified_files = ["src/example.py"]
            self.run_state = SimpleNamespace(
                inspected_files=[],
                verification=SimpleNamespace(status="not_run"),
            )

        def run(self) -> AgentRunResult:
            return AgentRunResult(
                status="stopped",
                final_text=None,
                steps=20,
                tool_calls=23,
                stop_reason="max_steps",
                stop_summary=(
                    "Run stopped because the model-call budget was exhausted.\n"
                    "Changes retained: src/example.py."
                ),
            )

    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.AgentLoop",
        FakeLoop,
    )
    monkeypatch.setattr(
        "minicode_harness.runtime.run_executor.RepositoryMemoryStore",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("memory must stay disabled")
        ),
    )

    result = RunExecutor(
        run_store=run_store,
        session_memory=session_memory,
    ).execute(
        RunExecutionRequest(
            task="fix inventory",
            workspace=workspace,
            repository_memory_enabled=False,
        ),
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
    )

    assert result.status == "stopped"
    assert result.final_text == ""
    assert "model-call budget was exhausted" in result.stop_summary
    assistant_turn = session_memory.dialogue[-1]
    assert assistant_turn.role == "assistant"
    assert result.stop_summary not in assistant_turn.content
    assert assistant_turn.content == "Run finished with status stopped: max_steps"
