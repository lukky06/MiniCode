from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.state import CheckpointStore, RunCheckpoint
from minicode_harness.swebench import (
    RepositoryCache,
    SweBenchBudget,
    SweBenchEvaluator,
    SweBenchInstance,
    SweBenchRunner,
    SweBenchRunnerConfig,
    SweBenchWorkspaceManager,
)
from minicode_harness.swebench.prompt import build_task_prompt

from .helpers import init_git_repo


class ScriptedModelClient(ModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def call_request(self, request) -> ModelResponse:
        return self.responses.pop(0)


def test_runner_executes_agent_and_exports_prediction(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(
        tmp_path / "source",
        {"app.py": "value = 1\n"},
    )
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": base_commit,
                "problem_statement": "Change value to 2.",
                "patch": "must never reach the agent",
                "test_patch": "must never reach the agent",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def factory(instance):
        assert instance.model_dump() == {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": base_commit,
            "problem_statement": "Change value to 2.",
            "version": None,
        }
        return ScriptedModelClient(
            [
                ModelResponse(
                    tool_calls=[
                        NormalizedToolCall(
                            id="read_1",
                            name="read",
                            arguments={"source": "workspace", "target": "app.py"},
                        )
                    ]
                ),
                ModelResponse(
                    tool_calls=[
                        NormalizedToolCall(
                            id="call_1",
                            name="write",
                            arguments={
                                "path": "app.py",
                                "content": "value = 2\n",
                                "overwrite": True,
                            },
                        )
                    ]
                ),
                ModelResponse(final_text="Updated app.py."),
            ]
        )

    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider="fake",
            model="fake-model",
            budget=SweBenchBudget(max_steps=5, max_tool_calls=5),
            enable_skills=False,
            repository_source_urls={"owner/repo": str(source)},
        ),
        model_client_factory=factory,
    )
    output = tmp_path / "output"

    summary = runner.run_dataset(dataset, output)

    assert summary.total_instances == 1
    assert summary.patches_generated == 1
    result_path = output / "instances" / "owner__repo-1" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "agent_completed"
    assert result["modified_files"] == ["app.py"]
    assert result["patch_bytes"] > 0
    prediction = json.loads(
        (output / "instances" / "owner__repo-1" / "prediction.json").read_text(
            encoding="utf-8"
        )
    )
    assert prediction["instance_id"] == "owner__repo-1"
    assert "value = 2" in prediction["model_patch"]
    jsonl = (output / "predictions.jsonl").read_text(encoding="utf-8")
    assert json.loads(jsonl)["model_patch"] == prediction["model_patch"]
    assert (source / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_runner_hides_run_command_when_agent_verification_is_disabled(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(
        tmp_path / "source",
        {"app.py": "value = 1\n"},
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "base_commit": base_commit,
                    "problem_statement": "Inspect the repository.",
                }
            ]
        ),
        encoding="utf-8",
    )
    exposed_tools: set[str] = set()

    class CapturingClient(ModelClient):
        def call_request(self, request):
            exposed_tools.update(tool["function"]["name"] for tool in request.tools)
            return ModelResponse(final_text="No change required.")

    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider="fake",
            agent_verification_enabled=False,
            enable_skills=False,
            repository_source_urls={"owner/repo": str(source)},
        ),
        model_client_factory=lambda instance: CapturingClient(),
    )

    runner.run_dataset(dataset, tmp_path / "output")

    assert {"edit", "write", "apply_patch"}.issubset(exposed_tools)
    assert "run_command" not in exposed_tools
    assert {
        "delegate_task",
        "delegate_worktree",
        "runtime_task_status",
        "runtime_task_stop",
    }.isdisjoint(exposed_tools)


def test_runner_reserves_last_step_for_final_answer(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(
        tmp_path / "source",
        {"app.py": "value = 1\n"},
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "base_commit": base_commit,
                    "problem_statement": "Change value to 2.",
                }
            ]
        ),
        encoding="utf-8",
    )
    calls = 0

    class FinalStepClient(ModelClient):
        def call_request(self, request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    tool_calls=[
                        NormalizedToolCall(
                            id="edit_1",
                            name="edit",
                            arguments={
                                "path": "app.py",
                                "old_text": "value = 1",
                                "new_text": "value = 2",
                            },
                        )
                    ]
                )
            assert request.tools == []
            assert any(
                "工具阶段已经结束" in str(message.get("content") or "")
                for message in request.as_chat_messages()
            )
            return ModelResponse(final_text="Updated app.py.")

    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider="fake",
            model="fake-model",
            budget=SweBenchBudget(max_steps=2, max_tool_calls=5),
            agent_verification_enabled=False,
            enable_skills=False,
            repository_source_urls={"owner/repo": str(source)},
        ),
        model_client_factory=lambda instance: FinalStepClient(),
    )

    summary = runner.run_dataset(dataset, tmp_path / "output")

    assert summary.patches_generated == 1
    assert summary.results[0].status == "agent_completed"
    assert "value = 2" in Path(summary.results[0].patch_path).read_text(encoding="utf-8")


def test_runner_retains_patch_when_final_step_is_invalid(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(
        tmp_path / "source",
        {"app.py": "value = 1\n"},
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "base_commit": base_commit,
                    "problem_statement": "Change value to 2.",
                }
            ]
        ),
        encoding="utf-8",
    )

    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider="fake",
            model="fake-model",
            budget=SweBenchBudget(max_steps=2, max_tool_calls=5),
            agent_verification_enabled=False,
            enable_skills=False,
            repository_source_urls={"owner/repo": str(source)},
        ),
        model_client_factory=lambda instance: ScriptedModelClient(
            [
                ModelResponse(
                    tool_calls=[
                        NormalizedToolCall(
                            id="edit_1",
                            name="edit",
                            arguments={
                                "path": "app.py",
                                "old_text": "value = 1",
                                "new_text": "value = 2",
                            },
                        )
                    ]
                ),
                ModelResponse(),
            ]
        ),
    )

    summary = runner.run_dataset(dataset, tmp_path / "output")

    assert summary.patches_generated == 1
    assert summary.results[0].status == "step_budget_exhausted"
    assert summary.results[0].modified_files == ["app.py"]
    assert "value = 2" in Path(summary.results[0].patch_path).read_text(encoding="utf-8")


def test_runner_skips_persisted_completed_instance(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(tmp_path / "source")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "base_commit": base_commit,
                    "problem_statement": "No change required.",
                }
            ]
        ),
        encoding="utf-8",
    )
    calls = 0

    def factory(instance):
        nonlocal calls
        _ = instance
        calls += 1
        return ScriptedModelClient([ModelResponse(final_text="No change required.")])

    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider="fake",
            enable_skills=False,
            repository_source_urls={"owner/repo": str(source)},
        ),
        model_client_factory=factory,
    )
    output = tmp_path / "output"

    first = runner.run_dataset(dataset, output)
    second = runner.run_dataset(dataset, output)

    assert first.results[0].status == "no_patch"
    assert second.results[0].status == "no_patch"
    assert calls == 1


def test_runner_resumes_checkpoint_without_readding_user_task(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(tmp_path / "source")
    instance = SweBenchInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit=base_commit,
        problem_statement="Inspect the repository.",
    )
    output = tmp_path / "output"
    manager = SweBenchWorkspaceManager(
        output_root=output,
        repository_cache=RepositoryCache(output / "cache"),
    )
    manager.prepare(instance, source_url=str(source))
    instance_dir = manager.instance_dir(instance.instance_id)
    task = build_task_prompt(instance)
    CheckpointStore(instance_dir / "checkpoints").save(
        RunCheckpoint(
            run_id="swebench_resume",
            step=1,
            task=task,
            workspace=str(instance_dir / "workspace"),
            message_history=[{"role": "user", "content": task}],
            model_call_count=1,
            status="running",
            reason="interrupted",
        )
    )
    observed_messages: list[list[dict[str, Any]]] = []

    class ResumeClient(ScriptedModelClient):
        def call_request(self, request):
            observed_messages.append(request.as_chat_messages())
            return super().call_request(request)

    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider="fake",
            enable_skills=False,
            repository_source_urls={"owner/repo": str(source)},
        ),
        model_client_factory=lambda current: ResumeClient(
            [ModelResponse(final_text="Resumed.")]
        ),
    )

    summary = runner.run_instances(
        [instance],
        output,
        dataset_label="local",
    )

    assert summary.results[0].run_id == "swebench_resume"
    assert summary.results[0].status == "no_patch"
    assert [message for message in observed_messages[0] if message["role"] == "user"] == [
        {"role": "user", "content": task}
    ]


def test_multiple_attempts_are_evaluated_in_isolated_directories(tmp_path: Path) -> None:
    source, base_commit = init_git_repo(
        tmp_path / "source",
        {"app.py": "value = 0\n"},
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "base_commit": base_commit,
                    "problem_statement": "Change the value.",
                }
            ]
        ),
        encoding="utf-8",
    )
    model_attempt = 0

    def factory(instance):
        nonlocal model_attempt
        _ = instance
        model_attempt += 1
        return ScriptedModelClient(
            [
                ModelResponse(
                    tool_calls=[
                        NormalizedToolCall(
                            id=f"read_{model_attempt}",
                            name="read",
                            arguments={"source": "workspace", "target": "app.py"},
                        )
                    ]
                ),
                ModelResponse(
                    tool_calls=[
                        NormalizedToolCall(
                            id=f"call_{model_attempt}",
                            name="write",
                            arguments={
                                "path": "app.py",
                                "content": f"value = {model_attempt}\n",
                                "overwrite": True,
                            },
                        )
                    ]
                ),
                ModelResponse(final_text=f"Attempt {model_attempt}."),
            ]
        )

    def fake_evaluation(command, **kwargs):
        output = Path(kwargs["cwd"])
        run_id = command[command.index("--run_id") + 1]
        resolved = run_id.endswith("attempt_2")
        (output / "report.json").write_text(
            json.dumps({"owner__repo-1": {"resolved": resolved}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    output = tmp_path / "output"
    runner = SweBenchRunner(
        SweBenchRunnerConfig(
            provider="fake",
            model="fake-model",
            attempts=2,
            enable_skills=False,
            repository_source_urls={"owner/repo": str(source)},
        ),
        model_client_factory=factory,
    )
    evaluator = SweBenchEvaluator(
        evaluator_python="python",
        output_dir=output / "evaluation",
        process_runner=fake_evaluation,
    )

    summary = runner.run_dataset(
        dataset,
        output,
        evaluator=evaluator,
        evaluator_dataset_name="dataset",
        evaluation_run_id="run",
    )

    assert summary.total_instances == 1
    assert summary.total_attempts == 2
    assert summary.resolved == 1
    assert summary.unresolved == 1
    assert (output / "predictions.attempt_1.jsonl").is_file()
    assert (output / "predictions.attempt_2.jsonl").is_file()
    assert (output / "evaluation" / "attempt_1" / "evaluation.stdout.log").is_file()
    assert (output / "evaluation" / "attempt_2" / "evaluation.stdout.log").is_file()
    assert [result.status for result in summary.results] == ["unresolved", "resolved"]
