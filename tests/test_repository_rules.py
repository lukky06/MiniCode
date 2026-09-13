from __future__ import annotations

from pathlib import Path
from typing import Any

from minicode_harness.context import RepositoryRuleLoader
from minicode_harness.loop import AgentLoop
from minicode_harness.models import ModelClient, ModelResponse, NormalizedToolCall
from minicode_harness.trace import TraceWriter


def test_repository_rules_load_general_to_specific_for_touched_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "src" / "module" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('ok')\n", encoding="utf-8")

    user_rules = tmp_path / "user" / "AGENTS.md"
    user_rules.parent.mkdir()
    user_rules.write_text("user rule", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("root rule", encoding="utf-8")
    (workspace / "src" / "AGENTS.md").write_text("src rule", encoding="utf-8")
    (workspace / "src" / "module" / "AGENTS.md").write_text(
        "module rule",
        encoding="utf-8",
    )
    unrelated = workspace / "other"
    unrelated.mkdir()
    (unrelated / "AGENTS.md").write_text("unrelated rule", encoding="utf-8")

    snapshot = RepositoryRuleLoader(
        workspace,
        user_rules_path=user_rules,
    ).load(paths=["src/module/app.py"])

    assert snapshot.sources == (
        "user:~/.minicode/AGENTS.md",
        "repository:AGENTS.md; scope=.",
        "repository:src/AGENTS.md; scope=src",
        "repository:src/module/AGENTS.md; scope=src/module",
    )
    assert snapshot.content.index("user rule") < snapshot.content.index("root rule")
    assert snapshot.content.index("root rule") < snapshot.content.index("src rule")
    assert snapshot.content.index("src rule") < snapshot.content.index("module rule")
    assert "unrelated rule" not in snapshot.content
    assert snapshot.truncated is False


def test_repository_rules_are_bounded_by_utf8_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("规则" * 2000, encoding="utf-8")

    snapshot = RepositoryRuleLoader(
        workspace,
        user_rules_path=tmp_path / "missing.md",
        max_bytes=1024,
    ).load()

    assert snapshot.truncated is True
    assert snapshot.size_bytes <= 1024
    assert snapshot.content.endswith("[Repository Rules truncated at configured byte limit]")


class RuleAwareClient(ModelClient):
    def __init__(self) -> None:
        self.system_prompts: list[str] = []
        self.responses = [
            ModelResponse(
                tool_calls=[
                    NormalizedToolCall(
                        id="read_target",
                        name="read",
                        arguments={"source": "workspace", "target": "src/app.py"},
                    )
                ]
            ),
            ModelResponse(final_text="done"),
        ]

    def call_request(self, request) -> ModelResponse:
        messages = request.as_chat_messages()
        system = next(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "system"
        )
        self.system_prompts.append(system)
        return self.responses.pop(0)


def test_agent_loop_loads_nested_rules_after_target_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source_dir = workspace / "src"
    source_dir.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("root acceptance rule", encoding="utf-8")
    (source_dir / "AGENTS.md").write_text("nested acceptance rule", encoding="utf-8")
    (source_dir / "app.py").write_text("value = 1\n", encoding="utf-8")
    run_path = tmp_path / "run"
    run_path.mkdir()
    client = RuleAwareClient()

    result = AgentLoop(
        task="inspect the target",
        workspace=workspace,
        model_client=client,
        trace_writer=TraceWriter(run_path / "trace.jsonl"),
        data_dir=tmp_path / "data",
        repository_rule_loader=RepositoryRuleLoader(
            workspace,
            user_rules_path=tmp_path / "missing.md",
        ),
        no_skills=True,
        run_id="run_rules",
    ).run()

    assert result.status == "completed"
    assert len(client.system_prompts) == 2
    assert "root acceptance rule" in client.system_prompts[0]
    assert "nested acceptance rule" not in client.system_prompts[0]
    assert "root acceptance rule" in client.system_prompts[1]
    assert "nested acceptance rule" in client.system_prompts[1]
    assert client.system_prompts[1].index("root acceptance rule") < client.system_prompts[1].index(
        "nested acceptance rule"
    )
