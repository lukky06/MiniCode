from types import SimpleNamespace

import pytest

from minicode_harness.policy import CommandRule, CommandRuleDecision
from minicode_harness.tui_bridge import launcher


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("v22.19.0", (22, 19, 0)),
        ("24.14.0", (24, 14, 0)),
        ("unknown", None),
    ],
)
def test_parse_node_version(raw, expected) -> None:
    assert launcher._parse_node_version(raw) == expected


def test_missing_node_points_to_exec(monkeypatch) -> None:
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="minicode exec"):
        launcher._find_compatible_node()


def test_old_node_points_to_exec(monkeypatch) -> None:
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "node")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="v22.18.0\n",
        ),
    )

    with pytest.raises(RuntimeError, match="Node.js >= 22.19.0"):
        launcher._find_compatible_node()


def test_environment_carries_runtime_options_without_stale_flags(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env = launcher._build_environment(
        workspace=workspace,
        provider="deepseek",
        model="deepseek-reasoner",
        write_enabled=False,
        approval_policy="never",
        permission_mode="workspace-write",
        collaboration_mode="plan",
        sandbox_mode="docker",
        sandbox_image="python:3.11-slim",
        command_rules=[
            CommandRule(
                decision=CommandRuleDecision.ASK,
                prefix=("npm", "install"),
            )
        ],
        skills=["reviewer", "handoff"],
        skills_enabled=True,
        repository_memory_enabled=False,
        subagents_enabled=False,
        mcp_config=tmp_path / "mcp.json",
        session_mode="exact",
        session_id="session_abc",
        initial_task="fix the test",
        no_color=False,
        base={"NO_COLOR": "1", "MINICODE_TUI_NO_SKILLS": "1"},
    )

    assert env["MINICODE_WORKSPACE"] == str(workspace)
    assert env["MINICODE_PROVIDER"] == "deepseek"
    assert env["MINICODE_MODEL"] == "deepseek-reasoner"
    assert env["MINICODE_TUI_APPROVAL_POLICY"] == "never"
    assert env["MINICODE_TUI_PERMISSION_MODE"] == "workspace-write"
    assert env["MINICODE_TUI_COLLABORATION_MODE"] == "plan"
    assert env["MINICODE_TUI_SANDBOX_MODE"] == "docker"
    assert env["MINICODE_TUI_SANDBOX_IMAGE"] == "python:3.11-slim"
    assert env["MINICODE_TUI_COMMAND_RULES"] == '[{"decision": "ask", "prefix": ["npm", "install"]}]'
    assert env["MINICODE_TUI_SKILLS"] == '["reviewer", "handoff"]'
    assert env["MINICODE_TUI_NO_WRITE"] == "1"
    assert env["MINICODE_TUI_NO_REPOSITORY_MEMORY"] == "1"
    assert env["MINICODE_TUI_NO_SUBAGENTS"] == "1"
    assert env["MINICODE_TUI_SESSION_MODE"] == "exact"
    assert env["MINICODE_TUI_SESSION_ID"] == "session_abc"
    assert env["MINICODE_TUI_INITIAL_TASK"] == "fix the test"
    assert "MINICODE_TUI_NO_SKILLS" not in env
    assert "NO_COLOR" not in env


def test_launch_tui_runs_packaged_entrypoint_in_workspace(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    entrypoint = tmp_path / "main.mjs"
    entrypoint.write_text("// bundled", encoding="utf-8")
    calls = {}

    monkeypatch.setattr(launcher, "_find_compatible_node", lambda: "node")
    monkeypatch.setattr(launcher, "_runtime_entrypoint", lambda: entrypoint)

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    launcher.launch_tui(
        workspace=workspace,
        provider="qwen",
        model=None,
        write_enabled=True,
        approval_policy="on-request",
        permission_mode="read-only",
        collaboration_mode="default",
        sandbox_mode="local",
        sandbox_image=None,
        skills=None,
        skills_enabled=True,
        repository_memory_enabled=True,
        subagents_enabled=True,
        mcp_config=None,
        session_mode="new",
        session_id=None,
        initial_task=None,
        no_color=False,
    )

    assert calls["argv"] == ["node", str(entrypoint)]
    assert calls["kwargs"]["cwd"] == workspace
    assert calls["kwargs"]["check"] is False
