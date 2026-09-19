from pathlib import Path
import json

import pytest
from typer.testing import CliRunner

from minicode_harness import __version__
from minicode_harness.config import UserConfig, load_user_config
import minicode_harness.cli as cli_module
from minicode_harness.benchmark.models import BenchmarkSummary
from minicode_harness.benchmark.scenario_models import BenchmarkScenarioSummary
from minicode_harness.cli import (
    _main_help,
    _parse_interactive_task_args,
    _resolve_mcp_config,
    app,
)
from minicode_harness.state import ReplSessionStore
from minicode_harness.memory import RepositoryMemoryStore
from minicode_harness.terminal import SessionLaunchMode


runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_user_config(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_load_user_config", lambda: UserConfig())


def _only_standalone_run(minicode_home: Path) -> Path:
    session_files = list(
        (minicode_home / "sessions").glob("*/*/*/session_*/session.json")
    )
    assert session_files == []
    run_dirs = list((minicode_home / "runs").glob("run_*"))
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"minicode {__version__}" in result.output


def test_help_lists_phase_zero_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "exec" in result.output
    assert "resume" in result.output
    assert "recover" in result.output
    assert "trace" in result.output
    assert "report" in result.output
    assert "runs" in result.output
    assert "sessions" in result.output
    assert "memory" in result.output
    assert "bench" in result.output


def test_top_level_console_help_prefers_minicode_shorthand() -> None:
    help_text = _main_help("minicode")

    assert 'minicode "<task>" [options]' in help_text
    assert "minicode --continue" in help_text
    assert "minicode --session <session_id>" not in help_text
    assert "minicode resume [session_id]" in help_text
    assert "minicode recover [run_id]" in help_text
    assert "minicode sessions [--workspace <dir>]" in help_text
    assert "minicode runs [run_id] [--workspace <dir>]" in help_text
    assert "minicode memory [--workspace <dir>]" in help_text
    assert "~/.minicode/config.toml" in help_text
    assert "MINICODE_PROVIDER" in help_text
    assert "MINICODE_RUNS_DIR" in help_text


def test_user_config_toml_supplies_interactive_defaults(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "deepseek"',
                'model = "deepseek-chat"',
                'permission_mode = "workspace-write"',
                'approval_policy = "never"',
                'sandbox = "docker"',
                'sandbox_image = "python:3.12"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_user_config(config_path)
    monkeypatch.setattr(cli_module, "_load_user_config", lambda: config)
    captured = {}
    monkeypatch.setattr(
        cli_module,
        "launch_tui",
        lambda **kwargs: captured.update(kwargs),
    )

    launch = _parse_interactive_task_args(["plan this change"])
    cli_module._run_terminal(launch)

    assert captured["provider"] == "deepseek"
    assert captured["model"] == "deepseek-chat"
    assert captured["permission_mode"] == "workspace-write"
    assert captured["approval_policy"] == "never"
    assert captured["sandbox_mode"] == "docker"
    assert captured["sandbox_image"] == "python:3.12"


def test_interactive_terminal_allows_unconfigured_default_docker(monkeypatch) -> None:
    monkeypatch.delenv("MINICODE_SANDBOX_IMAGE", raising=False)
    monkeypatch.setattr(cli_module, "_load_user_config", lambda: UserConfig())
    captured = {}
    monkeypatch.setattr(
        cli_module,
        "launch_tui",
        lambda **kwargs: captured.update(kwargs),
    )

    launch = _parse_interactive_task_args([])
    cli_module._run_terminal(launch)

    assert captured["sandbox_mode"] == "docker"
    assert captured["sandbox_image"] is None


def test_user_config_toml_accepts_declarative_command_rules(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'sandbox = "docker"',
                '[[command_rules]]',
                'decision = "ask"',
                'prefix = ["npm", "install"]',
                '[[command_rules]]',
                'decision = "deny"',
                'prefix = ["git", "push"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_user_config(config_path)

    assert [(rule.decision.value, rule.prefix) for rule in config.command_rules] == [
        ("ask", ("npm", "install")),
        ("deny", ("git", "push")),
    ]


def test_user_config_precedence_keeps_cli_and_environment_above_file(monkeypatch) -> None:
    config = UserConfig(
        provider="deepseek",
        model="config-model",
        permission_mode="workspace-write",
        approval_policy="never",
        sandbox="docker",
        sandbox_image="config-image:latest",
    )
    monkeypatch.setenv("MINICODE_PROVIDER", "ollama")
    monkeypatch.setenv("MINICODE_MODEL", "env-model")
    monkeypatch.setenv("MINICODE_SANDBOX_IMAGE", "env-image:latest")

    assert cli_module._resolve_provider(None, config=config) == "ollama"
    assert cli_module._resolve_provider("openai", config=config) == "openai"
    assert cli_module._resolve_model(None, config=config) == "env-model"
    assert cli_module._resolve_model("cli-model", config=config) == "cli-model"
    assert (
        cli_module._resolve_permission_mode("full-access", config=config).value
        == "full-access"
    )
    assert (
        cli_module._resolve_approval_policy("on-request", config=config).value
        == "on-request"
    )
    assert cli_module._resolve_sandbox_mode("local", config=config).value == "local"
    assert (
        cli_module._resolve_sandbox_image(
            cli_module.SandboxMode.DOCKER,
            None,
            config=config,
        )
        == "env-image:latest"
    )
    assert (
        cli_module._resolve_sandbox_image(
            cli_module.SandboxMode.DOCKER,
            "cli-image:latest",
            config=config,
        )
        == "cli-image:latest"
    )


def test_code_default_uses_docker_and_requires_configured_image(monkeypatch) -> None:
    monkeypatch.delenv("MINICODE_SANDBOX_IMAGE", raising=False)
    config = UserConfig()

    assert cli_module._resolve_sandbox_mode(None, config=config).value == "docker"
    with pytest.raises(ValueError, match="Docker is the default command sandbox"):
        cli_module._resolve_sandbox_image(
            cli_module.SandboxMode.DOCKER,
            None,
            config=config,
        )
    assert (
        cli_module._resolve_sandbox_image(
            cli_module.SandboxMode.LOCAL,
            None,
            config=config,
        )
        is None
    )


def test_user_config_rejects_unknown_fields(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('provider = "deepseek"\nprofiles = "unsupported"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid MiniCode user config"):
        load_user_config(config_path)


def test_exec_uses_user_config_when_options_are_omitted(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINICODE_PROVIDER", raising=False)
    monkeypatch.delenv("MINICODE_MODEL", raising=False)
    minicode_home = tmp_path.parent / f"{tmp_path.name}-config-home"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))
    monkeypatch.setattr(
        cli_module,
        "_load_user_config",
        lambda: UserConfig(
            provider="deepseek",
            model="deepseek-chat",
            permission_mode="workspace-write",
            approval_policy="never",
            sandbox="local",
        ),
        raising=False,
    )

    result = runner.invoke(
        app,
        ["exec", "noop", "--workspace", ".", "--dry-run"],
    )

    assert result.exit_code == 0
    run_dir = _only_standalone_run(minicode_home)
    session_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert session_json["provider"] == "deepseek"
    assert session_json["model"] == "deepseek-chat"
    assert session_json["permission_mode"] == "workspace-write"
    assert session_json["approval_policy"] == "never"
    assert session_json["sandbox_mode"] == "local"


def test_exec_command_creates_standalone_run_before_agent_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINICODE_PROVIDER", raising=False)
    monkeypatch.delenv("MINICODE_MODEL", raising=False)
    minicode_home = tmp_path.parent / f"{tmp_path.name}-minicode-home"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))
    result = runner.invoke(app, ["exec", "noop", "--workspace", ".", "--dry-run"])

    assert result.exit_code == 0
    assert "Created run run_" in result.output
    assert "Dry run: agent execution skipped." in result.output

    run_dir = _only_standalone_run(minicode_home)
    assert run_dir.is_dir()
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "trace.jsonl").is_file()
    assert (run_dir / "artifacts").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "approvals").is_dir()
    assert (run_dir / "debug").is_dir()

    session_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert session_json["task"] == "noop"
    assert session_json["workspace"] == str(tmp_path.resolve())
    assert session_json["conversation_session_id"] is None
    assert session_json["collaboration_mode"] == "default"
    assert session_json["repository_memory_enabled"] is True
    assert session_json["subagents_enabled"] is True
    assert session_json["mcp_config"] is None

    trace_lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 1
    trace_event = json.loads(trace_lines[0])
    assert trace_event["type"] == "run_started"
    assert trace_event["run_id"] == session_json["run_id"]
    assert trace_event["task"] == "noop"
    assert trace_event["workspace"] == str(tmp_path.resolve())
    assert trace_event["provider"] == "qwen"
    assert trace_event["model"] is None
    assert trace_event["dry_run"] is True
    assert trace_event["collaboration_mode"] == "default"
    assert trace_event["repository_memory_enabled"] is True
    assert trace_event["context_architecture"] == "canonical_messages"
    assert trace_event["subagents_enabled"] is True
    assert trace_event["mcp_config"] is None
    assert "enable_readonly_subagent" not in trace_event


def test_exec_dry_run_persists_plan_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    minicode_home = tmp_path.parent / f"{tmp_path.name}-minicode-plan-mode"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))

    result = runner.invoke(
        app,
        ["exec", "plan change", "--workspace", ".", "--dry-run", "--mode", "plan"],
    )

    assert result.exit_code == 0
    run_dir = _only_standalone_run(minicode_home)
    session_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    trace_event = json.loads((run_dir / "trace.jsonl").read_text(encoding="utf-8"))
    assert session_json["collaboration_mode"] == "plan"
    assert trace_event["collaboration_mode"] == "plan"


def test_exec_dry_run_persists_command_sandbox(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    minicode_home = tmp_path.parent / f"{tmp_path.name}-minicode-sandbox"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))

    result = runner.invoke(
        app,
        [
            "exec",
            "verify safely",
            "--workspace",
            ".",
            "--dry-run",
            "--sandbox",
            "docker",
            "--sandbox-image",
            "python:3.11-slim",
        ],
    )

    assert result.exit_code == 0
    run_dir = _only_standalone_run(minicode_home)
    session_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    trace_event = json.loads((run_dir / "trace.jsonl").read_text(encoding="utf-8"))
    assert session_json["sandbox_mode"] == "docker"
    assert session_json["sandbox_image"] == "python:3.11-slim"
    assert trace_event["sandbox_mode"] == "docker"
    assert trace_event["sandbox_image"] == "python:3.11-slim"


def test_exec_dry_run_persists_repository_memory_opt_out(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    minicode_home = tmp_path.parent / f"{tmp_path.name}-minicode-memory-options"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))

    result = runner.invoke(
        app,
        [
            "exec",
            "noop",
            "--workspace",
            ".",
            "--dry-run",
            "--no-repository-memory",
        ],
    )

    assert result.exit_code == 0
    run_dir = _only_standalone_run(minicode_home)
    session_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    trace_event = json.loads((run_dir / "trace.jsonl").read_text(encoding="utf-8"))
    assert session_json["repository_memory_enabled"] is False
    assert trace_event["repository_memory_enabled"] is False

    retired = runner.invoke(
        app,
        ["exec", "noop", "--workspace", ".", "--dry-run", "--no-project-conventions"],
    )
    assert retired.exit_code != 0


def test_exec_dry_run_persists_mcp_and_subagent_options(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    minicode_home = tmp_path.parent / f"{tmp_path.name}-minicode-options"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))
    config = tmp_path / "mcp.json"
    config.write_text('{"servers": {}}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "exec",
            "noop",
            "--workspace",
            ".",
            "--dry-run",
            "--mcp-config",
            str(config),
            "--no-subagents",
        ],
    )

    assert result.exit_code == 0
    run_dir = _only_standalone_run(minicode_home)
    session_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    trace_event = json.loads((run_dir / "trace.jsonl").read_text(encoding="utf-8"))
    assert session_json["mcp_config"] == str(config.resolve())
    assert session_json["subagents_enabled"] is False
    assert trace_event["mcp_config"] == str(config.resolve())
    assert trace_event["subagents_enabled"] is False


def test_removed_readonly_subagent_flags_are_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    minicode_home = tmp_path.parent / f"{tmp_path.name}-minicode-home"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))

    result = runner.invoke(
        app,
        [
            "run",
            "noop",
            "--workspace",
            ".",
            "--dry-run",
            "--subagent",
            "--subagent-max-steps",
            "2",
            "--subagent-max-tool-calls",
            "3",
            "--no-subagent-digest",
            "--no-subagent-pre-context",
            "--no-subagent-on-duplicate",
            "--no-subagent-on-failure",
        ],
    )

    assert result.exit_code == 2
    assert not list((minicode_home / "sessions").glob("*/*/*/session_*/runs/run_*"))
    assert not list((minicode_home / "runs").glob("run_*"))


def test_mcp_config_resolution_uses_explicit_path_then_environment(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "explicit.json"
    configured = tmp_path / "configured.json"
    monkeypatch.setenv("MINICODE_MCP_CONFIG", str(configured))

    assert _resolve_mcp_config(explicit) == explicit.resolve()
    assert _resolve_mcp_config(None) == configured.resolve()


def test_interactive_task_parser_accepts_options_before_and_after_task() -> None:
    launch = _parse_interactive_task_args(
        [
            "--provider",
            "deepseek",
            "explain",
            "this repo",
            "--workspace",
            "workspace",
            "--no-write",
            "--permission-mode",
            "workspace-write",
            "--approval-policy",
            "never",
            "--mode",
            "plan",
            "--skills",
            "repo-explain",
            "--no-subagents",
        ]
    )

    assert launch.task == "explain this repo"
    assert launch.workspace == Path("workspace")
    assert launch.provider == "deepseek"
    assert launch.no_write is True
    assert launch.permission_mode.value == "workspace-write"
    assert launch.approval_policy.value == "never"
    assert launch.collaboration_mode.value == "plan"
    assert launch.skills == "repo-explain"
    assert launch.no_subagents is True


def test_interactive_task_parser_uses_last_workspace_alias() -> None:
    launch = _parse_interactive_task_args(
        ["--workspace", "first", "explain", "repo", "-w", "second"]
    )

    assert launch.workspace == Path("second")


def test_session_launch_options_reject_invalid_forms() -> None:
    with pytest.raises(ValueError, match="Unknown interactive option"):
        _parse_interactive_task_args(["--session", "session_abcdef123456"])
    with pytest.raises(ValueError, match="requires a value"):
        _parse_interactive_task_args(["--workspace", "--no-write"])
    with pytest.raises(ValueError, match="only with `minicode exec`"):
        _parse_interactive_task_args(["fix", "tests", "--dry-run"])
    with pytest.raises(ValueError, match="Invalid permission mode"):
        _parse_interactive_task_args(["--permission-mode", "unsafe"])
    with pytest.raises(ValueError, match="Invalid approval policy"):
        _parse_interactive_task_args(["--approval-policy", "always"])
    with pytest.raises(ValueError, match="Invalid collaboration mode"):
        _parse_interactive_task_args(["--mode", "execute"])


def test_interactive_parser_preserves_continue_launch_mode() -> None:
    continued = _parse_interactive_task_args(["--continue", "--no-color"])

    assert continued.session_mode == SessionLaunchMode.CONTINUE
    assert continued.no_color is True
    assert continued.permission_mode.value == "read-only"
    assert continued.approval_policy.value == "on-request"
    assert continued.collaboration_mode.value == "default"


def test_trace_and_report_commands_read_run_store(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    minicode_home = tmp_path / "minicode-home"
    monkeypatch.setenv("MINICODE_HOME", str(minicode_home))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_result = runner.invoke(
        app,
        ["exec", "noop", "--workspace", str(workspace), "--dry-run"],
    )
    assert run_result.exit_code == 0
    run_dir = _only_standalone_run(minicode_home)
    run_id = run_dir.name

    trace_result = runner.invoke(app, ["trace", run_id])
    report_result = runner.invoke(app, ["report", run_id])

    assert trace_result.exit_code == 0
    assert f"Run Trace: {run_id}" in trace_result.output
    assert "run_started" in trace_result.output
    assert report_result.exit_code == 0
    assert "Report generated:" in report_result.output
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "final.diff").is_file()

    latest_trace_result = runner.invoke(
        app, ["trace", "--workspace", str(workspace)]
    )
    latest_report_result = runner.invoke(
        app, ["report", "--workspace", str(workspace)]
    )

    assert latest_trace_result.exit_code == 0
    assert f"Run Trace: {run_id}" in latest_trace_result.output
    assert latest_report_result.exit_code == 0
    assert "Report generated:" in latest_report_result.output


def test_history_and_memory_commands_read_local_state(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "minicode-home"
    monkeypatch.setenv("MINICODE_HOME", str(data_dir))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    run_result = runner.invoke(
        app,
        ["exec", "remember the repository rule", "--workspace", str(workspace), "--dry-run"],
    )
    assert run_result.exit_code == 0
    run_dir = _only_standalone_run(data_dir)
    run_id = run_dir.name
    conversation = ReplSessionStore(data_dir).create(workspace)
    conversation.add_user_turn("remember the repository rule")
    session_id = conversation.session_id

    repository = RepositoryMemoryStore(workspace, data_dir=data_dir)
    entry = repository.topic_store.add_entry(
        topic="build-and-test",
        entry_type="procedure",
        summary="Known: prefer focused verification.",
        evidence_ids=["test"],
    )
    repository.refresh_index()

    runs_result = runner.invoke(
        app, ["runs", "--workspace", str(workspace)]
    )
    detail_result = runner.invoke(app, ["runs", run_id])
    sessions_result = runner.invoke(
        app, ["sessions", "--workspace", str(workspace)]
    )
    memory_result = runner.invoke(app, ["memory", "--workspace", str(workspace)])

    assert runs_result.exit_code == 0
    assert "Run History" in runs_result.output
    assert run_id in runs_result.output
    assert "remember the repository rule" in runs_result.output
    assert detail_result.exit_code == 0
    assert f"Run: {run_id}" in detail_result.output
    assert "No checkpoint recorded." in detail_result.output
    assert sessions_result.exit_code == 0
    assert "Sessions" in sessions_result.output
    assert session_id in sessions_result.output
    assert memory_result.exit_code == 0
    assert "Repository Memory" in memory_result.output
    assert "build-and-test" in memory_result.output

    remember_result = runner.invoke(
        app,
        [
            "memory",
            "remember",
            "Prefer targeted tests.",
            "--workspace",
            str(workspace),
            "--topic",
            "build-and-test",
        ],
    )
    assert remember_result.exit_code == 0
    assert "Memory saved" in remember_result.output

    forget_result = runner.invoke(
        app,
        ["memory", "forget", entry.entry_id, "--workspace", str(workspace)],
    )
    assert forget_result.exit_code == 0
    assert repository.topic_store.get_entry("build-and-test", entry.entry_id).status == "inactive"


def test_bench_run_command_invokes_runner(monkeypatch) -> None:
    calls = {}

    class FakeBenchmarkRunner:
        def __init__(self, config) -> None:
            calls["config"] = config

        def run_suite(self, suite, output):
            calls["suite"] = suite
            calls["output"] = output
            return BenchmarkSummary(
                suite="basic-java",
                total_tasks=10,
                resolved=8,
                failed=2,
                resolve_rate=0.8,
                by_category={},
                avg_steps=3.0,
                avg_tool_calls=4.0,
                avg_context_tokens=100.0,
                compression_count=1,
                checkpoint_count=2,
                elapsed_seconds=1.0,
                tasks=[],
            )

    monkeypatch.setattr("minicode_harness.cli.BenchmarkRunner", FakeBenchmarkRunner)

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "benchmarks/suites/basic-java",
            "--output",
            "runs/bench_demo",
            "--provider",
            "ollama",
            "--model",
            "qwen2.5-coder",
        ],
    )

    assert result.exit_code == 0
    assert "Benchmark Suite: basic-java" in result.output
    assert "Resolve Rate: 80%" in result.output
    assert calls["config"].provider == "ollama"
    assert calls["config"].model == "qwen2.5-coder"
    assert calls["config"].max_steps == 20
    assert calls["config"].max_tool_calls == 30
    assert calls["config"].enable_subagents is False
    assert Path(calls["suite"]).as_posix() == "benchmarks/suites/basic-java"
    assert Path(calls["output"]).as_posix() == "runs/bench_demo"


def test_bench_scenario_command_invokes_shared_session_runner(monkeypatch) -> None:
    calls = {}

    class FakeScenarioRunner:
        def __init__(self, config) -> None:
            calls["config"] = config

        def run_suite(self, suite, output):
            calls["suite"] = suite
            calls["output"] = output
            return BenchmarkScenarioSummary(
                suite="memory-context",
                total_runs=6,
                resolved_runs=5,
                resolve_rate=5 / 6,
                elapsed_seconds=1.0,
                variants=[],
                scenarios=[],
            )

    monkeypatch.setattr(
        "minicode_harness.cli.BenchmarkScenarioRunner",
        FakeScenarioRunner,
    )

    result = runner.invoke(
        app,
        [
            "bench",
            "scenario",
            "benchmarks/suites/memory-context",
            "--output",
            "runs/memory-context",
            "--provider",
            "ollama",
            "--model",
            "qwen2.5-coder",
            "--ablation",
            "--baseline",
        ],
    )

    assert result.exit_code == 0
    assert "Scenario Suite: memory-context" in result.output
    assert "Scenario Runs: 6" in result.output
    assert calls["config"].ablation_matrix is True
    assert calls["config"].baseline_mode is True
    assert calls["config"].provider == "ollama"
    assert calls["config"].model == "qwen2.5-coder"
    assert Path(calls["suite"]).as_posix() == "benchmarks/suites/memory-context"
    assert Path(calls["output"]).as_posix() == "runs/memory-context"


def test_bench_scenario_accepts_memory_v2_mode_override(monkeypatch) -> None:
    calls = {}

    class FakeScenarioRunner:
        def __init__(self, config) -> None:
            calls["config"] = config

        def run_suite(self, suite, output):
            return BenchmarkScenarioSummary(
                suite="memory-v2-eval",
                total_runs=1,
                resolved_runs=1,
                resolve_rate=1.0,
                elapsed_seconds=0.1,
                variants=[],
                scenarios=[],
            )

    monkeypatch.setattr(
        "minicode_harness.cli.BenchmarkScenarioRunner",
        FakeScenarioRunner,
    )

    result = runner.invoke(
        app,
        [
            "bench",
            "scenario",
            "benchmarks/suites/memory-v2-eval",
            "--memory-mode",
            "index_topic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["config"].memory_mode == "index_topic"


def test_bench_run_uses_minicode_environment_defaults(monkeypatch) -> None:
    calls = {}

    class FakeBenchmarkRunner:
        def __init__(self, config) -> None:
            calls["config"] = config

        def run_suite(self, suite, output):
            return BenchmarkSummary(
                suite="basic-java",
                total_tasks=1,
                resolved=1,
                failed=0,
                resolve_rate=1.0,
                by_category={},
                avg_steps=1.0,
                avg_tool_calls=1.0,
                avg_context_tokens=100.0,
                compression_count=0,
                checkpoint_count=1,
                elapsed_seconds=1.0,
                tasks=[],
            )

    monkeypatch.setenv("MINICODE_PROVIDER", "ollama")
    monkeypatch.setenv("MINICODE_MODEL", "qwen2.5-coder")
    monkeypatch.setattr("minicode_harness.cli.BenchmarkRunner", FakeBenchmarkRunner)

    result = runner.invoke(app, ["bench", "run", "benchmarks/suites/basic-java"])

    assert result.exit_code == 0
    assert calls["config"].provider == "ollama"
    assert calls["config"].model == "qwen2.5-coder"
