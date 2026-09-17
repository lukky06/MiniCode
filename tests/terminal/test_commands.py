from pathlib import Path
from types import SimpleNamespace

from minicode_harness.output import NullOutputSink
from minicode_harness.policy import ApprovalPolicy, PermissionMode
from minicode_harness.state import ReplSessionStore
from minicode_harness.state import (
    CheckpointStore,
    RunCheckpoint,
    RunStore,
    StaticApprovalClient,
)
from minicode_harness.terminal import commands as terminal_commands
from minicode_harness.terminal.commands import SlashCommandRouter, parse_slash_command
from minicode_harness.terminal.types import TerminalContext, TerminalSessionSettings
from minicode_harness.trace import TraceWriter
from minicode_harness.tools import SandboxMode


def _context(tmp_path: Path, *, executor=None) -> TerminalContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return TerminalContext(
        workspace=workspace,
        provider="qwen",
        model="qwen-plus",
        write_enabled=True,
        repository_memory_enabled=True,
        subagents_enabled=True,
        mcp_config=None,
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
        run_store=RunStore(tmp_path / "runs"),
        executor=executor,
    )


def test_parse_slash_command_handles_quoted_arguments() -> None:
    command = parse_slash_command('/trace "run_20260711_001"')

    assert command.name == "trace"
    assert command.arguments == ["run_20260711_001"]


def test_parse_slash_command_rejects_normal_task() -> None:
    try:
        parse_slash_command("explain /status")
    except ValueError as exc:
        assert "must start" in str(exc)
    else:
        raise AssertionError("normal tasks must not be parsed as slash commands")


def test_router_handles_status_model_permissions_and_plan(tmp_path) -> None:
    context = _context(tmp_path)
    router = SlashCommandRouter()

    status = router.execute(parse_slash_command("/status"), context)
    model = router.execute(parse_slash_command("/model"), context)
    permissions = router.execute(parse_slash_command("/permissions"), context)
    sandbox_status = router.execute(parse_slash_command("/sandbox status"), context)
    plan_status = router.execute(parse_slash_command("/plan"), context)
    plan_on = router.execute(parse_slash_command("/plan on"), context)
    plan_status_on = router.execute(parse_slash_command("/plan status"), context)
    plan_off = router.execute(parse_slash_command("/plan off"), context)

    assert status.status == "completed"
    assert f"Workspace: {context.workspace}" in (status.content or "")
    assert "Provider: qwen" in (model.content or "")
    assert "Write tools enabled: True" in (permissions.content or "")
    assert "Permission mode: read-only" in (permissions.content or "")
    assert "Approval policy: on-request" in (permissions.content or "")
    assert "Command sandbox: local" in (permissions.content or "")
    assert "Command sandbox: local" in (sandbox_status.content or "")
    assert "Plan mode: off" in (plan_status.content or "")
    assert "enabled for subsequent Runs" in (plan_on.content or "")
    assert "Next Run mode: plan" in (plan_status_on.content or "")
    assert "disabled for subsequent Runs" in (plan_off.content or "")
    assert context.session_settings.collaboration_mode.value == "default"


def test_command_catalog_is_ordered_and_exposes_argument_choices() -> None:
    catalog = terminal_commands.command_catalog()

    assert [item.name for item in catalog[:6]] == [
        "permissions",
        "sandbox",
        "plan",
        "diff",
        "review",
        "context",
    ]
    permissions = next(item for item in catalog if item.name == "permissions")
    sandbox = next(item for item in catalog if item.name == "sandbox")
    assert permissions.argument_hint == "[mode <value> | approval <value>]"
    assert permissions.argument_choices == (
        "mode read-only",
        "mode workspace-write",
        "mode full-access",
        "approval on-request",
        "approval never",
    )
    assert permissions.availability == "idle"
    assert sandbox.argument_hint == "[status|local|docker [image]]"
    assert sandbox.argument_choices == ("status", "local", "docker")


def test_sandbox_command_updates_next_run_without_touching_current_run(tmp_path) -> None:
    settings = TerminalSessionSettings(
        sandbox_mode=SandboxMode.DOCKER,
        sandbox_image="minicode-sandbox:latest",
    )
    base = _context(tmp_path)
    context = TerminalContext(
        **{
            **base.__dict__,
            "sandbox_mode": SandboxMode.DOCKER,
            "sandbox_image": "minicode-sandbox:latest",
            "session_settings": settings,
        }
    )
    router = SlashCommandRouter()

    local = router.execute(parse_slash_command("/sandbox local"), context)
    status_local = router.execute(parse_slash_command("/sandbox status"), context)
    local_mode = settings.sandbox_mode
    docker = router.execute(
        parse_slash_command("/sandbox docker python:3.12-slim"),
        context,
    )
    status_docker = router.execute(parse_slash_command("/sandbox status"), context)

    assert local.status == "completed"
    assert local_mode == SandboxMode.LOCAL
    assert "Command sandbox: local" in (status_local.content or "")
    assert docker.status == "completed"
    assert settings.sandbox_mode == SandboxMode.DOCKER
    assert settings.sandbox_image == "python:3.12-slim"
    assert "Command sandbox: docker" in (status_docker.content or "")
    assert "Sandbox image: python:3.12-slim" in (status_docker.content or "")
    assert context.sandbox_mode == SandboxMode.DOCKER
    assert context.sandbox_image == "minicode-sandbox:latest"


def test_sandbox_command_requires_image_when_switching_to_docker(tmp_path) -> None:
    context = _context(tmp_path)
    result = SlashCommandRouter().execute(parse_slash_command("/sandbox docker"), context)

    assert result.status == "failed"
    assert "requires an image" in (result.content or "")


def test_permissions_command_updates_session_defaults_for_subsequent_runs(tmp_path) -> None:
    settings = TerminalSessionSettings()
    context = _context(tmp_path)
    context = TerminalContext(
        **{
            **context.__dict__,
            "session_settings": settings,
        }
    )
    router = SlashCommandRouter()

    write_mode = router.execute(
        parse_slash_command("/permissions mode workspace-write"),
        context,
    )
    approval = router.execute(
        parse_slash_command("/permissions approval never"),
        context,
    )
    status = router.execute(parse_slash_command("/permissions status"), context)

    assert write_mode.status == "completed"
    assert approval.status == "completed"
    assert settings.permission_mode == PermissionMode.WORKSPACE_WRITE
    assert settings.approval_policy == ApprovalPolicy.NEVER
    assert "Permission mode: workspace-write" in (status.content or "")
    assert "Approval policy: never" in (status.content or "")


def test_permissions_picker_applies_both_choices_atomically(tmp_path) -> None:
    class InterruptedInput:
        def __init__(self) -> None:
            self.calls = 0

        def choose(self, request):
            del request
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(selected_index=1)
            raise RuntimeError("selection cancelled")

    settings = TerminalSessionSettings()
    base = _context(tmp_path)
    context = TerminalContext(
        **{
            **base.__dict__,
            "session_settings": settings,
            "user_input_client": InterruptedInput(),
        }
    )

    try:
        SlashCommandRouter().execute(parse_slash_command("/permissions"), context)
    except RuntimeError as exc:
        assert "cancelled" in str(exc)
    else:
        raise AssertionError("interrupted picker must surface cancellation")

    assert settings.permission_mode == PermissionMode.READ_ONLY
    assert settings.approval_policy == ApprovalPolicy.ON_REQUEST


def test_permissions_command_rejects_invalid_values(tmp_path) -> None:
    router = SlashCommandRouter()
    context = _context(tmp_path)

    bad_mode = router.execute(
        parse_slash_command("/permissions mode root"),
        context,
    )
    bad_approval = router.execute(
        parse_slash_command("/permissions approval always"),
        context,
    )

    assert bad_mode.status == "failed"
    assert "read-only, workspace-write, or full-access" in (bad_mode.content or "")
    assert bad_approval.status == "failed"
    assert "on-request or never" in (bad_approval.content or "")


def test_router_rejects_invalid_plan_arguments(tmp_path) -> None:
    context = _context(tmp_path)
    router = SlashCommandRouter()

    invalid = router.execute(parse_slash_command("/plan execute"), context)
    too_many = router.execute(parse_slash_command("/plan on now"), context)

    assert invalid.status == "failed"
    assert "on, off, or status" in (invalid.content or "")
    assert too_many.status == "failed"
    assert "at most one argument" in (too_many.content or "")


def test_router_displays_latest_context_breakdown(tmp_path) -> None:
    context = _context(tmp_path)
    session = context.run_store.create_run(
        task="inspect context",
        workspace=context.workspace,
        run_id="run_20260726_001",
    )
    writer = TraceWriter(context.run_store.path_for(session.run_id) / "trace.jsonl")
    writer.write_event(
        "context_built",
        token_estimate=1_200,
        prompt_budget=2_000,
        source_tokens={"system": 200, "tools": 300, "current_turn": 700},
        token_estimator_version="mixed-language-v1",
    )

    result = SlashCommandRouter().execute(
        parse_slash_command("/context"),
        context,
    )

    assert result.status == "completed"
    assert "Usage: 1.2K / 2.0K (60.0%)" in (result.content or "")
    assert "System Prompt" in (result.content or "")


def test_router_runs_manual_compaction_with_focus(tmp_path) -> None:
    calls: list[dict[str, str | None]] = []

    class FakeExecutor:
        def compact_session(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                changed=True,
                before_tokens=4_000,
                after_tokens=1_200,
                removed_groups=8,
                reason="semantic",
            )

    context = _context(tmp_path, executor=FakeExecutor())
    result = SlashCommandRouter().execute(
        parse_slash_command('/compact "保留当前重构约束" "以及失败测试"'),
        context,
    )

    assert result.status == "completed"
    assert len(calls) == 1
    assert calls[0]["provider"] == "qwen"
    assert calls[0]["model"] == "qwen-plus"
    assert calls[0]["focus"] == "保留当前重构约束 以及失败测试"
    assert calls[0]["output_sink"] is context.output_sink
    assert "4000 -> 1200 tokens" in (result.content or "")
    assert "removed groups: 8" in (result.content or "")


def test_router_reports_manual_compaction_without_active_executor(tmp_path) -> None:
    result = SlashCommandRouter().execute(
        parse_slash_command("/compact"),
        _context(tmp_path),
    )

    assert result.status == "failed"
    assert "No active session executor" in (result.content or "")


def test_router_renames_and_forks_current_session(tmp_path) -> None:
    workspace = tmp_path / "workspace-session"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "session-data-command")
    session = session_store.create(workspace)
    session.replace_message_history(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
        ]
    )
    session.add_user_turn("first", run_id="run_1")
    session.add_assistant_turn("first answer", run_id="run_1", history_length=2)
    executor = SimpleNamespace(session_memory=session)
    context = TerminalContext(
        workspace=workspace,
        provider="qwen",
        model="qwen-plus",
        write_enabled=True,
        repository_memory_enabled=True,
        subagents_enabled=True,
        mcp_config=None,
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
        run_store=RunStore(tmp_path / "runs-session-command"),
        session_store=session_store,
        executor=executor,
        session_id=session.session_id,
    )
    router = SlashCommandRouter()

    renamed = router.execute(parse_slash_command('/rename "API cleanup"'), context)
    forked = router.execute(parse_slash_command("/fork 1"), context)

    assert renamed.status == "completed"
    assert session_store.load(workspace, session.session_id).name == "API cleanup"
    assert forked.status == "completed"
    assert forked.session_id is not None
    assert forked.session_id != session.session_id
    assert executor.session_memory.session_id == forked.session_id
    assert session_store.load(workspace, forked.session_id).load_message_history() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_router_review_uses_executor_readonly_review(tmp_path) -> None:
    calls = []

    class FakeExecutor:
        def review_current_diff(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status="completed",
                summary="P1 src/app.py: stale state\nGate recommendation: FAIL",
                run_id="run_20260908_001",
            )

    context = _context(tmp_path, executor=FakeExecutor())
    result = SlashCommandRouter().execute(
        parse_slash_command('/review "focus on recovery"'),
        context,
    )

    assert result.status == "completed"
    assert "P1 src/app.py" in (result.content or "")
    assert calls == [
        {
            "workspace": context.workspace,
            "provider": "qwen",
            "model": "qwen-plus",
            "focus": "focus on recovery",
            "cancellation_token": None,
        }
    ]


def test_router_reports_unknown_command_without_raising(tmp_path) -> None:
    result = SlashCommandRouter().execute(
        parse_slash_command("/unknown"),
        _context(tmp_path),
    )

    assert result.status == "failed"
    assert "Unknown command" in (result.content or "")


def test_router_scopes_run_inspection_to_current_session(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "session-data")
    first_session = session_store.create(workspace)
    first_session.rename("API cleanup")
    first_session.add_user_turn("first task")
    second_session = session_store.create(workspace)
    run_store = RunStore(tmp_path / "runs")
    first_run = run_store.create_run(
        task="first",
        workspace=workspace,
        run_id="run_20260827_001",
        conversation_session_id=first_session.session_id,
    )
    second_run = run_store.create_run(
        task="second",
        workspace=workspace,
        run_id="run_20260827_002",
        conversation_session_id=second_session.session_id,
    )
    context = TerminalContext(
        workspace=workspace,
        provider="qwen",
        model="qwen-plus",
        write_enabled=True,
        repository_memory_enabled=True,
        subagents_enabled=True,
        mcp_config=None,
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
        run_store=run_store,
        session_store=session_store,
        session_id=first_session.session_id,
    )
    router = SlashCommandRouter()

    runs = router.execute(parse_slash_command("/runs"), context)
    other = router.execute(
        parse_slash_command(f"/runs {second_run.run_id}"), context
    )
    sessions = router.execute(parse_slash_command("/sessions"), context)

    assert runs.status == "completed"
    assert first_run.run_id in (runs.content or "")
    assert second_run.run_id not in (runs.content or "")
    assert other.status == "failed"
    assert "another Session" in (other.content or "")
    assert sessions.status == "completed"
    assert first_session.session_id in (sessions.content or "")
    assert second_session.session_id in (sessions.content or "")
    assert "API cleanup" in (sessions.content or "")
    assert "first task" in (sessions.content or "")


def test_router_recovers_only_runs_in_current_session(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = ReplSessionStore(tmp_path / "session-data")
    first_session = session_store.create(workspace)
    second_session = session_store.create(workspace)
    run_store = RunStore(tmp_path / "runs")
    first_run = run_store.create_run(
        task="first",
        workspace=workspace,
        run_id="run_20260827_011",
        conversation_session_id=first_session.session_id,
    )
    second_run = run_store.create_run(
        task="second",
        workspace=workspace,
        run_id="run_20260827_012",
        conversation_session_id=second_session.session_id,
    )
    for run in (first_run, second_run):
        CheckpointStore(run_store.path_for(run.run_id) / "checkpoints").save(
            RunCheckpoint(
                run_id=run.run_id,
                step=1,
                task=run.task,
                workspace=run.workspace,
                status="stopped",
            )
        )
    context = TerminalContext(
        workspace=workspace,
        provider="qwen",
        model="qwen-plus",
        write_enabled=True,
        repository_memory_enabled=True,
        subagents_enabled=True,
        mcp_config=None,
        output_sink=NullOutputSink(),
        approval_client=StaticApprovalClient(),
        run_store=run_store,
        session_store=session_store,
        session_id=first_session.session_id,
    )
    recovered: list[str] = []

    def fake_resume_run(run_id, **kwargs):
        recovered.append(run_id)
        return SimpleNamespace(
            status="continued",
            reason="checkpoint",
            conflicts=[],
        )

    monkeypatch.setattr(terminal_commands, "resume_run", fake_resume_run)
    router = SlashCommandRouter()

    current = router.execute(parse_slash_command("/recover"), context)
    other = router.execute(
        parse_slash_command(f"/recover {second_run.run_id}"), context
    )

    assert current.status == "completed"
    assert recovered == [first_run.run_id]
    assert other.status == "failed"
    assert "minicode recover" in (other.content or "")
