"""JSONL stdio backend for the TypeScript terminal UI."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from threading import Lock, Thread
from typing import Callable, TextIO

from minicode_harness.policy import ApprovalPolicy, CommandRule, PermissionMode
from minicode_harness.runtime import CollaborationMode
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.run_executor import (
    RunExecutionRequest,
    RunExecutor,
)
from minicode_harness.runtime.steering import SteeringQueue
from minicode_harness.state import (
    ApprovalClient,
    ReplSessionStore,
    RunStore,
    UserInputClient,
    UserInputRequest,
)
from minicode_harness.terminal.commands import (
    SlashCommandRouter,
    command_catalog,
    parse_slash_command,
)
from minicode_harness.terminal.types import TerminalContext, TerminalSessionSettings
from minicode_harness.tools import SandboxMode

from .approval import JsonlApprovalClient
from .output import JsonlOutputSink
from .panels import panel_title
from .protocol import (
    ApprovalResponseMessage,
    CancelMessage,
    ClientMessage,
    CommandMessage,
    JsonlProtocolError,
    SteerMessage,
    TaskMessage,
    UserInputResponseMessage,
    parse_client_message,
)
from .user_input import JsonlUserInputClient
from .writer import JsonlEventWriter


class JsonlBackend:
    """Own one conversation Session and route JSONL input into the existing runtime."""

    def __init__(
        self,
        *,
        session_id: str,
        executor: RunExecutor,
        output_sink: JsonlOutputSink,
        approval_client: ApprovalClient,
        request_factory: Callable[[str], RunExecutionRequest],
        user_input_client: UserInputClient | None = None,
        command_router: SlashCommandRouter | None = None,
        command_context: TerminalContext | None = None,
    ) -> None:
        self.session_id = session_id
        self.executor = executor
        self.output_sink = output_sink
        self.approval_client = approval_client
        self.request_factory = request_factory
        self.user_input_client = user_input_client
        self.command_router = command_router
        self.command_context = command_context
        self._lock = Lock()
        self._active_thread: Thread | None = None
        self._active_cancellation: CancellationToken | None = None
        self._active_steering: SteeringQueue | None = None

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active_thread is not None and self._active_thread.is_alive()

    def run_forever(self, input_stream: TextIO) -> None:
        """Consume protocol records until EOF, then cooperatively stop any active Run."""

        self.output_sink.session_started(self.session_id)
        self.output_sink.command_catalog(command_catalog())
        if self.command_context is not None:
            self.output_sink.session_settings(self.command_context.session_settings)
        for raw_line in input_stream:
            line = raw_line.rstrip("\r\n")
            try:
                message = parse_client_message(line)
            except JsonlProtocolError as exc:
                self.output_sink.error(str(exc))
                continue
            self.handle_message(message)
        self.close()

    def handle_message(self, message: ClientMessage) -> None:
        if isinstance(message, TaskMessage):
            self._start_task(message.text)
            return
        if isinstance(message, SteerMessage):
            self._steer(message.text)
            return
        if isinstance(message, CancelMessage):
            self._cancel()
            return
        if isinstance(message, CommandMessage):
            self._command(message.text)
            return
        if isinstance(message, UserInputResponseMessage):
            resolver = getattr(self.user_input_client, "resolve", None)
            if not callable(resolver):
                self.output_sink.error("No structured user input is currently available.")
                return
            resolved, error = resolver(message.id, message.selected_index)
            if not resolved:
                self.output_sink.error(error or "User input response was rejected.")
            return
        if isinstance(message, ApprovalResponseMessage):
            resolver = getattr(self.approval_client, "resolve", None)
            if not callable(resolver):
                self.output_sink.error("No interactive approval is currently available.")
                return
            resolved, error = resolver(message.id, message.decision)
            if not resolved:
                self.output_sink.error(error or "Approval response was rejected.")

    def close(self, *, timeout: float = 2.0) -> None:
        with self._lock:
            token = self._active_cancellation
            thread = self._active_thread
        if token is not None:
            token.cancel()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))
        if thread is not None and thread.is_alive():
            self.output_sink.error(
                "Backend input closed while the active Run was still stopping.",
                fatal=True,
            )

    def _start_task(self, text: str) -> None:
        task = text.strip()
        if not task:
            self.output_sink.error("Task must not be empty.")
            return

        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                self.output_sink.error(
                    "A Run is already active; send a steer message instead."
                )
                return
            cancellation = CancellationToken()
            steering = SteeringQueue()
            thread = Thread(
                target=self._execute_task,
                args=(task, cancellation, steering),
                name="minicode-tui-run",
                daemon=True,
            )
            self._active_cancellation = cancellation
            self._active_steering = steering
            self._active_thread = thread
        thread.start()

    def _execute_task(
        self,
        task: str,
        cancellation: CancellationToken,
        steering: SteeringQueue,
    ) -> None:
        bind_cancellation = getattr(self.approval_client, "bind_cancellation", None)
        bind_user_input_cancellation = getattr(
            self.user_input_client,
            "bind_cancellation",
            None,
        )
        if callable(bind_cancellation):
            bind_cancellation(cancellation)
        if callable(bind_user_input_cancellation):
            bind_user_input_cancellation(cancellation)
        try:
            kwargs = {
                "output_sink": self.output_sink,
                "approval_client": self.approval_client,
                "cancellation_token": cancellation,
                "steering_queue": steering,
            }
            if self.user_input_client is not None:
                kwargs["user_input_client"] = self.user_input_client
            request = self.request_factory(task)
            if (
                request.sandbox_mode == SandboxMode.DOCKER
                and not request.sandbox_image
                and not request.dry_run
            ):
                self.output_sink.error(
                    "Docker command sandbox has no image configured. "
                    "Use /sandbox docker <image> to configure one, or /sandbox local "
                    "to use the host for subsequent Runs."
                )
                return
            result = self.executor.execute(request, **kwargs)
            session_settings = (
                self.command_context.session_settings
                if self.command_context is not None
                else None
            )
            if (
                self.user_input_client is not None
                and session_settings is not None
                and request.collaboration_mode == CollaborationMode.PLAN
                and getattr(result, "status", None) == "completed"
            ):
                response = self.user_input_client.choose(
                    UserInputRequest(
                        question="The plan is ready. What should MiniCode do next?",
                        options=[
                            {
                                "label": "Execute plan",
                                "description": "Switch to default mode and execute the approved plan.",
                            },
                            {
                                "label": "Continue planning",
                                "description": "Keep Plan Mode enabled for the next task.",
                            },
                            {
                                "label": "Finish planning",
                                "description": "Leave Plan Mode without executing the plan.",
                            },
                        ],
                    )
                )
                if response.selected_index == 0:
                    session_settings.collaboration_mode = CollaborationMode.DEFAULT
                    execution_steering = SteeringQueue()
                    with self._lock:
                        self._active_steering = execution_steering
                    kwargs["steering_queue"] = execution_steering
                    execution_request = self.request_factory(
                        "Execute the approved plan above."
                    ).model_copy(
                        update={"collaboration_mode": CollaborationMode.DEFAULT}
                    )
                    self.executor.execute(execution_request, **kwargs)
                elif response.selected_index == 2:
                    session_settings.collaboration_mode = CollaborationMode.DEFAULT
                self.output_sink.session_settings(session_settings)
        except Exception as exc:
            self.output_sink.error(f"{type(exc).__name__}: {exc}")
            self.output_sink.run_finished(
                status="failed",
                stop_reason=type(exc).__name__,
            )
        finally:
            if callable(bind_cancellation):
                bind_cancellation(None)
            if callable(bind_user_input_cancellation):
                bind_user_input_cancellation(None)
            with self._lock:
                self._active_cancellation = None
                self._active_steering = None
                self._active_thread = None

    def _steer(self, text: str) -> None:
        with self._lock:
            thread = self._active_thread
            steering = self._active_steering
        if thread is None or not thread.is_alive() or steering is None:
            self.output_sink.error("No active Run is available for steering.")
            return
        try:
            steering.enqueue(text)
        except (TypeError, ValueError) as exc:
            self.output_sink.error(str(exc))

    def _cancel(self) -> None:
        with self._lock:
            thread = self._active_thread
            cancellation = self._active_cancellation
        if thread is None or not thread.is_alive() or cancellation is None:
            return
        cancellation.cancel()

    def _command(self, text: str) -> None:
        if self.is_active:
            self.output_sink.error(
                "TUI commands are unavailable while a Run is active; send steering instead."
            )
            return
        if self.command_router is None or self.command_context is None:
            self.output_sink.error("No slash command router is configured.")
            return
        try:
            command = parse_slash_command(text)
        except ValueError as exc:
            self.output_sink.error(str(exc))
            return

        cancellation = CancellationToken()
        steering = SteeringQueue()
        thread = Thread(
            target=self._execute_command,
            args=(command, cancellation, steering),
            name="minicode-tui-command",
            daemon=True,
        )
        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                self.output_sink.error(
                    "TUI commands are unavailable while a Run is active; send steering instead."
                )
                return
            self._active_cancellation = cancellation
            self._active_steering = steering
            self._active_thread = thread
        thread.start()

    def _execute_command(
        self,
        command,
        cancellation: CancellationToken,
        steering: SteeringQueue,
    ) -> None:
        bind_cancellation = getattr(self.approval_client, "bind_cancellation", None)
        bind_user_input_cancellation = getattr(
            self.user_input_client,
            "bind_cancellation",
            None,
        )
        if callable(bind_cancellation):
            bind_cancellation(cancellation)
        if callable(bind_user_input_cancellation):
            bind_user_input_cancellation(cancellation)
        try:
            context = replace(
                self.command_context,
                cancellation_token=cancellation,
                steering_queue=steering,
            )
            result = self.command_router.execute(command, context)
            if result.status == "failed":
                self.output_sink.error(result.content or f"/{command.name} failed.")
            elif result.status == "exit":
                self.output_sink.exit_requested()
            else:
                if result.session_id is not None:
                    self.session_id = result.session_id
                    if self.command_context is not None:
                        self.command_context = replace(
                            self.command_context,
                            session_id=result.session_id,
                        )
                    self.output_sink.session_started(result.session_id)
                if command.name in {"permissions", "plan"}:
                    self.output_sink.session_settings(context.session_settings)
                self.output_sink.panel(
                    name=command.name,
                    title=panel_title(command.name),
                    content=result.content or "",
                )
        except Exception as exc:
            if not cancellation.is_cancelled:
                self.output_sink.error(f"{type(exc).__name__}: {exc}")
        finally:
            if callable(bind_cancellation):
                bind_cancellation(None)
            if callable(bind_user_input_cancellation):
                bind_user_input_cancellation(None)
            with self._lock:
                self._active_cancellation = None
                self._active_steering = None
                self._active_thread = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--model")
    parser.add_argument(
        "--approval-policy",
        choices=tuple(item.value for item in ApprovalPolicy),
        default=ApprovalPolicy.ON_REQUEST.value,
    )
    parser.add_argument(
        "--permission-mode",
        choices=tuple(item.value for item in PermissionMode),
        default=PermissionMode.READ_ONLY.value,
    )
    parser.add_argument(
        "--mode",
        choices=tuple(item.value for item in CollaborationMode),
        default=CollaborationMode.DEFAULT.value,
    )
    parser.add_argument(
        "--sandbox-mode",
        choices=tuple(item.value for item in SandboxMode),
        default=SandboxMode.DOCKER.value,
    )
    parser.add_argument("--sandbox-image")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--skill", action="append", default=None)
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--no-repository-memory", action="store_true")
    parser.add_argument("--no-subagents", action="store_true")
    parser.add_argument("--mcp-config", type=Path)
    parser.add_argument(
        "--session-mode",
        choices=("new", "continue", "exact"),
        default="new",
    )
    parser.add_argument("--session-id")
    return parser


def _launch_session(
    session_store: ReplSessionStore,
    workspace: Path,
    *,
    session_mode: str,
    session_id: str | None,
):
    if session_mode == "new":
        if session_id is not None:
            raise ValueError("A session ID cannot be supplied when starting a new session.")
        return session_store.create(workspace, persist=False)
    if session_mode == "continue":
        if session_id is not None:
            raise ValueError("A session ID cannot be combined with continue mode.")
        return session_store.load_latest(workspace)
    if not session_id:
        raise ValueError("Exact session mode requires a session ID.")
    return session_store.load(workspace, session_id)


def _build_backend(
    args: argparse.Namespace,
    *,
    output_sink: JsonlOutputSink,
) -> JsonlBackend:
    workspace = args.workspace.expanduser().resolve()
    session_store = ReplSessionStore()
    session = _launch_session(
        session_store,
        workspace,
        session_mode=args.session_mode,
        session_id=args.session_id,
    )
    run_store = RunStore(sessions_root=session_store.sessions_root)
    executor = RunExecutor(run_store=run_store, session_memory=session)
    approval_client = JsonlApprovalClient(output_sink)
    user_input_client = JsonlUserInputClient(output_sink)
    command_rules = _command_rules_from_environment()
    session_settings = TerminalSessionSettings(
        collaboration_mode=CollaborationMode(args.mode),
        permission_mode=PermissionMode(args.permission_mode),
        approval_policy=ApprovalPolicy(args.approval_policy),
        sandbox_mode=SandboxMode(args.sandbox_mode),
        sandbox_image=args.sandbox_image,
    )

    def request_factory(task: str) -> RunExecutionRequest:
        return RunExecutionRequest(
            task=task,
            workspace=workspace,
            provider=args.provider,
            model=args.model,
            write_enabled=not args.no_write,
            approval_policy=session_settings.approval_policy,
            permission_mode=session_settings.permission_mode,
            sandbox_mode=session_settings.sandbox_mode,
            sandbox_image=(
                session_settings.sandbox_image
                if session_settings.sandbox_mode == SandboxMode.DOCKER
                else None
            ),
            command_rules=command_rules,
            collaboration_mode=session_settings.collaboration_mode,
            skills=args.skill,
            skills_enabled=not args.no_skills,
            repository_memory_enabled=not args.no_repository_memory,
            subagents_enabled=not args.no_subagents,
            mcp_config=args.mcp_config,
            stream_model=True,
        )

    command_context = TerminalContext(
        workspace=workspace,
        provider=args.provider,
        model=args.model,
        write_enabled=not args.no_write,
        approval_policy=ApprovalPolicy(args.approval_policy),
        permission_mode=PermissionMode(args.permission_mode),
        sandbox_mode=SandboxMode(args.sandbox_mode),
        sandbox_image=args.sandbox_image,
        session_settings=session_settings,
        repository_memory_enabled=not args.no_repository_memory,
        subagents_enabled=not args.no_subagents,
        mcp_config=args.mcp_config,
        output_sink=output_sink,
        approval_client=approval_client,
        user_input_client=user_input_client,
        run_store=run_store,
        session_store=session_store,
        executor=executor,
        session_id=session.session_id,
    )
    return JsonlBackend(
        session_id=session.session_id,
        executor=executor,
        output_sink=output_sink,
        approval_client=approval_client,
        request_factory=request_factory,
        user_input_client=user_input_client,
        command_router=SlashCommandRouter(),
        command_context=command_context,
    )


def _command_rules_from_environment() -> list[CommandRule]:
    raw = os.environ.get("MINICODE_TUI_COMMAND_RULES")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MINICODE_TUI_COMMAND_RULES JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("MINICODE_TUI_COMMAND_RULES must be a JSON array.")
    return [CommandRule.model_validate(item) for item in payload]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol_stdout = sys.stdout
    writer = JsonlEventWriter(protocol_stdout)
    output_sink = JsonlOutputSink(writer)

    try:
        with redirect_stdout(sys.stderr):
            backend = _build_backend(args, output_sink=output_sink)
            backend.run_forever(sys.stdin)
    except Exception as exc:
        output_sink.error(f"{type(exc).__name__}: {exc}", fatal=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
