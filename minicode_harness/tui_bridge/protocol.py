"""Small, strict JSONL protocol shared with the TypeScript TUI."""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class _ProtocolMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionStarted(_ProtocolMessage):
    type: Literal["session_started"] = "session_started"
    session_id: str


class CommandCatalogItem(_ProtocolMessage):
    name: str
    description: str
    argument_hint: str | None = None
    argument_choices: list[str] = Field(default_factory=list)
    availability: Literal["idle", "active", "both"] = "idle"


class CommandCatalogEvent(_ProtocolMessage):
    type: Literal["command_catalog"] = "command_catalog"
    commands: list[CommandCatalogItem]


class SessionSettingsEvent(_ProtocolMessage):
    type: Literal["session_settings"] = "session_settings"
    permission_mode: str
    approval_policy: str
    collaboration_mode: str


class RunStarted(_ProtocolMessage):
    type: Literal["run_started"] = "run_started"
    run_id: str


class ContextEvent(_ProtocolMessage):
    type: Literal["context"] = "context"
    used: int = Field(ge=0)
    window: int = Field(gt=0)
    prompt_budget: int = Field(gt=0)
    reserved_output: int = Field(ge=0)


class ToolStarted(_ProtocolMessage):
    type: Literal["tool_started"] = "tool_started"
    id: str
    step: int = Field(ge=0)
    tool: str
    target: str | None = None


class ToolFinished(_ProtocolMessage):
    type: Literal["tool_finished"] = "tool_finished"
    id: str
    step: int = Field(ge=0)
    tool: str
    status: str
    summary: str | None = None
    command_status: str | None = None
    returncode: int | None = None
    duration_ms: int | None = Field(None, ge=0)
    runtime_task_id: str | None = None
    diff_preview: str | None = None
    diff_truncated: bool = False


class AssistantDelta(_ProtocolMessage):
    type: Literal["assistant_delta"] = "assistant_delta"
    text: str


class ReasoningDelta(_ProtocolMessage):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class ApprovalRequired(_ProtocolMessage):
    type: Literal["approval_required"] = "approval_required"
    id: str
    tool_call_id: str
    tool: str
    summary: str | None = None
    details: str | None = None
    can_approve_session: bool = False


class UserInputOptionPayload(_ProtocolMessage):
    label: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=240)


class UserInputRequired(_ProtocolMessage):
    type: Literal["user_input_required"] = "user_input_required"
    id: str
    question: str = Field(min_length=1, max_length=500)
    options: list[UserInputOptionPayload] = Field(min_length=2, max_length=4)


class RunFinished(_ProtocolMessage):
    type: Literal["run_finished"] = "run_finished"
    status: str
    run_id: str | None = None
    stop_reason: str | None = None


class ErrorEvent(_ProtocolMessage):
    type: Literal["error"] = "error"
    message: str
    fatal: bool = False


class PanelEvent(_ProtocolMessage):
    type: Literal["panel"] = "panel"
    name: str
    title: str
    content: str


class ExitRequested(_ProtocolMessage):
    type: Literal["exit_requested"] = "exit_requested"


class TaskMessage(_ProtocolMessage):
    type: Literal["task"] = "task"
    text: str = Field(min_length=1)


class SteerMessage(_ProtocolMessage):
    type: Literal["steer"] = "steer"
    text: str = Field(min_length=1)


class CommandMessage(_ProtocolMessage):
    type: Literal["command"] = "command"
    text: str = Field(min_length=1, max_length=256)


ApprovalDecision = Literal["approve", "approve_session", "reject", "skip", "abort"]


class ApprovalResponseMessage(_ProtocolMessage):
    type: Literal["approval_response"] = "approval_response"
    id: str
    decision: ApprovalDecision


class UserInputResponseMessage(_ProtocolMessage):
    type: Literal["user_input_response"] = "user_input_response"
    id: str
    selected_index: int = Field(ge=0, le=3)


class CancelMessage(_ProtocolMessage):
    type: Literal["cancel"] = "cancel"


ServerMessage: TypeAlias = Annotated[
    SessionStarted
    | CommandCatalogEvent
    | SessionSettingsEvent
    | RunStarted
    | ContextEvent
    | ToolStarted
    | ToolFinished
    | AssistantDelta
    | ReasoningDelta
    | ApprovalRequired
    | UserInputRequired
    | RunFinished
    | ErrorEvent
    | PanelEvent
    | ExitRequested,
    Field(discriminator="type"),
]

ClientMessage: TypeAlias = Annotated[
    TaskMessage
    | SteerMessage
    | CommandMessage
    | ApprovalResponseMessage
    | UserInputResponseMessage
    | CancelMessage,
    Field(discriminator="type"),
]

_SERVER_ADAPTER = TypeAdapter(ServerMessage)
_CLIENT_ADAPTER = TypeAdapter(ClientMessage)


class JsonlProtocolError(ValueError):
    """Raised when one transport line is not a valid protocol message."""


def encode_message(message: BaseModel) -> str:
    """Serialize one protocol message as exactly one UTF-8-safe JSONL record."""

    payload = json.dumps(
        message.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return payload + "\n"


def parse_server_message(line: str) -> ServerMessage:
    return _parse_line(line, _SERVER_ADAPTER)


def parse_client_message(line: str) -> ClientMessage:
    return _parse_line(line, _CLIENT_ADAPTER)


def _parse_line(line: str, adapter: TypeAdapter):
    if not isinstance(line, str):
        raise JsonlProtocolError("Protocol record must be text.")
    record = line[:-1] if line.endswith("\n") else line
    if record.endswith("\r"):
        record = record[:-1]
    if not record:
        raise JsonlProtocolError("Protocol record must not be empty.")
    if "\n" in record or "\r" in record:
        raise JsonlProtocolError("Protocol record must occupy exactly one physical line.")
    try:
        raw = json.loads(record)
    except json.JSONDecodeError as exc:
        raise JsonlProtocolError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise JsonlProtocolError("Protocol record must be a JSON object.")
    try:
        return adapter.validate_python(raw)
    except ValidationError as exc:
        raise JsonlProtocolError("Invalid protocol message.") from exc
