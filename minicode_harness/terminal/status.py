"""Deterministic terminal status rendering."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from minicode_harness.state import RunStore
from minicode_harness.workspace import scan_workspace_profile


@dataclass(frozen=True)
class TerminalStatus:
    workspace: str
    project: str
    state: str
    model: str
    session: str
    latest_run: str


def build_terminal_status(
    workspace: Path,
    *,
    run_store: RunStore,
    provider: str,
    model: str | None,
    session_id: str | None = None,
) -> TerminalStatus:
    resolved = workspace.resolve()
    return TerminalStatus(
        workspace=str(resolved),
        project=_project_label(resolved),
        state="ready",
        model=f"{provider} / {display_model_name(provider, model)}",
        session=_session_label(session_id),
        latest_run=_latest_run_label(run_store, session_id=session_id),
    )


def render_terminal_status(status: TerminalStatus) -> str:
    return "\n".join(
        [
            f"Workspace: {status.workspace}",
            f"Project: {status.project}",
            f"State: {status.state}",
            f"Model: {status.model}",
            f"Session: {status.session}",
            f"Latest run: {status.latest_run}",
        ]
    )


def display_model_name(provider: str, model: str | None) -> str:
    if model:
        return model
    normalized = provider.strip().lower()
    if normalized == "openai":
        return os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
    if normalized == "anthropic":
        return os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest"
    if normalized == "deepseek":
        return os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    if normalized in {"qwen", "dashscope", "tongyi"}:
        return os.environ.get("QWEN_MODEL") or os.environ.get("MINICODE_MODEL") or "qwen-plus"
    if normalized in {"kimi", "moonshot"}:
        return os.environ.get("KIMI_MODEL") or os.environ.get("MOONSHOT_MODEL") or "kimi-k3"
    if normalized == "ollama":
        return os.environ.get("OLLAMA_MODEL") or "unspecified"
    return "unspecified"


def _project_label(workspace: Path) -> str:
    try:
        profile = scan_workspace_profile(workspace)
    except (FileNotFoundError, OSError, ValueError):
        return "workspace unavailable"
    if profile.language and profile.build_system:
        return f"{profile.language} / {profile.build_system}"
    if profile.language:
        return profile.language
    if profile.build_system:
        return profile.build_system
    return "general code workspace"


def _session_label(session_id: str | None) -> str:
    if not session_id:
        return "none"
    return session_id if len(session_id) <= 20 else session_id[:20]


def _latest_run_label(run_store: RunStore, *, session_id: str | None) -> str:
    try:
        return run_store.latest_run_id(conversation_session_id=session_id)
    except FileNotFoundError:
        return "none"
