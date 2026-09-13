"""Display helpers for run history and project memory."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from minicode_harness.memory import RepositoryMemoryStore
from minicode_harness.state import (
    CheckpointStore,
    ReplSessionStore,
    RunCheckpoint,
    RunSession,
    RunStore,
)


def format_run_history(
    limit: int = 10,
    run_store: RunStore | None = None,
    *,
    workspace: Path | str | None = None,
    conversation_session_id: str | None = None,
) -> str:
    """Return a compact table of recent Runs in the selected scope."""

    if limit < 1:
        raise ValueError("History limit must be at least 1.")

    store = run_store or RunStore()
    run_ids = store.list_run_ids(
        workspace=workspace,
        conversation_session_id=conversation_session_id,
    )
    if not run_ids:
        return "Run History\nNo matching runs found."

    selected_run_ids = list(reversed(run_ids[-limit:]))
    lines = [
        "Run History",
        "",
        "| Run ID | Status | Reason | Created | Task |",
        "|---|---|---|---|---|",
    ]
    for run_id in selected_run_ids:
        session = store.load_session(run_id)
        checkpoint = _load_latest_checkpoint(store, run_id)
        status, reason = _session_status(session, checkpoint)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run_id}`",
                    _table_cell(status),
                    _table_cell(reason or "-"),
                    _table_cell(_format_datetime(session.created_at)),
                    _table_cell(_single_line(session.task, 72)),
                ]
            )
            + " |"
        )
    lines.extend(["", f"Showing {len(selected_run_ids)} of {len(run_ids)} runs."])
    return "\n".join(lines)


def format_run_detail(
    run_id: str,
    *,
    run_store: RunStore | None = None,
    observation_limit: int = 5,
    conversation_session_id: str | None = None,
) -> str:
    """Return detailed Run metadata and recent checkpoint observations."""

    if observation_limit < 0:
        raise ValueError("Observation limit must be non-negative.")

    store = run_store or RunStore()
    session = store.load_session(run_id)
    if (
        conversation_session_id is not None
        and session.conversation_session_id != conversation_session_id
    ):
        raise ValueError("Run belongs to another Session.")
    run_path = store.path_for(run_id)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run not found: {run_path}")

    checkpoint = _load_latest_checkpoint(store, run_id)
    status, reason = _session_status(session, checkpoint)
    lines = [
        f"Run: {session.run_id}",
        "",
        "Task:",
        session.task,
        "",
        "Metadata:",
        f"- Workspace: `{session.workspace}`",
        f"- Provider: `{session.provider}`",
        f"- Model: `{session.model or 'default'}`",
        f"- Write Tools: `{_enabled_label(not session.no_write)}`",
        f"- Skills: `{session.skills or 'auto'}`",
        f"- Created: `{_format_datetime(session.created_at)}`",
        f"- Updated: `{_format_datetime(session.updated_at)}`",
        f"- Status: `{status}`",
        f"- Reason: `{reason or '-'}`",
        f"- Run Path: `{run_path}`",
    ]

    if checkpoint is None:
        lines.extend(["", "Latest Checkpoint:", "No checkpoint recorded."])
        return "\n".join(lines)

    modified_files = ", ".join(f"`{path}`" for path in checkpoint.modified_files) or "None"
    lines.extend(
        [
            "",
            "Latest Checkpoint:",
            f"- Step: `{checkpoint.step}`",
            f"- Tool Calls: `{checkpoint.tool_calls}`",
            f"- Modified Files: {modified_files}",
            f"- Inspected Files: `{len(checkpoint.run_state.inspected_files)}`",
            f"- Recent Observations: `{len(checkpoint.recent_observations)}`",
            f"- Stop Reason: `{checkpoint.reason or '-'}`",
            "- Verification:",
            f"  - Status: `{checkpoint.run_state.verification.status}`",
        ]
    )
    verification = checkpoint.run_state.verification
    if verification.command:
        lines.append(f"  - Command: `{verification.command}`")
    if verification.returncode is not None:
        lines.append(f"  - Return Code: `{verification.returncode}`")
    if verification.reason:
        lines.append(f"  - Reason: {verification.reason}")

    if observation_limit == 0:
        return "\n".join(lines)

    observations = checkpoint.recent_observations[-observation_limit:]
    lines.extend(["", f"Recent Observations (last {len(observations)}):"])
    if not observations:
        lines.append("- None")
        return "\n".join(lines)

    for observation in observations:
        preview = _single_line(
            observation.summary or observation.output_preview or observation.content,
            140,
        )
        marker = " important" if observation.is_important else ""
        lines.append(
            f"- `{observation.tool_name}` `{observation.tool_call_id}`{marker}: {preview}"
        )
        if observation.artifact_path:
            lines.append(f"  Artifact: `{observation.artifact_path}`")
    return "\n".join(lines)


def format_conversation_sessions(
    workspace: Path | str,
    *,
    limit: int = 10,
    session_store: ReplSessionStore | None = None,
) -> str:
    """Return the recent conversation Sessions for one Workspace."""

    if limit < 1:
        raise ValueError("Session limit must be at least 1.")
    store = session_store or ReplSessionStore()
    sessions = store.list(workspace, limit=limit)
    if not sessions:
        return "Sessions\nNo sessions found for this workspace."
    lines = [
        "Sessions",
        "",
        "| Session ID | Name | Updated | Last User Task |",
        "|---|---|---|---|",
    ]
    for session in sessions:
        last_user_task = next(
            (
                turn.content
                for turn in reversed(session.dialogue)
                if turn.role == "user"
            ),
            "-",
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{session.session_id}`",
                    _table_cell(session.name or "-"),
                    _table_cell(_format_timestamp(session.updated_at)),
                    _table_cell(_single_line(last_user_task, 72)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_project_memory(
    workspace: Path | str,
    *,
    repository_memory: RepositoryMemoryStore | None = None,
) -> str:
    """Return the bounded Repository Memory index and workflow counts."""

    resolved_workspace = Path(workspace).resolve()
    store = repository_memory or RepositoryMemoryStore(resolved_workspace)
    index = store.index_store.ensure().content.rstrip()
    return "\n".join(
        [
            "Repository Memory",
            f"Workspace: {resolved_workspace}",
            f"Repository ID: {store.repository_id}",
            f"Path: {store.memory_dir}",
            f"Pending reviews: {len(store.workflow_store.pending_reviews())}",
            "Pending candidates: "
            f"{len(store.workflow_store.list_candidates(status='pending'))}",
            "",
            index or "No durable memory Topics are registered.",
        ]
    )


def _load_latest_checkpoint(store: RunStore, run_id: str) -> RunCheckpoint | None:
    return CheckpointStore(store.path_for(run_id) / "checkpoints").load_latest()


def _session_status(
    session: RunSession,
    checkpoint: RunCheckpoint | None,
) -> tuple[str, str | None]:
    if checkpoint is not None:
        return checkpoint.status, checkpoint.reason
    return session.status, None


def _format_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _format_timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value).isoformat(timespec="seconds")
    except ValueError:
        return value


def _enabled_label(enabled: bool) -> str:
    return "enabled" if enabled else "disabled"


def _single_line(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    if limit <= 3:
        return compact[:limit]
    return compact[: limit - 3] + "..."


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|")
