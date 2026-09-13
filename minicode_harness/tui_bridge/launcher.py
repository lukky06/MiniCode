"""Launch the packaged TypeScript/pi-tui frontend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

_MIN_NODE_VERSION = (22, 19, 0)
_NODE_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def launch_tui(
    *,
    workspace: Path,
    provider: str,
    model: str | None,
    write_enabled: bool,
    approval_policy: str,
    permission_mode: str,
    sandbox_mode: str,
    sandbox_image: str | None,
    collaboration_mode: str,
    skills: Sequence[str] | None,
    skills_enabled: bool,
    prompt_cache_enabled: bool,
    repository_memory_enabled: bool,
    subagents_enabled: bool,
    mcp_config: Path | None,
    session_mode: str,
    session_id: str | None,
    initial_task: str | None,
    no_color: bool,
) -> None:
    """Validate Node and run the packaged TUI in the selected workspace."""

    node = _find_compatible_node()
    entrypoint = _runtime_entrypoint()
    if not entrypoint.is_file():
        raise RuntimeError(
            "MiniCode TUI runtime is missing. Reinstall minicode-harness "
            "or rebuild the packaged TUI runtime."
        )

    env = _build_environment(
        workspace=workspace,
        provider=provider,
        model=model,
        write_enabled=write_enabled,
        approval_policy=approval_policy,
        permission_mode=permission_mode,
        sandbox_mode=sandbox_mode,
        sandbox_image=sandbox_image,
        collaboration_mode=collaboration_mode,
        skills=skills,
        skills_enabled=skills_enabled,
        prompt_cache_enabled=prompt_cache_enabled,
        repository_memory_enabled=repository_memory_enabled,
        subagents_enabled=subagents_enabled,
        mcp_config=mcp_config,
        session_mode=session_mode,
        session_id=session_id,
        initial_task=initial_task,
        no_color=no_color,
    )
    completed = subprocess.run(
        [node, str(entrypoint)],
        cwd=workspace,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MiniCode TUI exited with status {completed.returncode}."
        )


def _runtime_entrypoint() -> Path:
    return Path(__file__).resolve().parents[1] / "tui_runtime" / "main.mjs"


def _find_compatible_node() -> str:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError(_node_requirement_message())

    try:
        completed = subprocess.run(
            [node, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(_node_requirement_message()) from exc

    version = _parse_node_version(completed.stdout.strip())
    if completed.returncode != 0 or version is None or version < _MIN_NODE_VERSION:
        raise RuntimeError(_node_requirement_message())
    return node


def _parse_node_version(value: str) -> tuple[int, int, int] | None:
    match = _NODE_VERSION_RE.match(value)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _node_requirement_message() -> str:
    return (
        "MiniCode interactive TUI requires Node.js >= 22.19.0. "
        'Install or upgrade Node.js, or use `minicode exec "<task>"` '
        "for non-interactive execution."
    )


def _build_environment(
    *,
    workspace: Path,
    provider: str,
    model: str | None,
    write_enabled: bool,
    approval_policy: str,
    permission_mode: str,
    sandbox_mode: str,
    sandbox_image: str | None,
    collaboration_mode: str,
    skills: Sequence[str] | None,
    skills_enabled: bool,
    prompt_cache_enabled: bool,
    repository_memory_enabled: bool,
    subagents_enabled: bool,
    mcp_config: Path | None,
    session_mode: str,
    session_id: str | None,
    initial_task: str | None,
    no_color: bool,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    values = {
        "MINICODE_PYTHON": sys.executable,
        "MINICODE_WORKSPACE": str(workspace),
        "MINICODE_PROVIDER": provider,
        "MINICODE_MODEL": model,
        "MINICODE_TUI_APPROVAL_POLICY": approval_policy,
        "MINICODE_TUI_PERMISSION_MODE": permission_mode,
        "MINICODE_TUI_SANDBOX_MODE": sandbox_mode,
        "MINICODE_TUI_SANDBOX_IMAGE": sandbox_image,
        "MINICODE_TUI_COLLABORATION_MODE": collaboration_mode,
        "MINICODE_TUI_SKILLS": json.dumps(list(skills or [])),
        "MINICODE_TUI_MCP_CONFIG": str(mcp_config) if mcp_config else None,
        "MINICODE_TUI_SESSION_MODE": session_mode,
        "MINICODE_TUI_SESSION_ID": session_id,
        "MINICODE_TUI_INITIAL_TASK": initial_task,
    }
    for key, value in values.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    flags = {
        "MINICODE_TUI_NO_WRITE": not write_enabled,
        "MINICODE_TUI_NO_SKILLS": not skills_enabled,
        "MINICODE_TUI_NO_PROMPT_CACHE": not prompt_cache_enabled,
        "MINICODE_TUI_NO_REPOSITORY_MEMORY": not repository_memory_enabled,
        "MINICODE_TUI_NO_SUBAGENTS": not subagents_enabled,
        "NO_COLOR": no_color,
    }
    for key, enabled in flags.items():
        if enabled:
            env[key] = "1"
        else:
            env.pop(key, None)
    return env
