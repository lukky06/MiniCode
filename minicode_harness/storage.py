"""Shared Harness storage paths and workspace identities.

This module is neutral infrastructure used by context, memory, and state
modules. It keeps shared storage paths independent from repository memory.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessDataStore:
    """Neutral carrier for Harness-owned data paths.

    This is not a memory implementation. It exists so tests and adapters can
    bind prompt caches, sessions, and run artifacts to an isolated data root.
    """

    data_dir: Path

    def __init__(self, data_dir: Path | str | None = None) -> None:
        object.__setattr__(
            self,
            "data_dir",
            Path(data_dir) if data_dir is not None else default_data_dir(),
        )


def default_data_dir() -> Path:
    """Return the platform-specific Harness data directory."""

    configured = os.environ.get("MINICODE_HOME")
    if configured:
        return Path(configured)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "MiniCodeHarness"
        return Path.home() / "AppData" / "Local" / "MiniCodeHarness"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "minicode-harness"
    return Path.home() / ".local" / "share" / "minicode-harness"


def workspace_hash(workspace: Path | str) -> str:
    """Return the stable workspace key used by Harness-owned storage."""

    resolved = os.path.normcase(str(Path(workspace).resolve()))
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
