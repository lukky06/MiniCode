"""Presentation helpers for slash-command results in the TypeScript TUI."""

from __future__ import annotations


_TITLES = {
    "status": "Status",
    "sessions": "Sessions",
    "runs": "Runs",
    "history": "Runs",
    "context": "Context",
    "compact": "Compact",
    "memory": "Memory",
    "trace": "Trace",
    "recover": "Recover",
    "resume": "Recover",
    "report": "Report",
    "diff": "Diff",
    "model": "Model",
    "permissions": "Permissions",
    "rename": "Rename",
    "fork": "Fork",
    "review": "Review",
    "help": "Help",
    "clear": "Clear",
}


def panel_title(command_name: str) -> str:
    """Return a display-only title without owning command semantics."""

    return _TITLES.get(command_name, command_name.replace("_", " ").title())
