"""Workspace safety helpers."""

from .guard import WorkspaceAccessError, WorkspaceGuard
from .profile import (
    WorkspaceProfile,
    extract_source_paths,
    is_code_path,
    scan_workspace_profile,
)

__all__ = [
    "WorkspaceAccessError",
    "WorkspaceGuard",
    "WorkspaceProfile",
    "extract_source_paths",
    "is_code_path",
    "scan_workspace_profile",
]
