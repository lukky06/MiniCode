"""Workspace safety helpers."""

from .guard import WorkspaceAccessError, WorkspaceGuard
from .profile import (
    WorkspaceProfile,
    extract_source_paths,
    is_code_path,
    preferred_verification_command_for_paths,
    scan_workspace_profile,
    workspace_profile_may_change,
)

__all__ = [
    "WorkspaceAccessError",
    "WorkspaceGuard",
    "WorkspaceProfile",
    "extract_source_paths",
    "is_code_path",
    "preferred_verification_command_for_paths",
    "scan_workspace_profile",
    "workspace_profile_may_change",
]
