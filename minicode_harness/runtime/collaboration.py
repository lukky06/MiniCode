"""Collaboration modes that shape one Run without adding another agent loop."""

from __future__ import annotations

from enum import StrEnum


class CollaborationMode(StrEnum):
    """How the model should collaborate for one Run."""

    DEFAULT = "default"
    PLAN = "plan"


DEFAULT_COLLABORATION_MODE = CollaborationMode.DEFAULT
