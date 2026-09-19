"""Tool risk classification."""

from __future__ import annotations

from enum import StrEnum


class RiskLevel(StrEnum):
    """MVP risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
