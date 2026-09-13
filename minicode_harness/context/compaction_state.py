"""Persistent cursors for deriving model-visible history from canonical messages."""

from __future__ import annotations

from pydantic import BaseModel


class ExecutionCompactionState(BaseModel):
    """Stable boundary for deterministic Tool Group receipts."""

    boundary_group_id: str
    policy_version: int = 1
    source_digest: str


class SemanticCompactionState(BaseModel):
    """Persisted semantic summary and the first exact group kept after it."""

    summary: str
    first_kept_group_id: str
    source_digest: str
    policy_version: int = 1


class SessionCompactionState(BaseModel):
    """All model-projection state for one canonical conversation history."""

    execution: ExecutionCompactionState | None = None
    semantic: SemanticCompactionState | None = None
    semantic_attempt_group_id: str | None = None
    semantic_attempt_source_digest: str | None = None
