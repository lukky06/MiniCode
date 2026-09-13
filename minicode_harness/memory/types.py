"""Structured contracts for repository-scoped memory."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MemoryCandidateType = Literal[
    "user_instruction",
    "user_correction",
    "procedure",
    "pitfall",
    "decision",
    "environment",
]
MemoryTopicName = Literal[
    "instructions",
    "build-and-test",
    "debugging",
    "decisions",
    "environment",
]
MemoryEntryStatus = Literal["active", "inactive"]

TOPIC_NAMES: tuple[MemoryTopicName, ...] = (
    "instructions",
    "build-and-test",
    "debugging",
    "decisions",
    "environment",
)

_MAX_SUMMARY_CHARS = 600
_SENSITIVE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|secret|password|passwd)\s*[:=]\s*"
        r"[^\s,;]{8,}",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contains_sensitive_content(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _SENSITIVE_PATTERNS)


class MemoryTopicEntry(BaseModel):
    entry_id: str = Field(min_length=1, max_length=160)
    type: MemoryCandidateType
    summary: str = Field(min_length=1, max_length=_MAX_SUMMARY_CHARS)
    evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    status: MemoryEntryStatus = "active"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if contains_sensitive_content(normalized):
            raise ValueError("Memory topic entry contains sensitive content.")
        return normalized


class MemoryTopicDocument(BaseModel):
    topic: MemoryTopicName
    entries: list[MemoryTopicEntry] = Field(default_factory=list, max_length=128)
    updated_at: str = Field(default_factory=utc_now)
