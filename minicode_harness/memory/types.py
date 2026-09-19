"""Structured contracts for Repository Memory V3."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryPipelineState(BaseModel):
    schema_version: Literal[3] = 3
    latest_stage1_seq: int = Field(default=0, ge=0)
    last_phase2_input_seq: int = Field(default=0, ge=0)
    last_phase2_success_at: str | None = None


class Stage1Record(BaseModel):
    run_id: str = Field(min_length=1)
    status: Literal["memory", "no_output"]
    updated_at: str = Field(default_factory=utc_now)
    seq: int | None = Field(default=None, ge=1)
    rollout_slug: str = ""
    raw_memory: str = ""
    rollout_summary: str = ""

    @model_validator(mode="after")
    def validate_terminal_record(self) -> "Stage1Record":
        if self.status == "no_output":
            if self.seq is not None:
                raise ValueError("no_output Stage-1 records must not have a sequence.")
            return self
        if self.seq is None:
            raise ValueError("memory Stage-1 records require a sequence.")
        if not (
            self.raw_memory.strip()
            and self.rollout_summary.strip()
            and self.rollout_slug.strip()
        ):
            raise ValueError(
                "memory Stage-1 records require raw memory, summary, and slug."
            )
        return self
