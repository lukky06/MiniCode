"""Cross-rollout Memory V3 consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any

from pydantic import BaseModel

from minicode_harness.models import ModelClient, ModelRequest

from .store import RepositoryMemoryStore
from .types import MemoryPipelineState, Stage1Record


PHASE2_COOLDOWN = timedelta(hours=6)
MAX_MEMORY_SUMMARY_BYTES = 4 * 1024


class Phase2Output(BaseModel):
    model_config = {"extra": "forbid"}

    memory_md: str
    memory_summary_md: str


@dataclass(frozen=True)
class Phase2Result:
    status: str
    reason: str
    mode: str | None = None
    input_records: int = 0


class Phase2Consolidator:
    """Merge pending Stage-1 outputs into the two durable Memory V3 files."""

    def __init__(
        self,
        store: RepositoryMemoryStore,
        model_client: ModelClient,
    ) -> None:
        self.store = store
        self.model_client = model_client

    def consolidate(
        self,
        *,
        explicit: bool = False,
        initialize: bool = False,
        now: datetime | None = None,
    ) -> Phase2Result:
        current_time = now or datetime.now(timezone.utc)
        state = self.store.load_state()
        if state.latest_stage1_seq <= state.last_phase2_input_seq:
            return Phase2Result(
                status="noop" if explicit else "skipped",
                reason="clean",
            )

        mode = (
            "init"
            if initialize or not self.store.summary_path.is_file()
            else "incremental"
        )
        if (
            mode == "incremental"
            and not explicit
            and not _cooldown_elapsed(state, current_time)
        ):
            return Phase2Result(status="skipped", reason="cooldown", mode=mode)

        pending = self.store.pending_stage1_records()
        if not pending:
            raise ValueError("Memory state is dirty but no pending Stage-1 records exist.")

        raw_memories = _render_raw_memories(pending)
        self.store.write_raw_memories(raw_memories)
        rollout_summaries = [
            self._rollout_summary_input(record)
            for record in pending
        ]

        response = self.model_client.call_request(
            ModelRequest(
                system=_phase2_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "raw_memories_md": raw_memories,
                                "rollout_summaries": rollout_summaries,
                                "existing_memory_md": (
                                    "" if initialize else self.store.read_memory()
                                ),
                                "existing_memory_summary_md": (
                                    ""
                                    if initialize
                                    else self.store.read_memory_summary()
                                ),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                ],
                tools=[],
                metadata={
                    "purpose": "memory_phase2_v3",
                    "repository_id": self.store.repository_id,
                    "mode": mode,
                    "temperature": 0.0,
                },
            )
        ).enforce_turn_contract()
        if response.tool_calls:
            raise ValueError("Memory Phase 2 returned tool calls.")
        if str(response.stop_reason or "").lower() in {
            "length",
            "max_tokens",
            "max_output_tokens",
        }:
            raise ValueError("Memory Phase 2 response was truncated.")
        if not response.final_text:
            raise ValueError(
                response.invalid_reason or "Memory Phase 2 returned no JSON."
            )

        output = Phase2Output.model_validate(_parse_json_object(response.final_text))
        memory_md = output.memory_md
        summary_md = output.memory_summary_md
        _validate_phase2_output(memory_md, summary_md)

        self.store.commit_consolidation(
            rollout_summaries=[
                (
                    str(item["run_id"]),
                    str(item["rollout_slug"]),
                    str(item["content"]),
                )
                for item in rollout_summaries
            ],
            memory_md=memory_md,
            memory_summary_md=summary_md,
        )
        self.store.save_state(
            state.model_copy(
                update={
                    "last_phase2_input_seq": state.latest_stage1_seq,
                    "last_phase2_success_at": current_time.astimezone(
                        timezone.utc
                    ).isoformat(),
                }
            )
        )
        return Phase2Result(
            status="completed",
            reason="dirty",
            mode=mode,
            input_records=len(pending),
        )

    def _rollout_summary_input(
        self,
        record: Stage1Record,
    ) -> dict[str, Any]:
        content = _render_rollout_summary(record)
        return {
            "run_id": record.run_id,
            "seq": record.seq,
            "rollout_slug": record.rollout_slug,
            "path": self.store.rollout_summary_filename(
                run_id=record.run_id,
                rollout_slug=record.rollout_slug,
            ),
            "content": content,
        }


def _cooldown_elapsed(
    state: MemoryPipelineState,
    now: datetime,
) -> bool:
    if state.last_phase2_success_at is None:
        return True
    last_success = datetime.fromisoformat(state.last_phase2_success_at)
    if last_success.tzinfo is None:
        last_success = last_success.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc) - last_success.astimezone(timezone.utc) >= PHASE2_COOLDOWN


def _render_raw_memories(records: list[Stage1Record]) -> str:
    parts = ["# Pending Raw Memories"]
    for record in records:
        parts.extend(
            [
                "",
                f"## seq {record.seq} · {record.run_id} · {record.rollout_slug}",
                "",
                record.raw_memory.strip(),
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def _render_rollout_summary(record: Stage1Record) -> str:
    return (
        "# Rollout Summary\n\n"
        f"- Run: {record.run_id}\n"
        f"- Sequence: {record.seq}\n"
        f"- Slug: {record.rollout_slug}\n\n"
        f"{record.rollout_summary.strip()}\n"
    )


def _validate_phase2_output(memory_md: str, summary_md: str) -> None:
    if not memory_md.strip():
        raise ValueError("Memory Phase 2 returned an empty MEMORY.md.")
    if not summary_md.strip():
        raise ValueError("Memory Phase 2 returned an empty memory_summary.md.")
    lines = summary_md.splitlines()
    if not lines or lines[0] != "v1":
        raise ValueError("memory_summary.md must start with the exact line 'v1'.")
    size = len(summary_md.encode("utf-8"))
    if size > MAX_MEMORY_SUMMARY_BYTES:
        raise ValueError(
            f"memory_summary.md exceeds {MAX_MEMORY_SUMMARY_BYTES} bytes: {size}."
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fence = chr(96) * 3
    if candidate.startswith(fence):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == fence:
            candidate = "\n".join(lines[1:-1]).strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].lstrip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Memory Phase 2 response must be a JSON object.")
    return value


def _phase2_system_prompt() -> str:
    return """You consolidate durable repository memory across completed rollouts.

Return exactly one JSON object with string fields:
{"memory_md":"<complete MEMORY.md>","memory_summary_md":"<complete memory_summary.md>"}

Inputs contain only the current pending delta plus the existing durable memory files.
Produce complete replacement contents for both durable files.

MEMORY.md should be task-oriented and useful for future coding work. Organize it by
the actual reusable knowledge present in the inputs. Merge duplicates, preserve
confirmed user corrections and project decisions, keep practical failure shields,
and drop stale or low-value repetition. Do not invent facts or fixed topic buckets.

memory_summary.md is the small always-on entry point. Its first line must be exactly:
v1
Keep it concise enough to stay under 4 KiB. Summarize the durable knowledge and
point the future agent toward useful sections or rollout summaries when deeper
history is needed. Do not copy the full handbook into the summary."""
