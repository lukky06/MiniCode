"""Single-rollout Memory V3 extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from pydantic import BaseModel

from minicode_harness.models import ModelClient, ModelRequest
from minicode_harness.state import CheckpointStore, RunStore
from minicode_harness.trace import TraceWriter

from .store import RepositoryMemoryStore
from .types import Stage1Record


TERMINAL_RUN_STATUSES = {"completed", "stopped", "cancelled"}


class Phase1Output(BaseModel):
    model_config = {"extra": "forbid"}

    raw_memory: str = ""
    rollout_summary: str = ""
    rollout_slug: str = ""


@dataclass(frozen=True)
class Phase1BatchResult:
    processed_run_ids: list[str]
    failed_run_ids: list[str]


class Phase1Extractor:
    """Extract reusable memory from one terminal canonical rollout."""

    def __init__(
        self,
        store: RepositoryMemoryStore,
        run_store: RunStore,
        model_client: ModelClient,
    ) -> None:
        self.store = store
        self.run_store = run_store
        self.model_client = model_client

    def extract(self, run_id: str) -> Stage1Record:
        session = self.run_store.load_session(run_id)
        if session.status not in TERMINAL_RUN_STATUSES:
            raise ValueError(f"Run is not terminal for Memory Phase 1: {run_id}")
        if self.store.load_stage1(run_id) is not None:
            raise FileExistsError(f"Stage-1 terminal record already exists: {run_id}")

        checkpoint_store = CheckpointStore(
            self.run_store.path_for(run_id) / "checkpoints"
        )
        checkpoint = checkpoint_store.load_latest()
        if checkpoint is None:
            raise ValueError(f"Terminal Run has no checkpoint: {run_id}")
        rollout = filter_rollout_messages(
            checkpoint_store.load_history(checkpoint)
        )
        payload = {
            "run_id": run_id,
            "repository_id": self.store.repository_id,
            "workspace": session.workspace,
            "rollout": rollout,
        }
        response = self.model_client.call_request(
            ModelRequest(
                system=_phase1_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                ],
                tools=[],
                metadata={
                    "purpose": "memory_phase1_v3",
                    "repository_id": self.store.repository_id,
                    "run_id": run_id,
                    "temperature": 0.0,
                },
            )
        ).enforce_turn_contract()
        if response.tool_calls:
            raise ValueError("Memory Phase 1 returned tool calls.")
        if str(response.stop_reason or "").lower() in {
            "length",
            "max_tokens",
            "max_output_tokens",
        }:
            raise ValueError("Memory Phase 1 response was truncated.")
        if not response.final_text:
            raise ValueError(
                response.invalid_reason or "Memory Phase 1 returned no JSON."
            )

        output = Phase1Output.model_validate(_parse_json_object(response.final_text))
        values = (
            output.raw_memory.strip(),
            output.rollout_summary.strip(),
            output.rollout_slug.strip(),
        )
        if not any(values):
            return self.store.write_stage1_no_output(run_id)
        if not all(values):
            raise ValueError(
                "Memory Phase 1 must return raw_memory, rollout_summary, and rollout_slug together."
            )
        return self.store.write_stage1_memory(
            run_id=run_id,
            raw_memory=values[0],
            rollout_summary=values[1],
            rollout_slug=values[2],
        )


def filter_rollout_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove runtime-only metadata while preserving canonical message semantics."""

    return [
        {
            key: value
            for key, value in message.items()
            if not str(key).startswith("_minicode_")
        }
        for message in messages
    ]


def run_pending_phase1(
    *,
    store: RepositoryMemoryStore,
    run_store: RunStore,
    model_client: ModelClient,
    current_run_id: str,
    trace_writer: TraceWriter | None = None,
) -> Phase1BatchResult:
    """Sequentially extract earlier unprocessed terminal top-level Runs."""

    extractor = Phase1Extractor(store, run_store, model_client)
    processed: list[str] = []
    failed: list[str] = []
    for run_id in run_store.list_run_ids(workspace=store.workspace):
        if run_id == current_run_id or store.load_stage1(run_id) is not None:
            continue
        session = run_store.load_session(run_id)
        if (
            not session.repository_memory_enabled
            or session.status not in TERMINAL_RUN_STATUSES
        ):
            continue
        _event(
            store,
            trace_writer,
            "memory_phase1_started",
            source_run_id=run_id,
        )
        try:
            record = extractor.extract(run_id)
        except Exception as exc:
            failed.append(run_id)
            _event(
                store,
                trace_writer,
                "memory_phase1_failed",
                source_run_id=run_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue
        processed.append(run_id)
        _event(
            store,
            trace_writer,
            "memory_phase1_completed",
            source_run_id=run_id,
            status=record.status,
            seq=record.seq,
        )
    return Phase1BatchResult(
        processed_run_ids=processed,
        failed_run_ids=failed,
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
        raise ValueError("Memory Phase 1 response must be a JSON object.")
    return value


def _phase1_system_prompt() -> str:
    return """You extract durable repository memory from one completed coding-agent rollout.

Return exactly one JSON object with string fields:
{"raw_memory":"...","rollout_summary":"...","rollout_slug":"..."}

Keep memory only when the rollout contains reusable signal for future repository work:
- explicit durable user preferences or corrections;
- reusable workflows confirmed by the rollout;
- confirmed project conventions or historical decisions;
- failure modes with root cause and an effective fix;
- environment, toolchain, or project behavior with lasting value;
- information likely to reduce future tool calls or reasoning.

Return all three fields as empty strings when there is no durable signal.
Skip ordinary conversation, one-time progress/state, generic technical knowledge,
facts easily rediscovered from the current repository, unsupported guesses, and
low-value tool output.

raw_memory should preserve the reusable facts and practical failure shields.
rollout_summary should summarize the completed rollout for deeper historical reading.
rollout_slug should be a short filesystem-friendly task/failure-family slug.
Use only evidence present in the supplied rollout. Do not invent or re-verify facts."""


def _event(
    store: RepositoryMemoryStore,
    trace_writer: TraceWriter | None,
    event_type: str,
    **payload: Any,
) -> None:
    if trace_writer is not None:
        trace_writer.write_event(event_type, **payload)
    try:
        store.event_store.append(event_type, **payload)
    except OSError:
        pass
