"""One-time V2 Topic to Memory V3 migration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from minicode_harness.models import ModelClient

from .consolidation import Phase2Consolidator, Phase2Result
from .store import RepositoryMemoryStore


MIGRATION_RUN_ID = "migration_v2"
MIGRATION_SLUG = "v2-memory-migration"


@dataclass(frozen=True)
class MemoryMigrationResult:
    status: str
    migrated_entries: int = 0
    phase2: Phase2Result | None = None


def migrate_v2_topics(
    store: RepositoryMemoryStore,
    model_client: ModelClient,
) -> MemoryMigrationResult:
    """Convert active V2 Topic entries into one synthetic Stage-1 record."""

    if store.summary_path.is_file():
        return MemoryMigrationResult(status="skipped")

    entries = _read_active_v2_entries(store.memory_dir / "topics")
    if not entries:
        return MemoryMigrationResult(status="skipped")

    original_state = (
        store.state_path.read_bytes()
        if store.state_path.is_file()
        else None
    )
    record = store.load_stage1(MIGRATION_RUN_ID)
    try:
        if record is None:
            record = store.write_stage1_memory(
                run_id=MIGRATION_RUN_ID,
                raw_memory=_render_raw_memory(entries),
                rollout_summary=(
                    f"Migrated {len(entries)} active V2 repository memory entries."
                ),
                rollout_slug=MIGRATION_SLUG,
            )
        elif not store.state_path.is_file():
            store.save_state(
                store.load_state().model_copy(
                    update={"latest_stage1_seq": record.seq or 0}
                )
            )

        phase2 = Phase2Consolidator(store, model_client).consolidate(
            explicit=True,
            initialize=True,
        )
    except Exception:
        if original_state is None:
            store.state_path.unlink(missing_ok=True)
        else:
            store.state_path.write_bytes(original_state)
        raise

    return MemoryMigrationResult(
        status="completed",
        migrated_entries=len(entries),
        phase2=phase2,
    )


def _read_active_v2_entries(topics_dir: Path) -> list[tuple[str, str]]:
    if not topics_dir.is_dir():
        return []
    entries: list[tuple[str, str]] = []
    for path in sorted(topics_dir.glob("*.md")):
        payload = _parse_front_matter(path.read_text(encoding="utf-8"))
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            continue
        for item in raw_entries:
            if not isinstance(item, dict) or item.get("status") != "active":
                continue
            summary = " ".join(str(item.get("summary") or "").split())
            if summary:
                entries.append((path.stem, summary))
    return entries


def _parse_front_matter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}
    try:
        closing = lines[1:].index("---") + 1
    except ValueError:
        return {}
    payload = yaml.safe_load("\n".join(lines[1:closing])) or {}
    return payload if isinstance(payload, dict) else {}


def _render_raw_memory(entries: list[tuple[str, str]]) -> str:
    lines = [
        "Legacy V2 active repository memory entries migrated mechanically:",
        *[f"- [{topic}] {summary}" for topic, summary in entries],
    ]
    return "\n".join(lines)
