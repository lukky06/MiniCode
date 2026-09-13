"""Persistent prompt section cache for stable context prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel

from minicode_harness.storage import default_data_dir, workspace_hash

from .token import estimate_tokens
from .types import ContextSkill


DEFAULT_MAX_PROMPT_CACHE_RECORDS = 64


class PromptCacheEntry(BaseModel):
    """One cached prompt section."""

    key: str
    section_name: str
    content: str
    content_hash: str
    token_estimate: int
    created_at: datetime


@dataclass(frozen=True)
class PromptCacheLookup:
    """Result of a cache lookup or write."""

    entry: PromptCacheEntry
    hit: bool


class PromptSectionCache:
    """Workspace-scoped persistent cache for stable prompt sections."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        provider: str,
        model: str | None,
        data_dir: Path | str | None = None,
        max_records: int = DEFAULT_MAX_PROMPT_CACHE_RECORDS,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.provider = provider.strip().lower() or "unknown"
        self.model = model.strip() if model else "default"
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.max_records = max_records
        self.workspace_hash = workspace_hash(self.workspace)
        self.cache_dir = (
            self.data_dir
            / "prompt-cache"
            / self.workspace_hash
            / _safe_component(self.provider)
            / _safe_component(self.model)
        )
        self._ensure_outside_workspace()

    def system_prefix_key(
        self,
        *,
        builder_version: str,
        section_name: str,
        available_skills: list[ContextSkill],
        long_term_context: str,
        rendered_content_hash: str,
    ) -> str:
        """Return a stable key for the system prefix section.

        The rendered content hash is part of the key so edits to stable system
        rules cannot be hidden by a stale persistent cache entry when the
        builder version was not bumped.
        """

        material = {
            "builder_version": builder_version,
            "provider": self.provider,
            "model": self.model,
            "workspace_hash": self.workspace_hash,
            "section_name": section_name,
            "rendered_content_hash": rendered_content_hash,
            "long_term_context_hash": content_hash(long_term_context),
            "skills": [
                {
                    "name": skill.name,
                    "source": skill.source,
                    "description_hash": content_hash(skill.description),
                }
                for skill in available_skills
            ],
        }
        return content_hash(_stable_json(material))

    def get_or_write(
        self,
        *,
        key: str,
        section_name: str,
        content: str,
    ) -> PromptCacheLookup:
        """Return an existing entry or persist ``content`` under ``key``."""

        cached = self.read(key)
        if cached is not None:
            return PromptCacheLookup(entry=cached, hit=True)

        entry = PromptCacheEntry(
            key=key,
            section_name=section_name,
            content=content,
            content_hash=content_hash(content),
            token_estimate=estimate_tokens(content),
            created_at=datetime.now(timezone.utc),
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._entry_path(key).write_text(
            json.dumps(entry.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        self._evict_old_entries()
        return PromptCacheLookup(entry=entry, hit=False)

    def read(self, key: str) -> PromptCacheEntry | None:
        """Read an existing cache entry without writing."""

        path = self._entry_path(key)
        if not path.is_file():
            return None
        try:
            return PromptCacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _entry_path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Prompt cache key must be a SHA-256 hex digest.")
        return self.cache_dir / f"{key}.json"

    def _evict_old_entries(self) -> None:
        entries = sorted(
            self.cache_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in entries[self.max_records :]:
            path.unlink(missing_ok=True)

    def _ensure_outside_workspace(self) -> None:
        data_path = self.data_dir.resolve()
        if data_path == self.workspace or data_path.is_relative_to(self.workspace):
            raise ValueError("Prompt cache data_dir must not be inside the workspace.")


def content_hash(content: str) -> str:
    """Return a SHA-256 digest for prompt cache material."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return (safe or "default")[:96]
