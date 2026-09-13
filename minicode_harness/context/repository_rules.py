"""Deterministic hierarchical repository-rule loading.

Repository rules are explicit instructions, not Auto Memory. The loader keeps a
small general-to-specific view over user-level and workspace ``AGENTS.md``
files without semantic retrieval or another model call.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable


MAX_REPOSITORY_RULE_BYTES = 32 * 1024
RULE_FILE_NAME = "AGENTS.md"


@dataclass(frozen=True)
class RepositoryRuleDocument:
    """One loaded rule document and its model-visible source label."""

    source: str
    content: str


@dataclass(frozen=True)
class RepositoryRulesSnapshot:
    """Bounded rule content used for one model request."""

    content: str = ""
    sources: tuple[str, ...] = ()
    skipped_sources: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def size_bytes(self) -> int:
        return len(self.content.encode("utf-8"))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class RepositoryRuleLoader:
    """Load user, root, and touched-path rules in deterministic precedence order."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        user_rules_path: Path | str | None = None,
        max_bytes: int = MAX_REPOSITORY_RULE_BYTES,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.user_rules_path = (
            Path(user_rules_path).expanduser().resolve()
            if user_rules_path is not None
            else (Path.home() / ".minicode" / RULE_FILE_NAME).resolve()
        )
        self.max_bytes = max(1024, int(max_bytes))

    def load(self, *, paths: Iterable[str] = ()) -> RepositoryRulesSnapshot:
        documents: list[RepositoryRuleDocument] = []
        skipped: list[str] = []

        self._append_external_document(
            documents,
            skipped,
            path=self.user_rules_path,
            source="user:~/.minicode/AGENTS.md",
        )

        candidates = [self.workspace / RULE_FILE_NAME]
        nested = self._nested_rule_paths(paths)
        candidates.extend(nested)
        for candidate in candidates:
            source = self._workspace_source(candidate)
            self._append_workspace_document(
                documents,
                skipped,
                path=candidate,
                source=source,
            )

        if not documents:
            return RepositoryRulesSnapshot(skipped_sources=tuple(skipped))

        header = (
            "Repository Rules (general to specific; later scoped rules override earlier rules; "
            "current user instructions and runtime safety policy take precedence):"
        )
        sections = [header]
        for document in documents:
            sections.append(f"[{document.source}]\n{document.content.strip()}")
        rendered = "\n\n".join(section for section in sections if section.strip())
        bounded, truncated = _clip_utf8(rendered, self.max_bytes)
        return RepositoryRulesSnapshot(
            content=bounded,
            sources=tuple(document.source for document in documents),
            skipped_sources=tuple(skipped),
            truncated=truncated,
        )

    def _nested_rule_paths(self, paths: Iterable[str]) -> list[Path]:
        candidates: set[Path] = set()
        for value in paths:
            raw = str(value).strip()
            if not raw:
                continue
            candidate = Path(raw)
            absolute = candidate.resolve() if candidate.is_absolute() else (self.workspace / candidate).resolve()
            if absolute != self.workspace and not absolute.is_relative_to(self.workspace):
                continue
            directory = absolute if absolute.is_dir() else absolute.parent
            while directory != self.workspace and directory.is_relative_to(self.workspace):
                candidates.add(directory / RULE_FILE_NAME)
                directory = directory.parent
        return sorted(
            candidates,
            key=lambda path: (
                len(path.relative_to(self.workspace).parts),
                path.relative_to(self.workspace).as_posix(),
            ),
        )

    def _append_workspace_document(
        self,
        documents: list[RepositoryRuleDocument],
        skipped: list[str],
        *,
        path: Path,
        source: str,
    ) -> None:
        if not path.is_file():
            return
        try:
            resolved = path.resolve()
            if resolved != self.workspace and not resolved.is_relative_to(self.workspace):
                skipped.append(source)
                return
            content = resolved.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            skipped.append(source)
            return
        if content:
            documents.append(RepositoryRuleDocument(source=source, content=content))

    @staticmethod
    def _append_external_document(
        documents: list[RepositoryRuleDocument],
        skipped: list[str],
        *,
        path: Path,
        source: str,
    ) -> None:
        if not path.is_file():
            return
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            skipped.append(source)
            return
        if content:
            documents.append(RepositoryRuleDocument(source=source, content=content))

    def _workspace_source(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.workspace).as_posix()
        except ValueError:
            return path.name
        scope = Path(relative).parent.as_posix()
        return f"repository:{relative}; scope={scope}"


def _clip_utf8(value: str, max_bytes: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    marker = "\n\n[Repository Rules truncated at configured byte limit]"
    marker_bytes = marker.encode("utf-8")
    body_limit = max(0, max_bytes - len(marker_bytes))
    clipped = encoded[:body_limit].decode("utf-8", errors="ignore").rstrip()
    return clipped + marker, True
