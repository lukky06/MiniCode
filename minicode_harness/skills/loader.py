"""Discover and load built-in skill documents."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


BUILT_IN_SKILL_NAMES = (
    "code-debug",
    "unit-test",
    "repo-explain",
    "test-generation",
    "web-controller-test",
    "refactor",
    "review",
)
DEFAULT_SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills"


class SkillSummary(BaseModel):
    """Compact catalog entry exposed before a skill is loaded."""

    name: str
    description: str
    path: str


class Skill(BaseModel):
    """A fully loaded skill document."""

    name: str
    content: str
    path: str


class SkillLoader:
    """Discover ``<skill>/SKILL.md`` files and load them on demand."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_SKILL_ROOT

    def list_available(self) -> list[str]:
        """Return available skill names."""

        if not self.root.exists():
            return []
        return [
            name
            for name in BUILT_IN_SKILL_NAMES
            if (self.root / name / "SKILL.md").is_file()
        ]

    def resolve_names(self, names: Iterable[str] | None = None) -> list[str]:
        """Return available names, optionally restricted by an explicit list."""

        available = self.list_available()
        if names is None:
            return available
        available_set = set(available)
        resolved: list[str] = []
        seen: set[str] = set()
        for name in names:
            normalized = str(name).strip()
            if normalized and normalized in available_set and normalized not in seen:
                resolved.append(normalized)
                seen.add(normalized)
        return resolved

    def list_summaries(self, names: Iterable[str] | None = None) -> list[SkillSummary]:
        """Return compact catalog entries without loading full skill documents."""

        return [self.describe(name) for name in self.resolve_names(names)]

    def describe(self, name: str) -> SkillSummary:
        """Read only the short ``When to use`` paragraph for one skill."""

        path = self._skill_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Skill not found: {name}")
        return SkillSummary(
            name=name,
            description=_read_when_to_use(path),
            path=str(path),
        )

    def load(self, name: str) -> Skill:
        """Load one complete skill document by exact name."""

        path = self._skill_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Skill not found: {name}")
        return Skill(
            name=name,
            content=path.read_text(encoding="utf-8"),
            path=str(path),
        )

    def _skill_path(self, name: str) -> Path:
        return self.root / name / "SKILL.md"


def _read_when_to_use(path: Path) -> str:
    """Extract one bounded catalog description without reading the full file."""

    fallback: list[str] = []
    description: list[str] = []
    in_when_to_use = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "## When to use":
                in_when_to_use = True
                continue
            if in_when_to_use:
                if line.startswith("## "):
                    break
                if not line:
                    if description:
                        break
                    continue
                description.append(line)
                continue
            if line and not line.startswith("#") and len(fallback) < 2:
                fallback.append(line)
    text = " ".join(description or fallback).strip()
    return text or f"Load the {path.parent.name} skill for its full instructions."
