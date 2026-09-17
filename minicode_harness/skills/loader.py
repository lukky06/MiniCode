"""Discover and load built-in skill documents."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, Field, ValidationError


DEFAULT_SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills"


class SkillSummary(BaseModel):
    """Compact catalog entry exposed before a skill is loaded."""

    name: str = Field(pattern=r"^[a-z0-9-]+$")
    description: str = Field(min_length=1, max_length=1024)
    path: str


class Skill(BaseModel):
    """One loaded UTF-8 skill text resource."""

    name: str
    content: str
    path: str


class SkillLoader:
    """Discover ``<skill>/SKILL.md`` files and load them on demand."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_SKILL_ROOT

    def list_available(self) -> list[str]:
        """Return discovered skill names in stable order."""

        return [summary.name for summary in self._discover_summaries()]

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
        """Return frontmatter catalog entries without loading skill bodies."""

        summaries = self._discover_summaries()
        if names is None:
            return summaries
        by_name = {summary.name: summary for summary in summaries}
        return [by_name[name] for name in self.resolve_names(names) if name in by_name]

    def describe(self, name: str) -> SkillSummary:
        """Read and validate one skill's frontmatter metadata."""

        path = self._skill_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Skill not found: {name}")
        return _read_skill_summary(path)

    def load(self, name: str) -> Skill:
        """Load one complete skill document by exact name."""

        return self.read(name)

    def read(self, target: str) -> Skill:
        """Read one UTF-8 skill entry point or supporting resource."""

        normalized = target.strip().replace("\\", "/")
        raw_parts = normalized.split("/")
        resource = PurePosixPath(normalized)
        if (
            not normalized
            or resource.is_absolute()
            or any(part in {"", ".", ".."} for part in raw_parts)
        ):
            raise ValueError("Skill target must be a relative path without '.' or '..' segments.")

        parts = tuple(raw_parts)
        skill_name = parts[0]
        entry_path = self._skill_path(skill_name)
        if not entry_path.is_file():
            raise FileNotFoundError(f"Skill not found: {skill_name}")
        summary = _read_skill_summary(entry_path)

        if len(parts) == 1:
            path = entry_path
        else:
            skill_root = self._skill_root(skill_name)
            path = skill_root.joinpath(*parts[1:]).resolve()
            try:
                path.relative_to(skill_root)
            except ValueError as exc:
                raise ValueError(
                    f"Skill resource resolves outside its skill directory: {target}"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(f"Skill resource not found: {target}")

        payload = path.read_bytes()
        if b"\x00" in payload:
            raise ValueError(f"Skill resource must be a UTF-8 text resource: {target}")
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Skill resource must be a UTF-8 text resource: {target}") from exc
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        return Skill(name=summary.name, content=content, path=str(path))

    def _discover_summaries(self) -> list[SkillSummary]:
        if not self.root.exists():
            return []
        paths = sorted(self.root.glob("*/SKILL.md"), key=lambda path: path.parent.name)
        return [self.describe(path.parent.name) for path in paths]

    def _skill_root(self, name: str) -> Path:
        root = self.root.resolve()
        skill_root = (root / name).resolve()
        try:
            skill_root.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Skill path resolves outside the skill root: {name}") from exc
        return skill_root

    def _skill_path(self, name: str) -> Path:
        skill_root = self._skill_root(name)
        path = (skill_root / "SKILL.md").resolve()
        try:
            path.relative_to(skill_root)
        except ValueError as exc:
            raise ValueError(f"Skill entry point resolves outside its skill directory: {name}") from exc
        return path


def _read_skill_summary(path: Path) -> SkillSummary:
    """Read only YAML frontmatter and validate the skill catalog contract."""

    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "---":
            raise ValueError(f"Skill frontmatter is required: {path}")
        metadata_lines: list[str] = []
        for raw_line in handle:
            if raw_line.strip() == "---":
                break
            metadata_lines.append(raw_line)
        else:
            raise ValueError(f"Skill frontmatter is not closed: {path}")

    try:
        metadata = yaml.safe_load("".join(metadata_lines)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Skill frontmatter in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid Skill frontmatter in {path}: expected a mapping")

    name = str(metadata.get("name", "")).strip()
    description = str(metadata.get("description", "")).strip()
    if not name or not description:
        raise ValueError(f"Skill frontmatter must define name and description: {path}")
    if name != path.parent.name:
        raise ValueError(
            f"Skill frontmatter name must match directory name: {name!r} != {path.parent.name!r}"
        )
    try:
        return SkillSummary(name=name, description=description, path=str(path))
    except ValidationError as exc:
        raise ValueError(f"Invalid Skill frontmatter in {path}: {exc}") from exc
