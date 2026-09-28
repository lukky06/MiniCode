"""Bounded, language-neutral workspace profiling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import os
import re
from typing import Iterable

from pydantic import BaseModel, Field

from .guard import WorkspaceGuard


MAX_PROFILE_FILES = 2000
MAX_DISCOVERED_ROOTS = 20

IGNORED_DIR_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "runs",
    "target",
    "vendor",
}

BUILD_FILE_RULES: dict[str, tuple[str, str]] = {
    "pyproject.toml": ("Python", "pyproject"),
    "pytest.ini": ("Python", "pytest"),
    "setup.cfg": ("Python", "setuptools"),
    "setup.py": ("Python", "setuptools"),
    "requirements.txt": ("Python", "pip"),
    "package.json": ("JavaScript/TypeScript", "npm"),
    "pnpm-lock.yaml": ("JavaScript/TypeScript", "pnpm"),
    "yarn.lock": ("JavaScript/TypeScript", "yarn"),
    "Cargo.toml": ("Rust", "Cargo"),
    "go.mod": ("Go", "Go modules"),
    "pom.xml": ("Java/Kotlin", "Maven"),
    "build.gradle": ("Java/Kotlin", "Gradle"),
    "build.gradle.kts": ("Java/Kotlin", "Gradle"),
    "gradlew": ("Java/Kotlin", "Gradle Wrapper"),
    "gradlew.bat": ("Java/Kotlin", "Gradle Wrapper"),
    "Gemfile": ("Ruby", "Bundler"),
    "composer.json": ("PHP", "Composer"),
    "mix.exs": ("Elixir", "Mix"),
    "Package.swift": ("Swift", "Swift Package Manager"),
    "CMakeLists.txt": ("C/C++", "CMake"),
    "Makefile": ("unknown", "Make"),
}

LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".h": "C/C++",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".scala": "Scala",
    ".sh": "Shell",
    ".ps1": "PowerShell",
    ".m": "MATLAB/Objective-C",
    ".lua": "Lua",
    ".r": "R",
    ".dart": "Dart",
}

TEST_DIR_NAMES = {"test", "tests", "spec", "specs", "__tests__"}
TEST_FILE_PATTERNS = (
    re.compile(r"^test_.*\.py$", re.IGNORECASE),
    re.compile(r".*_test\.py$", re.IGNORECASE),
    re.compile(r".*(?:test|tests|spec)\.(?:js|jsx|ts|tsx)$", re.IGNORECASE),
    re.compile(r".*_test\.go$", re.IGNORECASE),
    re.compile(r".*(?:Test|Tests)\.(?:java|kt|scala)$"),
    re.compile(r".*Tests?\.cs$"),
    re.compile(r".*_spec\.rb$", re.IGNORECASE),
)

CODE_SUFFIXES = frozenset(LANGUAGE_BY_SUFFIX)


class WorkspaceProfile(BaseModel):
    """Small deterministic repository profile used by the CLI and prompt card."""

    language: str | None = None
    languages: list[str] = Field(default_factory=list)
    build_system: str | None = None
    build_systems: list[str] = Field(default_factory=list)
    build_files: list[str] = Field(default_factory=list)
    source_roots: list[str] = Field(default_factory=list)
    test_roots: list[str] = Field(default_factory=list)


def scan_workspace_profile(workspace: Path | str) -> WorkspaceProfile:
    """Infer bounded repository facts without classifying the user's task."""

    root = WorkspaceGuard(workspace).root
    build_files = _discover_build_files(root)
    file_paths = _bounded_source_files(root)
    language_counts = Counter(
        LANGUAGE_BY_SUFFIX[path.suffix.lower()]
        for path in file_paths
        if path.suffix.lower() in LANGUAGE_BY_SUFFIX
    )
    inferred_languages, build_systems = _build_metadata(build_files)
    if not language_counts:
        for language in inferred_languages:
            language_counts.setdefault(language, 0)

    languages = [name for name, _ in language_counts.most_common()]
    source_roots = _discover_source_roots(root, file_paths)
    test_roots = _discover_test_roots(root, file_paths)

    return WorkspaceProfile(
        language=languages[0] if languages else None,
        languages=languages,
        build_system=_primary_build_system(build_systems),
        build_systems=build_systems,
        build_files=build_files,
        source_roots=source_roots,
        test_roots=test_roots,
    )


def is_code_path(path: str) -> bool:
    """Return whether a path has a recognized source-code suffix."""

    return Path(path.strip()).suffix.lower() in CODE_SUFFIXES


def extract_source_paths(text: str, *, limit: int = 20) -> list[str]:
    """Extract bounded workspace-like source paths from diagnostics."""

    suffixes = "|".join(
        re.escape(suffix.lstrip("."))
        for suffix in sorted(CODE_SUFFIXES, key=lambda value: (-len(value), value))
    )
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_./-])((?:[A-Za-z0-9_.@+-]+/)+[A-Za-z0-9_.@+-]+\.(?:{suffixes}))",
        re.IGNORECASE,
    )
    normalized = text.replace("\\", "/")
    found: list[str] = []
    for match in pattern.finditer(normalized):
        path = match.group(1)
        if path not in found:
            found.append(path)
            if len(found) >= limit:
                break
    return found


def _discover_build_files(root: Path) -> list[str]:
    found = [name for name in BUILD_FILE_RULES if (root / name).is_file()]
    for pattern in ("*.sln", "*.csproj", "*.fsproj"):
        for path in sorted(root.glob(pattern)):
            if path.name not in found:
                found.append(path.name)
    return found


def _build_metadata(build_files: list[str]) -> tuple[list[str], list[str]]:
    languages: list[str] = []
    systems: list[str] = []
    for name in build_files:
        if name.endswith((".sln", ".csproj", ".fsproj")):
            language, system = "C#/.NET", "dotnet"
        else:
            language, system = BUILD_FILE_RULES[name]
        if language != "unknown":
            languages.append(language)
        systems.append(system)
    return _dedupe(languages), _dedupe(systems)


def _bounded_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = [name for name in dir_names if name not in IGNORED_DIR_NAMES]
        current = Path(current_root)
        for name in sorted(file_names):
            path = current / name
            if path.suffix.lower() not in CODE_SUFFIXES:
                continue
            files.append(path)
            if len(files) >= MAX_PROFILE_FILES:
                return files
    return files


def _discover_source_roots(root: Path, files: list[Path]) -> list[str]:
    roots: list[str] = []
    for path in files:
        relative = path.relative_to(root)
        candidate = relative.parts[0] if len(relative.parts) > 1 else "."
        if candidate.lower() in TEST_DIR_NAMES:
            continue
        if candidate not in roots:
            roots.append(candidate)
            if len(roots) >= MAX_DISCOVERED_ROOTS:
                break
    return roots


def _discover_test_roots(root: Path, files: list[Path]) -> list[str]:
    roots: list[str] = []
    for path in files:
        relative = path.relative_to(root)
        parts = relative.parts[:-1]
        matched_index = next(
            (index for index, part in enumerate(parts) if part.lower() in TEST_DIR_NAMES),
            None,
        )
        if matched_index is not None:
            candidate = Path(*parts[: matched_index + 1]).as_posix()
        elif _looks_like_test_file(path.name):
            candidate = Path(*parts).as_posix() if parts else "."
        else:
            continue
        if candidate not in roots:
            roots.append(candidate)
            if len(roots) >= MAX_DISCOVERED_ROOTS:
                break
    return roots


def _looks_like_test_file(filename: str) -> bool:
    return any(pattern.fullmatch(filename) for pattern in TEST_FILE_PATTERNS)


def _primary_build_system(build_systems: list[str]) -> str | None:
    priority = (
        "pnpm",
        "yarn",
        "npm",
        "Gradle Wrapper",
        "Gradle",
        "Maven",
        "pyproject",
        "pytest",
        "setuptools",
        "pip",
        "Cargo",
        "Go modules",
        "dotnet",
        "Bundler",
        "Composer",
        "Mix",
        "Swift Package Manager",
        "CMake",
        "Make",
    )
    return _first_available(build_systems, priority) or (build_systems[0] if build_systems else None)


def _first_available(values: Iterable[str], preferred: tuple[str, ...]) -> str | None:
    value_set = set(values)
    return next((value for value in preferred if value in value_set), None)


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
