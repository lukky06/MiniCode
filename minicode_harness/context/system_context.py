"""Render compact, language-neutral repository hints for the system prompt."""

from __future__ import annotations

from pathlib import Path

from minicode_harness.workspace import WorkspaceProfile, scan_workspace_profile


MAX_LIST_ITEMS = 20


def render_repository_structure_card(
    workspace: Path | str,
    *,
    profile: WorkspaceProfile | None = None,
) -> str:
    """Render only bounded repository facts that directly guide tool use."""

    profile = profile or scan_workspace_profile(Path(workspace).resolve())
    sections = [
        _render_list("build_files", profile.build_files),
        _render_list("source_roots", profile.source_roots),
        _render_list("test_roots", profile.test_roots),
        _render_list(
            "preferred_verification_commands",
            profile.preferred_verification_commands,
        ),
    ]
    if len(profile.build_systems) > 1:
        sections.insert(0, _render_list("build_systems", profile.build_systems))
    return "\n".join(section for section in sections if section)


def _render_list(title: str, values: list[str]) -> str:
    if not values:
        return ""
    shown = values[:MAX_LIST_ITEMS]
    lines = [f"- {title}:", *(f"  - {value}" for value in shown)]
    if len(values) > len(shown):
        lines.append(f"  - ...<{len(values) - len(shown)} more omitted>")
    return "\n".join(lines)
