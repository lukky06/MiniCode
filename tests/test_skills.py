import pytest

from minicode_harness.skills import SkillLoader
from minicode_harness.tools import ToolRegistry


def _tool_names(registry: ToolRegistry) -> set[str]:
    return {schema["function"]["name"] for schema in registry.schemas()}


def test_skill_loader_loads_language_neutral_builtin_documents() -> None:
    loader = SkillLoader()

    available = loader.list_available()
    skill = loader.load("code-debug")

    assert available == [
        "code-debug",
        "refactor",
        "repo-explain",
        "review",
        "test-generation",
        "unit-test",
        "web-controller-test",
    ]
    assert skill.name == "code-debug"
    assert "any supported language or toolchain" in skill.content
    assert "java-debug" not in available
    assert "spring-controller-test" not in available


def test_test_generation_skill_is_framework_neutral_and_runnable() -> None:
    skill = SkillLoader().load("test-generation")

    assert "## Test evidence record" in skill.content
    assert "Dependency access" in skill.content
    assert "Lifecycle resources" in skill.content
    assert "## Strategy selection" in skill.content
    assert "## Mocking convention" in skill.content
    assert "## Delegation-contract coverage" in skill.content
    assert "None" in skill.content
    assert "whitespace-only" in skill.content
    assert "## Mockito/JUnit checks when applicable" in skill.content
    assert "doThrow" in skill.content
    assert "verifyNoInteractions" in skill.content
    assert "## Pre-write runnable review" in skill.content
    assert "constructor, factory, and import pattern" in skill.content
    assert "never infer field names or module re-exports" in skill.content
    assert "## Mutation-effective test review" in skill.content
    assert "plausible faulty implementation" in skill.content
    assert "degenerate inputs" in skill.content
    assert "test discovery or import failure" in skill.content
    assert "any language or framework" in skill.content
    assert "SpringBootTest" not in skill.content


def test_unit_test_skill_requires_supported_optional_test_setup() -> None:
    skill = SkillLoader().load("unit-test")

    assert "When tests are optional rather than required" in skill.content
    assert "uninstalled assertion library" in skill.content
    assert "invented value-object constructor" in skill.content
    assert "## Contract authority and reconciliation" in skill.content
    assert "A self-authored test is a hypothesis" in skill.content
    assert "shared validators, domain rules, and public API contracts" in skill.content
    assert "before downstream collaborators are called" in skill.content
    assert "rejection, clamping, defaulting, and normalization" in skill.content


def test_skill_loader_default_root_is_independent_from_current_workspace(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace-without-skills"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    loader = SkillLoader()

    assert "code-debug" in loader.list_available()
    assert loader.load("test-generation").name == "test-generation"


def test_skill_catalog_uses_frontmatter_summaries() -> None:
    summaries = SkillLoader().list_summaries(["code-debug", "unit-test"])

    assert [summary.name for summary in summaries] == ["code-debug", "unit-test"]
    assert "failing tests" in summaries[0].description
    assert "any supported language" in summaries[0].description
    assert "## Evidence checklist" not in summaries[0].description
    assert "## Verification convention" not in summaries[0].description


def test_skill_loader_discovers_valid_frontmatter_without_registered_names(tmp_path) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "custom-check"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: custom-check\n"
        "description: Use when a custom repository check is requested.\n"
        "---\n\n"
        "# Custom Check\n\n"
        "Detailed instructions.\n",
        encoding="utf-8",
    )

    loader = SkillLoader(root)

    assert loader.list_available() == ["custom-check"]
    summary = loader.describe("custom-check")
    assert summary.description == "Use when a custom repository check is requested."


def test_skill_loader_rejects_missing_or_mismatched_frontmatter(tmp_path) -> None:
    root = tmp_path / "skills"
    missing = root / "missing"
    missing.mkdir(parents=True)
    (missing / "SKILL.md").write_text("# Missing metadata\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frontmatter"):
        SkillLoader(root).list_available()

    (missing / "SKILL.md").write_text(
        "---\nname: other\ndescription: Use when testing metadata validation.\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="directory"):
        SkillLoader(root).list_available()


def test_skill_loader_reads_one_supporting_resource_on_demand(tmp_path) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "custom-check"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: custom-check\ndescription: Use when a custom check is requested.\n---\n\n# Check\n",
        encoding="utf-8",
    )
    (references / "details.md").write_text("# Details\nOnly load me when needed.\n", encoding="utf-8")

    loaded = SkillLoader(root).read("custom-check/references/details.md")

    assert loaded.name == "custom-check"
    assert loaded.content == "# Details\nOnly load me when needed.\n"
    assert loaded.path.endswith("custom-check\\references\\details.md") or loaded.path.endswith(
        "custom-check/references/details.md"
    )


def test_skill_loader_rejects_supporting_resource_path_escape(tmp_path) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "custom-check"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: custom-check\ndescription: Use when a custom check is requested.\n---\n",
        encoding="utf-8",
    )
    (root / "outside.md").write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="relative path"):
        SkillLoader(root).read("custom-check/../outside.md")
    with pytest.raises(ValueError, match="relative path"):
        SkillLoader(root).read("custom-check/./SKILL.md")


def test_skill_loader_rejects_supporting_resource_symlink_escape(tmp_path) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "custom-check"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: custom-check\ndescription: Use when a custom check is requested.\n---\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    link = references / "outside.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")

    with pytest.raises(ValueError, match="outside its skill directory"):
        SkillLoader(root).read("custom-check/references/outside.md")


def test_skill_loader_rejects_binary_supporting_resource(tmp_path) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "custom-check"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: custom-check\ndescription: Use when a custom check is requested.\n---\n",
        encoding="utf-8",
    )
    (skill_dir / "asset.bin").write_bytes(b"abc\x00def")

    with pytest.raises(ValueError, match="text resource"):
        SkillLoader(root).read("custom-check/asset.bin")


def test_skill_loader_resolves_explicit_names_without_intent_selection() -> None:
    loader = SkillLoader()

    assert loader.resolve_names(["repo-explain", "missing", "repo-explain", "refactor"]) == [
        "repo-explain",
        "refactor",
    ]
    assert loader.resolve_names([]) == []


def test_repo_explain_catalog_exposes_search_before_read_boundary() -> None:
    summary = SkillLoader().describe("repo-explain")

    assert "load this Skill before broad exploration" in summary.description
    assert "search shallowly before reading direct source" in summary.description
    assert "separate implementation evidence" in summary.description


def test_repo_explain_skill_uses_bounded_independent_delegation() -> None:
    skill = SkillLoader().load("repo-explain")

    assert "representative by default, not exhaustive" in skill.content
    assert "Stop expanding once the evidence is sufficient" in skill.content
    assert "Independent call chains or bounded subsystem questions may be delegated" in skill.content
    assert "Do not bundle several modules" in skill.content
    assert "Avoid overlapping delegation" in skill.content
    assert "does not repeat the same exploration" in skill.content
    assert "known short files" in skill.content
    assert "Use the weakest evidence level that is sufficient for the claim" in skill.content
    assert "implementation-confirmed" not in skill.content


def test_tool_registry_loads_full_skill_only_on_model_request(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loader = SkillLoader()
    registry = ToolRegistry(
        str(workspace),
        skill_loader=loader,
        skill_names=["repo-explain"],
    )

    assert "read" in _tool_names(registry)
    assert "load_skill" not in _tool_names(registry)
    schema = next(
        item["function"]
        for item in registry.schemas()
        if item["function"]["name"] == "read"
    )
    assert "repo-explain" not in schema["description"]
    assert "Skill" in schema["description"]
    loaded = registry.execute_admitted(
        registry.admit(
            "read",
            {"source": "skill", "target": "repo-explain"},
        )
    )
    assert loaded.name == "repo-explain"
    assert "# Skill: Repo Explain" in loaded.content


def test_tool_registry_reads_supporting_resource_only_for_available_skill(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "skills"
    skill_dir = root / "custom-check"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: custom-check\ndescription: Use when a custom check is requested.\n---\n",
        encoding="utf-8",
    )
    (references / "details.md").write_text("resource detail\n", encoding="utf-8")
    registry = ToolRegistry(
        str(workspace),
        skill_loader=SkillLoader(root),
        skill_names=["custom-check"],
    )

    loaded = registry.execute_admitted(
        registry.admit(
            "read",
            {"source": "skill", "target": "custom-check/references/details.md"},
        )
    )
    assert loaded.name == "custom-check"
    assert loaded.content == "resource detail\n"

    with pytest.raises(FileNotFoundError, match="not available"):
        registry.execute_admitted(
            registry.admit(
                "read",
                {"source": "skill", "target": "other/references/details.md"},
            )
        )


def test_tool_registry_omits_load_skill_when_skills_are_disabled(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = ToolRegistry(str(workspace), skill_loader=None, skill_names=[])

    assert "read" in _tool_names(registry)
    assert "load_skill" not in _tool_names(registry)
    assert registry.execute_admitted(
        registry.admit(
            "read",
            {"source": "skill", "target": "repo-explain"},
        )
    ) == {"status": "unavailable", "reason": "skills_disabled"}
