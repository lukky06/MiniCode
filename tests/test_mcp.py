import json

import pytest

import minicode_harness.mcp.client as mcp_client
from minicode_harness.mcp import (
    InProcessMCPServer,
    MCPManager,
    MCPServerConfig,
    MCPToolAnnotations,
    MCPToolSpec,
    normalize_mcp_name,
)
from minicode_harness.policy import RiskLevel
from minicode_harness.tools import ToolRegistry


def test_mcp_manager_discovers_namespaced_readonly_tool(tmp_path) -> None:
    server = InProcessMCPServer(
        "docs server",
        [
            MCPToolSpec(
                server_name="docs server",
                name="search/docs",
                description="Search documentation",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                annotations=MCPToolAnnotations(read_only=True, destructive=False),
            )
        ],
        {"search/docs": lambda arguments: f"found:{arguments['query']}"},
    )
    manager = MCPManager([server])
    registry = ToolRegistry(str(tmp_path), mcp_manager=manager)

    name = "mcp__docs_server__search_docs"
    schemas = {schema["function"]["name"]: schema for schema in registry.schemas()}

    assert normalize_mcp_name("docs server") == "docs_server"
    assert name in schemas
    assert registry.risk_level(name) == RiskLevel.LOW
    admission = registry.admit(name, {"query": "agent loop"})
    assert registry.execute_admitted(admission) == "found:agent loop"


def test_destructive_mcp_tool_uses_existing_approval_risk(tmp_path) -> None:
    server = InProcessMCPServer(
        "deploy",
        [
            MCPToolSpec(
                server_name="deploy",
                name="release",
                annotations=MCPToolAnnotations(read_only=False, destructive=True),
            )
        ],
        {"release": lambda arguments: "released"},
    )
    registry = ToolRegistry(str(tmp_path), mcp_manager=MCPManager([server]))

    assert registry.risk_level("mcp__deploy__release") == RiskLevel.HIGH
    assert registry.requires_approval("mcp__deploy__release")
    preview = registry.preview_admitted(
        registry.admit("mcp__deploy__release", {"environment": "staging"})
    )
    assert preview["summary"]["arguments"] == {"environment": "staging"}


def test_missing_mcp_annotations_default_to_side_effecting(tmp_path) -> None:
    server = InProcessMCPServer(
        "tickets",
        [MCPToolSpec(server_name="tickets", name="create")],
        {"create": lambda arguments: {"id": "T-1"}},
    )
    registry = ToolRegistry(str(tmp_path), mcp_manager=MCPManager([server]))
    name = "mcp__tickets__create"

    assert registry.risk_level(name) == RiskLevel.MEDIUM
    assert registry.requires_approval(name)
    assert registry.history_effects()[name] == {
        "read_only": False,
        "destructive": False,
        "result_reconstructible": False,
    }
    schema = next(item for item in registry.schemas() if item["function"]["name"] == name)
    rendered_schema = json.dumps(schema)
    assert "read_only" not in rendered_schema
    assert "result_reconstructible" not in rendered_schema


def test_mcp_manager_applies_include_then_exclude_filters(tmp_path) -> None:
    server = InProcessMCPServer(
        "workspace",
        [
            MCPToolSpec(server_name="workspace", name="read_file"),
            MCPToolSpec(server_name="workspace", name="query_metadata"),
            MCPToolSpec(server_name="workspace", name="write_file"),
        ],
        {
            "read_file": lambda arguments: "read",
            "query_metadata": lambda arguments: "metadata",
            "write_file": lambda arguments: "write",
        },
    )
    manager = MCPManager(
        [server],
        tool_filters={
            "workspace": (
                ["read_file", "query_metadata"],
                ["read_file"],
            )
        },
    )
    registry = ToolRegistry(str(tmp_path), mcp_manager=manager)
    names = {schema["function"]["name"] for schema in registry.schemas()}

    assert "mcp__workspace__query_metadata" in names
    assert "mcp__workspace__read_file" not in names
    assert "mcp__workspace__write_file" not in names


def test_mcp_filter_runs_before_duplicate_registry_name_check() -> None:
    server = InProcessMCPServer(
        "docs",
        [
            MCPToolSpec(server_name="docs", name="search/docs"),
            MCPToolSpec(server_name="docs", name="search docs"),
        ],
        {
            "search/docs": lambda arguments: "slash",
            "search docs": lambda arguments: "space",
        },
    )
    manager = MCPManager(
        [server],
        tool_filters={"docs": (["search/docs"], [])},
    )

    assert [tool.name for tool in manager.list_tools()] == ["search/docs"]


def test_mcp_server_config_validates_command_and_filter_shape() -> None:
    config = MCPServerConfig.model_validate(
        {
            "command": "python",
            "include_tools": ["query_metadata"],
            "exclude_tools": ["read_file"],
        }
    )

    assert config.command == ["python"]
    assert config.include_tools == ["query_metadata"]
    assert config.exclude_tools == ["read_file"]

    with pytest.raises(ValueError):
        MCPServerConfig.model_validate({"transport": "http", "command": ["server"]})


def test_mcp_config_file_applies_filters_before_registry_exposure(
    tmp_path,
    monkeypatch,
) -> None:
    server = InProcessMCPServer(
        "docs",
        [
            MCPToolSpec(server_name="docs", name="search_docs"),
            MCPToolSpec(server_name="docs", name="read_file"),
        ],
        {
            "search_docs": lambda arguments: "found",
            "read_file": lambda arguments: "duplicate",
        },
    )
    created: dict[str, object] = {}

    def fake_stdio_server(**kwargs):
        created.update(kwargs)
        return server

    monkeypatch.setattr(mcp_client, "StdioMCPServer", fake_stdio_server)
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "docs": {
                        "transport": "stdio",
                        "command": ["python", "server.py"],
                        "cwd": "servers",
                        "include_tools": ["search_docs", "read_file"],
                        "exclude_tools": ["read_file"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager.from_config_file(config_path)

    assert [tool.name for tool in manager.list_tools()] == ["search_docs"]
    assert created["command"] == ["python", "server.py"]
    assert created["cwd"] == str((tmp_path / "servers").resolve())
