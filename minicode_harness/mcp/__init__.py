"""Optional MCP tool integration."""

from .client import (
    InProcessMCPServer,
    MCPConfig,
    MCPManager,
    MCPServer,
    MCPServerConfig,
    MCPToolAnnotations,
    MCPToolSpec,
    StdioMCPServer,
    normalize_mcp_name,
)

__all__ = [
    "InProcessMCPServer",
    "MCPConfig",
    "MCPManager",
    "MCPServer",
    "MCPServerConfig",
    "MCPToolAnnotations",
    "MCPToolSpec",
    "StdioMCPServer",
    "normalize_mcp_name",
]
