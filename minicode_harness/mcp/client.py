"""Small MCP tool-discovery and invocation layer.

The first implementation intentionally supports only explicit in-process test
adapters and local stdio servers. It does not add MCP resources or prompts to
model context; discovered MCP tools enter the existing ``ToolRegistry``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator


_MCP_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


class MCPServerConfig(BaseModel):
    """Validated configuration for one explicit local MCP server."""

    transport: Literal["stdio"] = "stdio"
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(30.0, gt=0, le=300)
    include_tools: list[str] | None = None
    exclude_tools: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("command", mode="before")
    @classmethod
    def normalize_command(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("MCP stdio command must contain non-empty strings.")
        return value


class MCPConfig(BaseModel):
    """Top-level MCP configuration file schema."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class MCPToolAnnotations(BaseModel):
    read_only: bool = False
    destructive: bool = False


class MCPToolSpec(BaseModel):
    server_name: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    annotations: MCPToolAnnotations = Field(default_factory=MCPToolAnnotations)

    @property
    def registry_name(self) -> str:
        return f"mcp__{normalize_mcp_name(self.server_name)}__{normalize_mcp_name(self.name)}"


class MCPServer(Protocol):
    name: str

    def list_tools(self) -> list[MCPToolSpec]:
        ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        ...

    def close(self) -> None:
        ...


class InProcessMCPServer:
    """Deterministic adapter used by tests and embedded callers."""

    def __init__(
        self,
        name: str,
        tools: list[MCPToolSpec],
        handlers: Mapping[str, Callable[[dict[str, Any]], Any]],
    ) -> None:
        self.name = name
        self._tools = [
            tool if tool.server_name == name else tool.model_copy(update={"server_name": name})
            for tool in tools
        ]
        self._handlers = dict(handlers)

    def list_tools(self) -> list[MCPToolSpec]:
        return [tool.model_copy(deep=True) for tool in self._tools]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        handler = self._handlers.get(name)
        if handler is None:
            raise KeyError(f"Unknown MCP tool {self.name}/{name}")
        return handler(dict(arguments))

    def close(self) -> None:
        return None


class StdioMCPServer:
    """Minimal persistent JSON-RPC client for a local MCP stdio server."""

    def __init__(
        self,
        *,
        name: str,
        command: list[str],
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("MCP stdio command must not be empty.")
        self.name = name
        self.command = list(command)
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else None
        self.env = {
            **os.environ,
            **{str(key): str(value) for key, value in dict(env or {}).items()},
        }
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._initialized = False
        self._tools: list[MCPToolSpec] | None = None

    def list_tools(self) -> list[MCPToolSpec]:
        with self._lock:
            self._ensure_initialized()
            if self._tools is None:
                result = self._request("tools/list", {})
                raw_tools = result.get("tools") if isinstance(result, dict) else []
                self._tools = [self._tool_spec(value) for value in raw_tools or []]
            return [tool.model_copy(deep=True) for tool in self._tools]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        with self._lock:
            self._ensure_initialized()
            result = self._request("tools/call", {"name": name, "arguments": arguments})
            if isinstance(result, dict) and result.get("isError"):
                raise RuntimeError(_render_mcp_content(result.get("content")))
            return _render_mcp_content(result.get("content")) if isinstance(result, dict) else result

    def close(self) -> None:
        process = self._process
        self._process = None
        self._initialized = False
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _ensure_initialized(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._initialized = False
            self._tools = None
        if self._initialized:
            return
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "minicode", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})
        self._initialized = True

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError(f"MCP server {self.name!r} is not running.")
        self._request_id += 1
        request_id = self._request_id
        process.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                ensure_ascii=False,
            )
            + "\n"
        )
        process.stdin.flush()
        while True:
            line = self._readline_with_timeout(process, method)
            if not line:
                raise RuntimeError(f"MCP server {self.name!r} closed while handling {method}.")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") != request_id:
                continue
            if "error" in payload:
                raise RuntimeError(f"MCP {method} failed: {payload['error']}")
            return payload.get("result")

    def _readline_with_timeout(
        self,
        process: subprocess.Popen[str],
        method: str,
    ) -> str:
        if process.stdout is None:
            raise RuntimeError(f"MCP server {self.name!r} has no stdout pipe.")
        results: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                results.put(process.stdout.readline())
            except BaseException as exc:  # pragma: no cover - defensive pipe path
                results.put(exc)

        threading.Thread(target=read_line, daemon=True).start()
        try:
            result = results.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(
                f"MCP server {self.name!r} timed out after {self.timeout_seconds}s "
                f"while handling {method}."
            ) from exc
        if isinstance(result, BaseException):
            raise RuntimeError(
                f"MCP server {self.name!r} failed while handling {method}: {result}"
            ) from result
        return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        process.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        )
        process.stdin.flush()

    def _tool_spec(self, payload: Any) -> MCPToolSpec:
        raw = payload if isinstance(payload, dict) else {}
        annotations = raw.get("annotations") or {}
        return MCPToolSpec(
            server_name=self.name,
            name=str(raw.get("name") or "unnamed"),
            description=str(raw.get("description") or ""),
            input_schema=dict(raw.get("inputSchema") or {"type": "object"}),
            annotations=MCPToolAnnotations(
                read_only=bool(annotations.get("readOnlyHint", False)),
                destructive=bool(annotations.get("destructiveHint", False)),
            ),
        )


class MCPManager:
    """Own configured MCP servers and expose a filtered flat tool namespace."""

    def __init__(
        self,
        servers: list[MCPServer] | None = None,
        *,
        tool_filters: Mapping[str, tuple[list[str] | None, list[str]]] | None = None,
    ) -> None:
        self._servers = {server.name: server for server in servers or []}
        self._tool_filters = {
            str(name): (
                list(include) if include is not None else None,
                list(exclude),
            )
            for name, (include, exclude) in dict(tool_filters or {}).items()
        }
        self._tools: dict[str, MCPToolSpec] | None = None

    @classmethod
    def from_config_file(cls, path: Path | str) -> "MCPManager":
        config_path = Path(path).resolve()
        config = MCPConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        servers: list[MCPServer] = []
        filters: dict[str, tuple[list[str] | None, list[str]]] = {}
        for name, server_config in config.servers.items():
            cwd = server_config.cwd
            if cwd is None:
                cwd = str(config_path.parent)
            elif not Path(cwd).is_absolute():
                cwd = str((config_path.parent / cwd).resolve())
            servers.append(
                StdioMCPServer(
                    name=str(name),
                    command=list(server_config.command),
                    cwd=cwd,
                    env=server_config.env,
                    timeout_seconds=server_config.timeout_seconds,
                )
            )
            filters[str(name)] = (
                list(server_config.include_tools)
                if server_config.include_tools is not None
                else None,
                list(server_config.exclude_tools),
            )
        return cls(servers, tool_filters=filters)

    def list_tools(self) -> list[MCPToolSpec]:
        if self._tools is None:
            discovered: dict[str, MCPToolSpec] = {}
            for server in self._servers.values():
                include_tools, exclude_tools = self._tool_filters.get(
                    server.name,
                    (None, []),
                )
                for tool in server.list_tools():
                    if not _tool_is_exposed(
                        tool.name,
                        include_tools=include_tools,
                        exclude_tools=exclude_tools,
                    ):
                        continue
                    if tool.registry_name in discovered:
                        raise ValueError(f"Duplicate MCP tool name: {tool.registry_name}")
                    discovered[tool.registry_name] = tool
            self._tools = discovered
        return [tool.model_copy(deep=True) for tool in self._tools.values()]

    def call_tool(self, registry_name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tool_map().get(registry_name)
        if tool is None:
            raise KeyError(f"Unknown MCP registry tool: {registry_name}")
        server = self._servers[tool.server_name]
        return server.call_tool(tool.name, arguments)

    def close(self) -> None:
        for server in self._servers.values():
            server.close()

    def _tool_map(self) -> dict[str, MCPToolSpec]:
        if self._tools is None:
            self.list_tools()
        return self._tools or {}


def _tool_is_exposed(
    tool_name: str,
    *,
    include_tools: list[str] | None,
    exclude_tools: list[str],
) -> bool:
    """Apply exact-name include filtering before exact-name exclusion."""

    if include_tools is not None and tool_name not in include_tools:
        return False
    return tool_name not in exclude_tools


def normalize_mcp_name(name: str) -> str:
    normalized = _MCP_NAME_RE.sub("_", name.strip())
    return normalized or "unnamed"


def _render_mcp_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False, default=str)
