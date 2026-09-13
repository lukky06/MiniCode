"""Anthropic model client adapter."""

from __future__ import annotations

import json
import os
from typing import Any

from .client import ModelClient, ModelClientConfigurationError
from .normalization import normalize_anthropic_message
from .request import ModelRequest
from .types import ModelResponse


DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_MAX_TOKENS = 4096


class AnthropicModelClient(ModelClient):
    """Model client for Anthropic Messages."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        self.configure_capabilities("anthropic", self.model)
        resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            raise ModelClientConfigurationError(
                "ANTHROPIC_API_KEY is required for provider anthropic."
            )

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ModelClientConfigurationError(
                "The anthropic package is required for provider anthropic."
            ) from exc

        self._client = Anthropic(api_key=resolved_api_key)

    def call_request(self, request: ModelRequest) -> ModelResponse:
        embedded_system, anthropic_messages = _to_anthropic_messages(request.messages)
        system = _join_system(request.system, embedded_system)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(
                request.metadata.get("max_output_tokens")
                or self.capabilities.max_output_tokens
                or DEFAULT_MAX_TOKENS
            ),
            "system": system or None,
            "messages": anthropic_messages,
            "tools": [_to_anthropic_tool(tool) for tool in request.tools],
        }
        if required_tool_name := request.required_tool_name():
            payload["tool_choice"] = {"type": "tool", "name": required_tool_name}
        response = self._client.messages.create(**payload)
        return normalize_anthropic_message(response)


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            system_parts.append(str(message.get("content", "")))
            continue
        if role == "assistant" and message.get("tool_calls"):
            converted.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": _parse_tool_arguments(
                                tool_call["function"].get("arguments", {})
                            ),
                        }
                        for tool_call in message["tool_calls"]
                    ],
                }
            )
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id"),
                            "content": str(message.get("content", "")),
                        }
                    ],
                }
            )
            continue
        converted.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": str(message.get("content", "")),
            }
        )

    return "\n\n".join(system_parts), _merge_consecutive_messages(converted)


def _merge_consecutive_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize adjacent same-role messages for Anthropic's alternating protocol."""

    merged: list[dict[str, Any]] = []
    for message in messages:
        if not merged or merged[-1].get("role") != message.get("role"):
            merged.append(message)
            continue
        merged[-1] = {
            "role": message["role"],
            "content": [
                *_anthropic_content_blocks(merged[-1].get("content")),
                *_anthropic_content_blocks(message.get("content")),
            ],
        }
    return merged


def _anthropic_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [dict(block) for block in content]
    return [{"type": "text", "text": str(content or "")}]


def _join_system(primary: str, embedded: str) -> str:
    return "\n\n".join(part.strip() for part in (primary, embedded) if part.strip())


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function["parameters"],
    }


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return {"_raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return dict(arguments or {})
