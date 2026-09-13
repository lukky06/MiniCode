"""OpenAI model client adapter."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .client import ModelClient, ModelClientConfigurationError
from .normalization import (
    normalize_openai_chat_completion,
    normalize_openai_chat_completion_chunks,
    openai_chat_completion_chunk_reasoning_delta,
    openai_chat_completion_chunk_text_delta,
)
from .request import ModelRequest
from .types import ModelResponse


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class OpenAIModelClient(ModelClient):
    """Model client for OpenAI Chat Completions."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self.configure_capabilities("openai", self.model)
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ModelClientConfigurationError("OPENAI_API_KEY is required for provider openai.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelClientConfigurationError(
                "The openai package is required for provider openai."
            ) from exc

        self._client = OpenAI(api_key=resolved_api_key, base_url=base_url)

    def call_request(self, request: ModelRequest) -> ModelResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(request.as_chat_messages()),
            tools=request.tools,
            tool_choice=request.openai_tool_choice(),
            max_tokens=int(
                request.metadata.get("max_output_tokens")
                or self.capabilities.max_output_tokens
            ),
        )
        return normalize_openai_chat_completion(response)

    def stream_request(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(request.as_chat_messages()),
            tools=request.tools,
            tool_choice=request.openai_tool_choice(),
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=int(
                request.metadata.get("max_output_tokens")
                or self.capabilities.max_output_tokens
            ),
        )
        chunks: list[Any] = []
        for chunk in stream:
            chunks.append(chunk)
            reasoning_delta = openai_chat_completion_chunk_reasoning_delta(chunk)
            if reasoning_delta and on_reasoning_delta is not None:
                on_reasoning_delta(reasoning_delta)
            delta = openai_chat_completion_chunk_text_delta(chunk)
            if delta and on_text_delta is not None:
                on_text_delta(delta)
        return normalize_openai_chat_completion_chunks(chunks)


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id"),
                    "content": message.get("content", ""),
                }
            )
            continue

        converted_message = {
            "role": message["role"],
            "content": message.get("content", ""),
        }
        if "tool_calls" in message:
            converted_message["tool_calls"] = message["tool_calls"]
        converted.append(converted_message)
    return converted
