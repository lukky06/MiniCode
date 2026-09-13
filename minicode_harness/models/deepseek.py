"""DeepSeek OpenAI-compatible model client adapter."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .client import ModelClient, ModelClientConfigurationError
from .http import post_json, stream_post_json
from .normalization import (
    normalize_openai_chat_completion,
    normalize_openai_chat_completion_chunks,
    openai_chat_completion_chunk_reasoning_delta,
    openai_chat_completion_chunk_text_delta,
)
from .openai import _to_openai_messages
from .request import ModelRequest
from .types import ModelResponse


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class DeepSeekModelClient(ModelClient):
    """Model client for DeepSeek's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        self.configure_capabilities("deepseek", self.model)
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or DEFAULT_DEEPSEEK_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ModelClientConfigurationError(
                "DEEPSEEK_API_KEY is required for provider deepseek."
            )

    def call_request(self, request: ModelRequest) -> ModelResponse:
        response = post_json(
            f"{self.base_url}/chat/completions",
            self._request_payload(request),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return normalize_openai_chat_completion(response)

    def stream_request(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Stream DeepSeek's OpenAI-compatible SSE response incrementally."""

        payload = self._request_payload(request)
        payload["stream"] = True
        chunks: list[dict[str, Any]] = []
        for chunk in stream_post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ):
            chunks.append(chunk)
            reasoning_delta = openai_chat_completion_chunk_reasoning_delta(chunk)
            if reasoning_delta and on_reasoning_delta is not None:
                on_reasoning_delta(reasoning_delta)
            delta = openai_chat_completion_chunk_text_delta(chunk)
            if delta and on_text_delta is not None:
                on_text_delta(delta)
        return normalize_openai_chat_completion_chunks(chunks)

    def _request_payload(self, request: ModelRequest) -> dict[str, Any]:
        """Build the exact DeepSeek payload for one provider-neutral request."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_deepseek_messages(request.as_chat_messages()),
            "tools": request.tools,
            "tool_choice": request.openai_tool_choice(),
            "max_tokens": int(
                request.metadata.get("max_output_tokens")
                or self.capabilities.max_output_tokens
            ),
        }
        if (temperature := request.temperature()) is not None:
            payload["temperature"] = temperature
        reasoning_effort = request.metadata.get("reasoning_effort")
        if isinstance(reasoning_effort, str) and reasoning_effort.strip():
            payload["reasoning_effort"] = reasoning_effort.strip()
        return payload


def _to_deepseek_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render OpenAI-compatible messages while preserving DeepSeek reasoning state."""

    converted = _to_openai_messages(messages)
    for source, target in zip(messages, converted, strict=True):
        reasoning_content = source.get("reasoning_content")
        if source.get("role") == "assistant" and isinstance(reasoning_content, str):
            target["reasoning_content"] = reasoning_content
    return converted
