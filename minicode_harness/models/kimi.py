"""Kimi / Moonshot OpenAI-compatible model client adapter."""

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


DEFAULT_KIMI_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_KIMI_MODEL = "kimi-k3"


class KimiModelClient(ModelClient):
    """Model client for Kimi through Moonshot's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = (
            model
            or os.environ.get("KIMI_MODEL")
            or os.environ.get("MOONSHOT_MODEL")
            or DEFAULT_KIMI_MODEL
        )
        self.configure_capabilities("kimi", self.model)
        self.base_url = (
            base_url
            or os.environ.get("KIMI_BASE_URL")
            or os.environ.get("MOONSHOT_BASE_URL")
            or DEFAULT_KIMI_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("MOONSHOT_API_KEY")
            or os.environ.get("KIMI_API_KEY")
        )
        if not self.api_key:
            raise ModelClientConfigurationError(
                "MOONSHOT_API_KEY or KIMI_API_KEY is required for provider kimi."
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
        chunks: list[dict[str, Any]] = []
        payload = {**self._request_payload(request), "stream": True}
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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_kimi_messages(request.as_chat_messages()),
            "max_completion_tokens": int(
                request.metadata.get("max_output_tokens")
                or self.capabilities.max_output_tokens
            ),
        }
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.openai_tool_choice()
        reasoning_effort = request.metadata.get("reasoning_effort")
        if isinstance(reasoning_effort, str) and reasoning_effort.strip():
            payload["reasoning_effort"] = reasoning_effort.strip()
        return payload


def _to_kimi_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve Kimi reasoning state required by multi-turn tool calls."""

    converted = _to_openai_messages(messages)
    for source, target in zip(messages, converted, strict=True):
        reasoning_content = source.get("reasoning_content")
        if source.get("role") == "assistant" and isinstance(reasoning_content, str):
            target["reasoning_content"] = reasoning_content
    return converted
