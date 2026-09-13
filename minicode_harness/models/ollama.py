"""Ollama local model client adapter."""

from __future__ import annotations

import os
from .client import ModelClient, ModelClientConfigurationError
from .http import post_json
from .openai import _to_openai_messages
from .normalization import normalize_ollama_chat_response
from .request import ModelRequest
from .types import ModelResponse


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaModelClient(ModelClient):
    """Model client for Ollama's local chat API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL")
        if not self.model:
            raise ModelClientConfigurationError(
                "An Ollama model is required. Pass --model or set OLLAMA_MODEL."
            )
        self.configure_capabilities("ollama", self.model)
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")

    def call_request(self, request: ModelRequest) -> ModelResponse:
        response = post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": _to_openai_messages(request.as_chat_messages()),
                "tools": request.tools,
                "stream": False,
                "options": {
                    "num_ctx": self.capabilities.context_window,
                    "num_predict": int(
                        request.metadata.get("max_output_tokens")
                        or self.capabilities.max_output_tokens
                    ),
                },
            },
        )
        return normalize_ollama_chat_response(response)
