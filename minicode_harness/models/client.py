"""Model client abstraction and provider factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from .capabilities import ModelCapabilities, resolve_model_capabilities
from .request import ModelRequest
from .types import ModelResponse


class ModelClientConfigurationError(RuntimeError):
    """Raised when a model client cannot be configured."""


class UnsupportedProviderError(ModelClientConfigurationError):
    """Raised when a provider name is not supported."""


class ModelClient(ABC):
    """Provider-neutral model client used by the Agent Loop."""

    capabilities: ModelCapabilities = ModelCapabilities()

    @abstractmethod
    def call_request(self, request: ModelRequest) -> ModelResponse:
        """Call a model using one exact prepared request."""

    def stream_request(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Stream one prepared request, falling back to one non-streaming request."""

        response = self.call_request(request)
        if response.reasoning_content and on_reasoning_delta is not None:
            on_reasoning_delta(response.reasoning_content)
        if response.final_text and on_text_delta is not None:
            on_text_delta(response.final_text)
        return response

    def configure_capabilities(self, provider: str, model: str | None = None) -> None:
        """Set conservative provider/model capabilities for harness budgeting."""

        self.capabilities = resolve_model_capabilities(provider, model)


def create_model_client(provider: str, model: str | None = None) -> ModelClient:
    """Create a provider-specific model client."""

    normalized_provider = provider.strip().lower()
    if normalized_provider == "openai":
        from .openai import OpenAIModelClient

        return OpenAIModelClient(model=model)
    if normalized_provider == "anthropic":
        from .anthropic import AnthropicModelClient

        return AnthropicModelClient(model=model)
    if normalized_provider == "deepseek":
        from .deepseek import DeepSeekModelClient

        return DeepSeekModelClient(model=model)
    if normalized_provider in {"qwen", "dashscope", "tongyi"}:
        from .qwen import QwenModelClient

        return QwenModelClient(model=model)
    if normalized_provider in {"kimi", "moonshot"}:
        from .kimi import KimiModelClient

        return KimiModelClient(model=model)
    if normalized_provider == "ollama":
        from .ollama import OllamaModelClient

        return OllamaModelClient(model=model)

    raise UnsupportedProviderError(
        "Unsupported provider. Expected one of: openai, anthropic, deepseek, qwen, dashscope, tongyi, kimi, moonshot, ollama."
    )
