"""Provider-neutral model capability metadata.

Capabilities configure the harness. They are never rendered into model-visible
context and do not classify user intent.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelCapabilities(BaseModel):
    context_window: int = 32_000
    max_output_tokens: int = 4_096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_prompt_cache: bool = False

    @property
    def reserved_output_tokens(self) -> int:
        return min(self.max_output_tokens, max(1_024, self.context_window // 4))


def resolve_model_capabilities(provider: str, model: str | None = None) -> ModelCapabilities:
    """Return conservative capabilities for known provider/model families."""

    provider_name = provider.strip().lower()
    model_name = (model or "").strip().lower()

    if provider_name == "anthropic":
        return ModelCapabilities(
            context_window=200_000,
            max_output_tokens=8_192,
            supports_prompt_cache=True,
        )
    if provider_name in {"openai"}:
        context_window = 128_000
        if any(marker in model_name for marker in ("gpt-5", "o3", "o4")):
            context_window = 200_000
        return ModelCapabilities(
            context_window=context_window,
            max_output_tokens=16_384,
            supports_prompt_cache=True,
        )
    if provider_name in {"qwen", "dashscope", "tongyi"}:
        return ModelCapabilities(
            context_window=131_072,
            max_output_tokens=8_192,
            supports_prompt_cache=True,
        )
    if provider_name == "deepseek":
        return ModelCapabilities(
            context_window=1_000_000,
            max_output_tokens=8_192,
            supports_prompt_cache=True,
        )
    if provider_name in {"kimi", "moonshot"}:
        return ModelCapabilities(
            context_window=1_000_000,
            max_output_tokens=131_072,
            supports_prompt_cache=True,
        )
    if provider_name == "ollama":
        return ModelCapabilities(
            context_window=32_000,
            max_output_tokens=4_096,
            supports_prompt_cache=False,
        )
    return ModelCapabilities()
