"""Model client abstractions and provider adapters."""

from .capabilities import ModelCapabilities, resolve_model_capabilities
from .client import (
    ModelClient,
    ModelClientConfigurationError,
    UnsupportedProviderError,
    create_model_client,
)
from .errors import ModelCallFailure, ModelProviderError, classify_model_exception
from .kimi import DEFAULT_KIMI_BASE_URL, DEFAULT_KIMI_MODEL, KimiModelClient
from .qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL, QwenModelClient
from .request import ModelRequest
from .types import ModelResponse, ModelUsage, NormalizedToolCall

__all__ = [
    "ModelCallFailure",
    "ModelCapabilities",
    "ModelClient",
    "DEFAULT_KIMI_BASE_URL",
    "DEFAULT_KIMI_MODEL",
    "DEFAULT_QWEN_BASE_URL",
    "DEFAULT_QWEN_MODEL",
    "ModelClientConfigurationError",
    "ModelProviderError",
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
    "NormalizedToolCall",
    "KimiModelClient",
    "QwenModelClient",
    "UnsupportedProviderError",
    "classify_model_exception",
    "create_model_client",
    "resolve_model_capabilities",
]
