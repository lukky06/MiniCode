"""Provider-neutral model-call error classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ModelErrorKind = Literal[
    "rate_limited",
    "overloaded",
    "timeout",
    "connection",
    "prompt_too_long",
    "authentication",
    "billing",
    "invalid_request",
    "unknown",
]


@dataclass(frozen=True)
class ModelCallFailure:
    kind: ModelErrorKind
    message: str
    retryable: bool
    status_code: int | None = None
    retry_after_seconds: float | None = None


class ModelProviderError(RuntimeError):
    """Normalized provider failure preserving retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        kind: ModelErrorKind = "unknown",
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.retryable = (
            retryable
            if retryable is not None
            else kind in {"rate_limited", "overloaded", "timeout", "connection"}
        )


def classify_model_exception(exc: Exception) -> ModelCallFailure:
    """Classify SDK and HTTP exceptions without depending on one provider."""

    if isinstance(exc, ModelProviderError):
        return ModelCallFailure(
            kind=exc.kind,
            message=str(exc),
            retryable=exc.retryable,
            status_code=exc.status_code,
            retry_after_seconds=exc.retry_after_seconds,
        )

    status_code = _optional_int(
        getattr(exc, "status_code", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
        or getattr(exc, "code", None)
    )
    retry_after = _retry_after(exc)
    text = f"{type(exc).__name__}: {exc}".lower()

    if looks_like_billing_error(text):
        kind: ModelErrorKind = "billing"
    elif status_code in {401, 403} or "authentication" in text or "api key" in text:
        kind = "authentication"
    elif status_code == 429 or "rate limit" in text or "too many requests" in text:
        kind = "rate_limited"
    elif status_code in {502, 503, 529} or "overloaded" in text or "temporarily unavailable" in text:
        kind = "overloaded"
    elif status_code == 413 or "prompt too long" in text or "context length" in text or "maximum context" in text:
        kind = "prompt_too_long"
    elif "timeout" in text or "timed out" in text:
        kind = "timeout"
    elif "connection" in text or "econnreset" in text or "broken pipe" in text:
        kind = "connection"
    elif status_code is not None and 400 <= status_code < 500:
        kind = "invalid_request"
    else:
        kind = "unknown"

    return ModelCallFailure(
        kind=kind,
        message=str(exc),
        retryable=kind in {"rate_limited", "overloaded", "timeout", "connection"},
        status_code=status_code,
        retry_after_seconds=retry_after,
    )


def looks_like_billing_error(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "arrearage",
            "overdue-payment",
            "overdue payment",
            "insufficient balance",
            "billing account",
        )
    )


def _retry_after(exc: Exception) -> float | None:
    direct = getattr(exc, "retry_after", None)
    if direct is not None:
        return _optional_float(direct)
    headers: Any = getattr(exc, "headers", None) or getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            value = None
        return _optional_float(value)
    return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
