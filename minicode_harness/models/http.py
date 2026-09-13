"""Small JSON HTTP helpers for model adapters."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import ModelProviderError, looks_like_billing_error


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """POST JSON and return a decoded JSON object."""

    request = _json_request(url, payload, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise _http_provider_error(exc, body) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ModelProviderError(str(exc), kind="timeout") from exc
    except URLError as exc:
        raise ModelProviderError(str(exc), kind="connection") from exc


def stream_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> Iterator[dict[str, Any]]:
    """POST JSON and yield server-sent JSON chunks."""

    request = _json_request(url, payload, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    break
                yield json.loads(line)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise _http_provider_error(exc, body) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ModelProviderError(str(exc), kind="timeout") from exc
    except URLError as exc:
        raise ModelProviderError(str(exc), kind="connection") from exc


def _http_provider_error(exc: HTTPError, body: str) -> ModelProviderError:
    code = int(exc.code)
    lowered = body.lower()
    if looks_like_billing_error(lowered):
        kind = "billing"
    elif code == 429:
        kind = "rate_limited"
    elif code in {502, 503, 529}:
        kind = "overloaded"
    elif code == 413 or ("context" in lowered and "length" in lowered):
        kind = "prompt_too_long"
    elif code in {401, 403}:
        kind = "authentication"
    else:
        kind = "invalid_request" if 400 <= code < 500 else "unknown"
    retry_after = None
    try:
        raw_retry_after = exc.headers.get("Retry-After")
        retry_after = float(raw_retry_after) if raw_retry_after is not None else None
    except (AttributeError, TypeError, ValueError):
        retry_after = None
    return ModelProviderError(
        f"HTTP {code} from model provider: {body}",
        kind=kind,
        status_code=code,
        retry_after_seconds=retry_after,
    )


def _json_request(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> Request:
    return Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
