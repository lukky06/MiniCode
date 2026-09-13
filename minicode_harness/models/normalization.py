"""Normalize provider-specific tool calling responses."""

from __future__ import annotations

from collections.abc import Mapping
from html import unescape
import json
import re
import shlex
from typing import Any

from .types import ModelResponse, ModelUsage, NormalizedToolCall


def normalize_openai_chat_completion(raw_response: Any) -> ModelResponse:
    """Normalize an OpenAI-compatible chat completion response."""

    raw = _to_mapping(raw_response)
    choices = _get(raw, "choices") or []
    message = _get(choices[0], "message") if choices else {}
    content = _get(message, "content")
    reasoning_content = _join_text(_get(message, "reasoning_content"))
    final_text = _join_text(content)
    text_tool_calls, cleaned_final_text = _extract_dsml_tool_calls(final_text)
    tool_calls = [
        _normalize_openai_tool_call(tool_call)
        for tool_call in (_get(message, "tool_calls") or [])
    ]
    if not tool_calls:
        tool_calls = text_tool_calls
        final_text = cleaned_final_text

    return _build_model_response(
        final_text=final_text,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        usage=_normalize_openai_usage(_get(raw, "usage")),
        stop_reason=_get(choices[0], "finish_reason") if choices else None,
        raw_response=raw_response,
    )


def normalize_openai_chat_completion_chunks(chunks: list[Any]) -> ModelResponse:
    """Normalize OpenAI-compatible streaming chat completion chunks."""

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, Any]] = {}
    usage: Any = None
    finish_reason: str | None = None
    raw_chunks = [_to_mapping(chunk) for chunk in chunks]

    for raw in raw_chunks:
        chunk_usage = _get(raw, "usage")
        if chunk_usage:
            usage = chunk_usage
        choices = _get(raw, "choices") or []
        if not choices:
            continue
        choice = choices[0]
        chunk_finish_reason = _get(choice, "finish_reason")
        if chunk_finish_reason:
            finish_reason = str(chunk_finish_reason)
        delta = _get(choice, "delta") or {}
        reasoning_content = _get(delta, "reasoning_content")
        if reasoning_content:
            reasoning_parts.append(str(reasoning_content))
        content = _get(delta, "content")
        if content:
            content_parts.append(str(content))
        for tool_call in _get(delta, "tool_calls") or []:
            index = _stream_tool_call_index(tool_call, len(tool_call_parts))
            merged = tool_call_parts.setdefault(
                index,
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            tool_call_id = _get(tool_call, "id")
            if tool_call_id:
                merged["id"] = str(tool_call_id)
            function = _get(tool_call, "function") or {}
            name_delta = _get(function, "name")
            if name_delta:
                merged["function"]["name"] += str(name_delta)
            arguments_delta = _get(function, "arguments")
            if arguments_delta:
                merged["function"]["arguments"] += str(arguments_delta)

    message: dict[str, Any] = {"content": "".join(content_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_call_parts:
        message["tool_calls"] = [
            tool_call_parts[index] for index in sorted(tool_call_parts)
        ]
    return normalize_openai_chat_completion(
        {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
            "stream_chunks": raw_chunks,
        }
    )


def openai_chat_completion_chunk_text_delta(chunk: Any) -> str:
    """Return the visible text delta from one OpenAI-compatible stream chunk."""

    raw = _to_mapping(chunk)
    choices = _get(raw, "choices") or []
    if not choices:
        return ""
    delta = _get(choices[0], "delta") or {}
    content = _get(delta, "content")
    return str(content) if content else ""


def openai_chat_completion_chunk_reasoning_delta(chunk: Any) -> str:
    """Return provider-exposed reasoning from one compatible stream chunk."""

    raw = _to_mapping(chunk)
    choices = _get(raw, "choices") or []
    if not choices:
        return ""
    delta = _get(choices[0], "delta") or {}
    reasoning = _get(delta, "reasoning_content")
    return str(reasoning) if reasoning else ""


def _stream_tool_call_index(tool_call: Any, fallback: int) -> int:
    index = _get(tool_call, "index")
    if index is None:
        return fallback
    try:
        return int(index)
    except (TypeError, ValueError):
        return fallback


def normalize_anthropic_message(raw_response: Any) -> ModelResponse:
    """Normalize an Anthropic Messages API response."""

    raw = _to_mapping(raw_response)
    text_parts: list[str] = []
    tool_calls: list[NormalizedToolCall] = []

    for block in _get(raw, "content") or []:
        block_type = _get(block, "type")
        if block_type == "text":
            text = _get(block, "text")
            if text:
                text_parts.append(str(text))
        elif block_type == "tool_use":
            tool_calls.append(
                NormalizedToolCall(
                    id=str(_get(block, "id") or f"tool_{len(tool_calls) + 1}"),
                    name=str(_get(block, "name")),
                    arguments=dict(_get(block, "input") or {}),
                )
            )

    usage = _get(raw, "usage")
    return _build_model_response(
        final_text="\n".join(text_parts).strip() or None,
        tool_calls=tool_calls,
        usage=ModelUsage(
            input_tokens=_get(usage, "input_tokens"),
            output_tokens=_get(usage, "output_tokens"),
            total_tokens=None,
        )
        if usage
        else None,
        stop_reason=_get(raw, "stop_reason"),
        raw_response=raw_response,
    )


def normalize_ollama_chat_response(raw_response: Any) -> ModelResponse:
    """Normalize an Ollama chat response."""

    raw = _to_mapping(raw_response)
    message = _get(raw, "message") or {}
    content = _get(message, "content")
    tool_calls = [
        _normalize_openai_tool_call(tool_call)
        for tool_call in (_get(message, "tool_calls") or [])
    ]

    return _build_model_response(
        final_text=_join_text(content),
        tool_calls=tool_calls,
        usage=ModelUsage(
            input_tokens=_get(raw, "prompt_eval_count"),
            output_tokens=_get(raw, "eval_count"),
            total_tokens=_sum_optional(
                _get(raw, "prompt_eval_count"),
                _get(raw, "eval_count"),
            ),
        ),
        stop_reason=_get(raw, "done_reason"),
        raw_response=raw_response,
    )


def _build_model_response(
    *,
    final_text: str | None,
    tool_calls: list[NormalizedToolCall],
    usage: ModelUsage | None,
    stop_reason: str | None,
    raw_response: Any,
    reasoning_content: str | None = None,
) -> ModelResponse:
    commentary = None
    normalized_final = final_text.strip() if isinstance(final_text, str) else final_text
    if tool_calls and normalized_final:
        commentary = normalized_final
        normalized_final = None
    return ModelResponse(
        final_text=normalized_final,
        tool_calls=tool_calls,
        assistant_commentary=commentary,
        reasoning_content=reasoning_content,
        usage=usage,
        stop_reason=str(stop_reason) if stop_reason is not None else None,
        raw_response=raw_response,
    ).enforce_turn_contract()


def _normalize_openai_tool_call(tool_call: Any) -> NormalizedToolCall:
    function = _get(tool_call, "function") or {}
    arguments = _get(function, "arguments") or {}
    raw_arguments: str | None = None
    argument_parse_error: str | None = None
    arguments_likely_truncated = False
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            parsed_arguments = {}
            raw_arguments = arguments
            argument_parse_error = f"{exc.msg} at char {exc.pos}"
            arguments_likely_truncated = _json_arguments_likely_truncated(arguments, exc)
    else:
        parsed_arguments = dict(arguments)

    return NormalizedToolCall(
        id=str(_get(tool_call, "id") or f"tool_{_get(tool_call, 'index') or 0}"),
        name=str(_get(function, "name")),
        arguments=parsed_arguments,
        raw_arguments=raw_arguments,
        argument_parse_error=argument_parse_error,
        arguments_likely_truncated=arguments_likely_truncated,
    )


def _json_arguments_likely_truncated(arguments: str, error: json.JSONDecodeError) -> bool:
    """Return whether malformed JSON most likely ended before the tool call completed."""

    stripped = arguments.rstrip()
    if not stripped:
        return False
    lowered = error.msg.lower()
    if "unterminated string" in lowered:
        return True
    if error.pos >= max(0, len(arguments) - 2):
        return True
    return stripped.endswith(("\\", "{", "[", ",", ":"))


_DSML_TOOL_CALLS_RE = re.compile(
    r"<[^>]*DSML[^>]*tool_calls[^>]*>.*?</[^>]*DSML[^>]*tool_calls[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_INVOKE_RE = re.compile(
    r"<[^>]*invoke\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</[^>]*invoke[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAMETER_RE = re.compile(
    r"<[^>]*parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</[^>]*parameter[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_TOOL_NAMES = {
    "read",
    "search",
    "edit",
    "write",
    "apply_patch",
    "run_command",
    "task",
}
_DSML_ARGUMENT_ALIASES = {
    "file_path": "target",
    "filepath": "target",
    "pattern": "query",
}


def _extract_dsml_tool_calls(text: str | None) -> tuple[list[NormalizedToolCall], str | None]:
    """Extract DSML-style textual tool calls emitted by some OpenAI-compatible models.

    DeepSeek-style failures can place tool invocations in ``message.content`` as
    markup instead of returning native ``tool_calls``. Treat known invocations as
    real tool calls so the agent loop does not terminate with raw markup.
    """

    if not text or "DSML" not in text or "invoke" not in text:
        return [], text

    tool_calls: list[NormalizedToolCall] = []
    for index, match in enumerate(_DSML_INVOKE_RE.finditer(text), start=1):
        raw_name = unescape(match.group(1)).strip()
        name = raw_name.lower()
        if name not in _DSML_TOOL_NAMES:
            continue
        arguments = _extract_dsml_arguments(match.group(2))
        tool_calls.append(
            NormalizedToolCall(
                id=f"dsml_call_{index}",
                name=name,
                arguments=_normalize_dsml_arguments(name, arguments),
            )
        )

    if not tool_calls:
        return [], text

    cleaned = _DSML_TOOL_CALLS_RE.sub("", text).strip() or None
    return tool_calls, cleaned


def _extract_dsml_arguments(invoke_body: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for match in _DSML_PARAMETER_RE.finditer(invoke_body):
        key = unescape(match.group(1)).strip()
        value = unescape(match.group(2)).strip()
        arguments[key] = value
    return arguments


def _normalize_dsml_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        _DSML_ARGUMENT_ALIASES.get(key, key): value
        for key, value in arguments.items()
    }
    if tool_name == "read" and "target" in normalized:
        return {
            "source": normalized.get("source", "workspace"),
            "target": normalized["target"],
        }
    if tool_name == "run_command" and "command" in normalized:
        return {"argv": _parse_dsml_command(str(normalized["command"]))}
    if tool_name == "search" and "query" in normalized:
        result = {
            "source": normalized.get("source", "workspace"),
            "kind": normalized.get("kind", "text"),
            "query": normalized["query"],
        }
        if "path" in normalized:
            result["path"] = normalized["path"]
        return result
    return normalized


def _parse_dsml_command(command: str) -> list[str]:
    """Parse a provider-emitted DSML command string into the current argv contract."""

    stripped = command.strip()
    if not stripped:
        raise ValueError("DSML command must not be empty.")
    if "\n" in stripped or "\r" in stripped or "\x00" in stripped:
        raise ValueError("DSML command must be one NUL-free line.")
    argv = shlex.split(stripped, posix=False)
    return [
        value[1:-1]
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}
        else value
        for value in argv
    ]


def _normalize_openai_usage(usage: Any) -> ModelUsage | None:
    if not usage:
        return None
    input_tokens = _get(usage, "prompt_tokens")
    cached_input_tokens = _get(usage, "prompt_cache_hit_tokens")
    if cached_input_tokens is None:
        prompt_token_details = _get(usage, "prompt_tokens_details") or {}
        cached_input_tokens = _get(prompt_token_details, "cached_tokens")
    cache_miss_input_tokens = _get(usage, "prompt_cache_miss_tokens")
    if (
        cache_miss_input_tokens is None
        and input_tokens is not None
        and cached_input_tokens is not None
    ):
        cache_miss_input_tokens = max(0, int(input_tokens) - int(cached_input_tokens))
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=_get(usage, "completion_tokens"),
        total_tokens=_get(usage, "total_tokens"),
        cached_input_tokens=cached_input_tokens,
        cache_miss_input_tokens=cache_miss_input_tokens,
    )


def _join_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = _get(part, "text")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip() or None
    return str(content)


def _sum_optional(first: Any, second: Any) -> int | None:
    if first is None and second is None:
        return None
    return int(first or 0) + int(second or 0)


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _to_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return value
