"""Provider-neutral request prepared for one model call."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    """Exact model-call payload before provider adaptation.

    ``system`` is kept separate from the conversation so Anthropic-style
    providers can pass it through their dedicated system parameter. OpenAI-
    compatible providers may prepend it as a normal system message through
    :meth:`as_chat_messages`.
    """

    system: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def required_tool_name(self) -> str | None:
        """Return one runtime-required tool name, when declared."""

        value = self.metadata.get("required_tool_name")
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    def openai_tool_choice(self) -> str | dict[str, Any]:
        """Render provider-neutral required-tool metadata for OpenAI-compatible APIs."""

        required = self.required_tool_name()
        if required is None:
            return "auto"
        return {"type": "function", "function": {"name": required}}

    def temperature(self) -> float | None:
        """Return an explicitly requested sampling temperature, when present."""

        value = self.metadata.get("temperature")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def as_chat_messages(self) -> list[dict[str, Any]]:
        """Return provider-visible messages without internal runtime metadata."""

        messages = [
            {
                key: value
                for key, value in message.items()
                if not str(key).startswith("_minicode_")
            }
            for message in self.messages
        ]
        if not self.system.strip():
            return messages
        return [{"role": "system", "content": self.system}, *messages]
