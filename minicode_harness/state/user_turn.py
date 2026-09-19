"""State owned by one user turn across multiple model calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field


class UserTurnState(BaseModel):
    """Canonical message history and call counter for one user request."""

    turn_id: str
    task: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    model_call_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        turn_id: str,
        task: str,
        messages: list[dict[str, Any]],
        model_call_count: int = 0,
        append_task: bool = True,
    ) -> "UserTurnState":
        """Create a turn from persistent conversation history.

        A fresh REPL turn appends the new user task to the existing session
        history. Resume passes ``append_task=False`` because the checkpoint
        already contains the task and its completed tool exchanges.
        """

        history = deepcopy(messages)
        user_message = {"role": "user", "content": task}
        if append_task:
            history.append(deepcopy(user_message))
        elif not history:
            history.append(deepcopy(user_message))
        return cls(
            turn_id=turn_id,
            task=task,
            messages=history,
            model_call_count=model_call_count,
        )

    def begin_model_call(self) -> int:
        """Advance and return the one-based model-call index for this turn."""

        self.model_call_count += 1
        return self.model_call_count

    def snapshot_messages(self) -> list[dict[str, Any]]:
        """Return an isolated copy suitable for one model request."""

        return deepcopy(self.messages)

    def append_message(self, message: dict[str, Any]) -> None:
        """Append one message to the single canonical protocol history."""

        self.messages.append(deepcopy(message))

    def append_final_text(self, content: str) -> None:
        """Append an accepted assistant answer to the canonical history."""

        if content.strip():
            self.append_message({"role": "assistant", "content": content.strip()})
