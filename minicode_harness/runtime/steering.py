"""Minimal thread-safe runtime steering queue.

The queue is intentionally transport-agnostic. Producers may enqueue text
while a Run is executing; the owning AgentLoop decides when a message is safe
to append to canonical provider-native history.
"""

from __future__ import annotations

from collections import deque
from threading import Lock


class SteeringQueue:
    """A small FIFO queue for text supplied during an active Run."""

    def __init__(self) -> None:
        self._messages: deque[str] = deque()
        self._lock = Lock()

    def enqueue(self, text: str) -> None:
        """Append one non-empty steering message in FIFO order."""

        if not isinstance(text, str):
            raise TypeError("Steering message must be text.")
        message = text.strip()
        if not message:
            raise ValueError("Steering message must not be empty.")
        with self._lock:
            self._messages.append(message)

    def dequeue(self) -> str | None:
        """Remove and return the oldest message, or ``None`` when empty."""

        with self._lock:
            if not self._messages:
                return None
            return self._messages.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)
