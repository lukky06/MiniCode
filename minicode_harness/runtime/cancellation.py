"""Cooperative cancellation primitives for bounded MiniCode runs."""

from __future__ import annotations

from threading import Event


class CancellationToken:
    """Thread-safe cancellation flag checked at deterministic runtime boundaries."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()
