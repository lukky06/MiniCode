"""Structured UserInputClient backed by JSONL request/response messages."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock

from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.state import (
    UserInputClient,
    UserInputRequest,
    UserInputResponse,
)

from .output import JsonlOutputSink
from .protocol import UserInputRequired


@dataclass
class _PendingUserInput:
    request: UserInputRequest
    completed: Event
    response: UserInputResponse | None = None


class JsonlUserInputClient(UserInputClient):
    """Block the Python Run until the TypeScript TUI selects one bounded option."""

    def __init__(self, output_sink: JsonlOutputSink) -> None:
        self.output_sink = output_sink
        self._choice_lock = Lock()
        self._pending_lock = Lock()
        self._pending: _PendingUserInput | None = None
        self._cancellation_token: CancellationToken | None = None

    def bind_cancellation(self, token: CancellationToken | None) -> None:
        self._cancellation_token = token

    def choose(self, request: UserInputRequest) -> UserInputResponse:
        with self._choice_lock:
            if self._is_cancelled():
                raise RuntimeError("Run was cancelled before user input.")

            pending = _PendingUserInput(request=request, completed=Event())
            with self._pending_lock:
                self._pending = pending

            self.output_sink.writer.emit(
                UserInputRequired(
                    id=request.id,
                    question=request.question,
                    options=[
                        {
                            "label": option.label,
                            "description": option.description,
                        }
                        for option in request.options
                    ],
                )
            )

            while not pending.completed.wait(0.05):
                if self._is_cancelled():
                    with self._pending_lock:
                        if self._pending is pending and not pending.completed.is_set():
                            pending.completed.set()

            with self._pending_lock:
                if self._pending is pending:
                    self._pending = None

            if pending.response is None:
                raise RuntimeError("User input ended without a selection.")
            return pending.response

    def resolve(
        self,
        request_id: str,
        selected_index: int,
    ) -> tuple[bool, str | None]:
        with self._pending_lock:
            pending = self._pending
            if pending is None:
                return False, "No user input request is currently pending."
            if pending.request.id != request_id:
                return False, f"Stale user input response: {request_id}."
            if pending.completed.is_set():
                return False, f"User input is already resolved: {request_id}."
            if selected_index < 0 or selected_index >= len(pending.request.options):
                return False, f"User input selection is outside the option range: {selected_index}."
            pending.response = UserInputResponse(selected_index=selected_index)
            pending.completed.set()
            return True, None

    def has_pending(self) -> bool:
        with self._pending_lock:
            return self._pending is not None and not self._pending.completed.is_set()

    def _is_cancelled(self) -> bool:
        return bool(
            self._cancellation_token is not None
            and self._cancellation_token.is_cancelled
        )
