"""Thread-safe JSONL event writer for the TypeScript TUI backend."""

from __future__ import annotations

import sys
from threading import Lock
from typing import TextIO

from pydantic import BaseModel

from .protocol import encode_message


class JsonlEventWriter:
    """Write complete protocol records atomically to one text stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self._lock = Lock()
        if stream is None:
            reconfigure = getattr(self.stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="strict")

    def emit(self, message: BaseModel) -> None:
        record = encode_message(message)
        with self._lock:
            self.stream.write(record)
            self.stream.flush()
