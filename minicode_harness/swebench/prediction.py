"""Atomic persistence for SWE-bench predictions and result artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from .models import SweBenchPrediction


def atomic_write_text(path: Path | str, content: str) -> Path:
    """Write text through a sibling temporary file and atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def atomic_write_json(
    path: Path | str,
    payload: BaseModel | dict[str, Any] | list[Any],
) -> Path:
    """Atomically write formatted UTF-8 JSON."""

    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )


def write_prediction(path: Path | str, prediction: SweBenchPrediction) -> Path:
    """Write one standalone prediction JSON file."""

    return atomic_write_json(path, prediction)


def write_predictions_jsonl(
    path: Path | str,
    predictions: Iterable[SweBenchPrediction],
) -> Path:
    """Atomically write official one-record-per-line predictions JSONL."""

    lines = [
        json.dumps(
            prediction.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for prediction in predictions
    ]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    return atomic_write_text(path, content)


def load_prediction(path: Path | str) -> SweBenchPrediction:
    """Load one persisted prediction file."""

    target = Path(path)
    return SweBenchPrediction.model_validate_json(target.read_text(encoding="utf-8"))
