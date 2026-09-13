"""Local SWE-bench dataset loading with deterministic selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .models import SweBenchInstance


def load_instances(
    dataset: Path | str,
    *,
    instance_ids: Iterable[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    split: str = "test",
    revision: str | None = None,
    cache_dir: Path | str | None = None,
) -> list[SweBenchInstance]:
    """Load local JSON/JSONL or an optional Hugging Face dataset."""

    dataset_value = str(dataset)
    candidate = Path(dataset_value).expanduser()
    path = candidate.resolve()
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    _validate_shard(shard_id, num_shards)

    if path.is_file():
        payloads = _read_payloads(path)
        source_label = str(path)
    else:
        if candidate.suffix.lower() in {".json", ".jsonl"}:
            raise FileNotFoundError(f"SWE-bench dataset does not exist: {path}")
        payloads = _read_huggingface_payloads(
            dataset_value,
            split=split,
            revision=revision,
            cache_dir=cache_dir,
        )
        source_label = f"{dataset_value}[{split}]"
    instances: list[SweBenchInstance] = []
    seen: set[str] = set()
    for index, payload in enumerate(payloads, start=1):
        try:
            instance = SweBenchInstance.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid SWE-bench instance at {source_label}:{index}: {exc}"
            ) from exc
        if instance.instance_id in seen:
            raise ValueError(f"Duplicate SWE-bench instance_id: {instance.instance_id}")
        seen.add(instance.instance_id)
        instances.append(instance)

    selected_ids = {
        value.strip()
        for value in (instance_ids or [])
        if value and value.strip()
    }
    if selected_ids:
        instances = [
            instance for instance in instances if instance.instance_id in selected_ids
        ]
        missing = selected_ids - {instance.instance_id for instance in instances}
        if missing:
            raise ValueError(
                "Requested SWE-bench instances were not found: "
                + ", ".join(sorted(missing))
            )

    instances.sort(key=lambda item: item.instance_id)
    if shard_id is not None and num_shards is not None:
        instances = instances[shard_id::num_shards]
    instances = instances[offset:]
    if limit is not None:
        instances = instances[:limit]
    return instances


def _read_huggingface_payloads(
    dataset_name: str,
    *,
    split: str,
    revision: str | None,
    cache_dir: Path | str | None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset loading requires the optional 'swebench' dependencies"
        ) from exc
    loaded = load_dataset(
        dataset_name,
        split=split,
        revision=revision,
        cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
    )
    return [dict(item) for item in loaded]


def _read_payloads(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        payloads: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"SWE-bench JSONL row must be an object: {path}:{line_number}"
                )
            payloads.append(payload)
        return payloads

    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON dataset {path}: {exc.msg}") from exc
        if isinstance(payload, dict):
            if isinstance(payload.get("instances"), list):
                payload = payload["instances"]
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("SWE-bench JSON dataset must be an object or array")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("Every SWE-bench JSON instance must be an object")
        return list(payload)

    raise ValueError("SWE-bench dataset must use .json or .jsonl")


def _validate_shard(shard_id: int | None, num_shards: int | None) -> None:
    if shard_id is None and num_shards is None:
        return
    if shard_id is None or num_shards is None:
        raise ValueError("shard_id and num_shards must be provided together")
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
