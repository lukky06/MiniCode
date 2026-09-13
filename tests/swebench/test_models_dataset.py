from __future__ import annotations

import json
from pathlib import Path

import pytest

from minicode_harness.swebench import SweBenchBudget, SweBenchInstance, load_instances


def _record(instance_id: str) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": "owner/repo",
        "base_commit": "abc123",
        "problem_statement": f"Fix {instance_id}",
        "version": "1.0",
        "patch": "gold patch",
        "test_patch": "gold tests",
        "FAIL_TO_PASS": ["test_a"],
        "PASS_TO_PASS": ["test_b"],
    }


def test_swebench_budget_uses_bounded_default_steps() -> None:
    budget = SweBenchBudget()

    assert budget.max_steps == 40
    assert budget.max_tool_calls == 120


def test_agent_instance_ignores_gold_fields() -> None:
    instance = SweBenchInstance.model_validate(_record("repo__1"))

    payload = instance.model_dump()

    assert payload == {
        "instance_id": "repo__1",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix repo__1",
        "version": "1.0",
    }
    assert "patch" not in payload
    assert "FAIL_TO_PASS" not in payload


def test_jsonl_loader_sorts_filters_and_shards(tmp_path: Path) -> None:
    dataset = tmp_path / "instances.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(_record(instance_id))
            for instance_id in ["repo__3", "repo__1", "repo__2", "repo__4"]
        )
        + "\n",
        encoding="utf-8",
    )

    selected = load_instances(
        dataset,
        instance_ids=["repo__4", "repo__2"],
    )
    shard = load_instances(dataset, shard_id=1, num_shards=2)

    assert [item.instance_id for item in selected] == ["repo__2", "repo__4"]
    assert [item.instance_id for item in shard] == ["repo__2", "repo__4"]


def test_loader_reports_missing_requested_instance(tmp_path: Path) -> None:
    dataset = tmp_path / "instances.json"
    dataset.write_text(json.dumps([_record("repo__1")]), encoding="utf-8")

    with pytest.raises(ValueError, match="were not found"):
        load_instances(dataset, instance_ids=["repo__missing"])


def test_missing_local_json_is_not_treated_as_huggingface(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_instances(tmp_path / "missing.jsonl")
