from __future__ import annotations

import json
from pathlib import Path

from minicode_harness.swebench import (
    PatchExporter,
    SweBenchPrediction,
    write_predictions_jsonl,
)

from .helpers import git, init_git_repo


def test_patch_export_includes_new_files_and_excludes_run_artifacts(
    tmp_path: Path,
) -> None:
    workspace, base_commit = init_git_repo(
        tmp_path / "repo",
        {"app.py": "value = 1\n"},
    )
    (workspace / "app.py").write_text("value = 2\n", encoding="utf-8")
    (workspace / "new_test.py").write_text("def test_value():\n    assert 2 == 2\n", encoding="utf-8")
    (workspace / "runs").mkdir()
    (workspace / "runs" / "trace.jsonl").write_text("noise\n", encoding="utf-8")

    result = PatchExporter().export(
        workspace,
        output_path=tmp_path / "final.patch",
        expected_head=base_commit,
    )

    assert result.status == "ok"
    assert "diff --git a/app.py b/app.py" in result.patch
    assert "diff --git a/new_test.py b/new_test.py" in result.patch
    assert "runs/trace.jsonl" not in result.patch
    assert result.changed_files == ["app.py", "new_test.py"]
    assert (tmp_path / "final.patch").read_text(encoding="utf-8") == result.patch


def test_empty_workspace_exports_no_patch(tmp_path: Path) -> None:
    workspace, base_commit = init_git_repo(tmp_path / "repo")

    result = PatchExporter().export(workspace, expected_head=base_commit)

    assert result.status == "no_patch"
    assert result.patch == ""
    assert result.changed_files == []


def test_patch_export_handles_rename_delete_binary_and_crlf(tmp_path: Path) -> None:
    workspace, _ = init_git_repo(
        tmp_path / "repo",
        {
            "old.py": "value = 1\n",
            "delete.txt": "remove me\n",
            "windows.txt": "line1\r\nline2\r\n",
        },
    )
    (workspace / "data.bin").write_bytes(b"\x00old\xff")
    git(workspace, "add", "data.bin")
    git(workspace, "commit", "--amend", "--no-edit")

    git(workspace, "mv", "old.py", "renamed.py")
    (workspace / "delete.txt").unlink()
    (workspace / "data.bin").write_bytes(b"\x00new\xff")
    (workspace / "windows.txt").write_bytes(b"line1\r\nchanged\r\n")

    result = PatchExporter().export(workspace)

    assert result.status == "ok"
    assert "rename from old.py" in result.patch
    assert "rename to renamed.py" in result.patch
    assert "deleted file mode" in result.patch
    assert "GIT binary patch" in result.patch
    assert set(result.changed_files) == {
        "data.bin",
        "delete.txt",
        "renamed.py",
        "windows.txt",
    }


def test_non_utf8_text_patch_is_rejected_without_corruption(tmp_path: Path) -> None:
    workspace, _ = init_git_repo(tmp_path / "repo")
    (workspace / "latin.txt").write_bytes(b"\xffold\n")
    git(workspace, "add", "latin.txt")
    git(workspace, "commit", "--amend", "--no-edit")
    (workspace / "latin.txt").write_bytes(b"\xffnew\n")

    result = PatchExporter().export(workspace)

    assert result.status == "invalid"
    assert "not valid UTF-8" in (result.validation_error or "")


def test_prediction_jsonl_is_valid_and_stable(tmp_path: Path) -> None:
    path = write_predictions_jsonl(
        tmp_path / "predictions.jsonl",
        [
            SweBenchPrediction(
                instance_id="b",
                model_name_or_path="model",
                model_patch="patch-b",
            ),
            SweBenchPrediction(
                instance_id="a",
                model_name_or_path="model",
                model_patch="patch-a",
            ),
        ],
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [row["instance_id"] for row in rows] == ["b", "a"]
    assert all(set(row) == {"instance_id", "model_name_or_path", "model_patch"} for row in rows)
