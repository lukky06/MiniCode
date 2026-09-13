from pathlib import Path

import pytest

from minicode_harness.workspace import WorkspaceAccessError, WorkspaceGuard


def test_workspace_guard_resolves_paths_inside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    source_file = workspace / "src" / "Main.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("class Main {}\n", encoding="utf-8")

    guard = WorkspaceGuard(workspace)

    assert guard.resolve("src/Main.java") == source_file.resolve()
    assert guard.relative_path(source_file) == "src/Main.java"


def test_workspace_guard_rejects_path_traversal(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret\n", encoding="utf-8")

    guard = WorkspaceGuard(workspace)

    with pytest.raises(WorkspaceAccessError, match="escapes workspace"):
        guard.resolve("../outside.txt")

    with pytest.raises(WorkspaceAccessError, match="escapes workspace"):
        guard.resolve(outside_file)


def test_workspace_guard_rejects_sensitive_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sensitive_paths = [
        ".env",
        "id_rsa",
        "cert.pem",
        "secret.key",
        "keystore.p12",
        "truststore.jks",
        ".git/config",
        ".mini-code/memory/project_memory.md",
        "runs/run_20260630_001/session.json",
    ]
    for relative_path in sensitive_paths:
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sensitive\n", encoding="utf-8")

    guard = WorkspaceGuard(workspace)

    for relative_path in sensitive_paths:
        with pytest.raises(WorkspaceAccessError, match="sensitive path"):
            guard.resolve(relative_path)


def test_workspace_guard_rejects_missing_workspace(tmp_path) -> None:
    with pytest.raises(WorkspaceAccessError, match="does not exist"):
        WorkspaceGuard(tmp_path / "missing")
