"""Rules-based observation compression."""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from pydantic import BaseModel

from minicode_harness.tools.semantics import read_source, search_kind, search_source

from .limits import (
    COMMAND_OUTPUT_CHAR_LIMIT,
    FAILED_COMMAND_OUTPUT_CHAR_LIMIT,
    FAILED_COMMAND_PREVIEW_CHAR_LIMIT,
    FILE_SEARCH_RESULT_LIMIT,
    IMPORTANT_FAILURE_LINE_LIMIT,
    OBSERVATION_PREVIEW_CHARS,
    OBSERVATION_TOKEN_LIMIT,
    RAW_TEXT_SUMMARY_CHARS,
    READ_FILE_LINE_LIMIT,
    SEARCH_MATCH_LIMIT,
)
from .token import estimate_tokens
from .types import ContextCompressionEvent, ContextObservation


def build_observation(
    *,
    tool_call_id: str,
    tool_name: str,
    result: Any,
    artifact_dir: Path | None = None,
    tool_arguments: dict[str, Any] | None = None,
) -> tuple[ContextObservation, list[ContextCompressionEvent]]:
    """Convert a tool result into a context observation and compression events."""

    arguments = tool_arguments or {}
    semantic_tool_name = _semantic_tool_name(tool_name, arguments)
    semantic_arguments = arguments
    payload = _payload_for_result(result)
    raw_text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    raw_tokens = estimate_tokens(raw_text)
    metadata = _metadata_for_result(semantic_tool_name, payload)
    source = read_source(tool_name, arguments)
    kind = search_kind(tool_name, arguments)
    search_root = search_source(tool_name, arguments)
    if source is not None:
        metadata["read_source"] = source
    if kind is not None:
        metadata["search_kind"] = kind
    if search_root is not None:
        metadata["search_source"] = search_root
    if isinstance(payload, dict) and isinstance(payload.get("status"), str):
        metadata.setdefault("status", payload["status"])
    _attach_tool_input_context(
        metadata=metadata,
        tool_name=semantic_tool_name,
        tool_arguments=semantic_arguments,
        artifact_dir=artifact_dir,
        tool_call_id=tool_call_id,
    )
    reasons = _compression_reasons(semantic_tool_name, payload, raw_text, raw_tokens)
    summary = _summary_for_result(semantic_tool_name, payload, reasons)
    is_important = _is_important_observation(semantic_tool_name, payload, reasons)
    evidence_refs = [f"tool_call:{tool_call_id}"]
    lossiness = "excerpted" if reasons else "none"

    artifact_path: str | None = None
    compressed_text = raw_text
    if reasons:
        if semantic_tool_name == "artifact_read":
            artifact_path = _existing_artifact_path(semantic_arguments, payload)
        else:
            artifact_path = _write_artifact(
                artifact_dir=artifact_dir,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=_artifact_content(semantic_tool_name, payload, raw_text),
            )
        compressed_text = _structured_compressed_observation(
            summary=summary,
            reasons=reasons,
            artifact_path=artifact_path,
            evidence_refs=evidence_refs,
            compressed_payload=_compress_payload(semantic_tool_name, payload, raw_text),
        )
    metadata.update(
        {
            "lossiness": lossiness,
            "source_refs": evidence_refs,
            "artifact_path": artifact_path,
        }
    )

    after_tokens = estimate_tokens(compressed_text)
    event = (
        ContextCompressionEvent(
            reason="tool_output",
            before_tokens=raw_tokens,
            after_tokens=after_tokens,
            details={
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "reasons": reasons,
                "artifact_path": artifact_path,
            },
        )
        if reasons
        else None
    )
    observation = ContextObservation(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=compressed_text,
        output_preview=compressed_text[:OBSERVATION_PREVIEW_CHARS],
        token_estimate=after_tokens,
        summary=summary,
        is_important=is_important,
        is_truncated=bool(reasons),
        artifact_path=artifact_path,
        lossiness=lossiness,
        evidence_refs=evidence_refs,
        metadata=metadata,
    )
    return observation, [event] if event else []


def _payload_for_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    return result


def _semantic_tool_name(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "read":
        return {
            "workspace": "workspace_read",
            "artifact": "artifact_read",
            "memory": "memory_read",
            "skill": "skill_read",
            "diff": "diff_read",
        }.get(read_source(tool_name, arguments) or "", tool_name)
    if tool_name == "search":
        return {
            "files": "file_search",
            "text": "text_search",
        }.get(search_kind(tool_name, arguments) or "", tool_name)
    return tool_name


def _attach_tool_input_context(
    *,
    metadata: dict[str, Any],
    tool_name: str,
    tool_arguments: dict[str, Any],
    artifact_dir: Path | None,
    tool_call_id: str,
) -> None:
    """Retain the latest write input so verification failures can be repaired without rereading."""

    if tool_name == "write":
        content = str(tool_arguments.get("content") or "")
        field = "written_content"
        artifact_tool_name = "write_input"
    elif tool_name == "apply_patch":
        content = str(tool_arguments.get("patch") or "")
        field = "applied_patch"
        artifact_tool_name = "apply_patch_input"
    else:
        return
    if not content:
        return
    if len(content) <= 12_000:
        metadata[field] = content
        metadata["input_content_status"] = "full"
        return
    artifact_path = _write_artifact(
        artifact_dir=artifact_dir,
        tool_call_id=tool_call_id,
        tool_name=artifact_tool_name,
        content=content,
    )
    metadata["input_content_status"] = "artifact_with_preview"
    metadata["input_artifact_path"] = artifact_path
    metadata["input_content_preview"] = _head_tail_lines(content.splitlines(), head=80, tail=40)


def _metadata_for_result(tool_name: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if tool_name == "file_search":
        return {
            "root": payload.get("root"),
            "pattern": payload.get("pattern"),
            "files": payload.get("files") or [],
            "file_details": payload.get("file_details") or [],
            "truncated": payload.get("truncated", False),
            "truncation_reason": payload.get("truncation_reason"),
            "scanned_entries": payload.get("scanned_entries", 0),
            "exists": payload.get("exists", True),
            "is_directory": payload.get("is_directory", True),
        }
    if tool_name in {"artifact_read", "workspace_read"}:
        return {
            "path": payload.get("path"),
            "total_lines": payload.get("total_lines"),
            "start_line": payload.get("start_line"),
            "end_line": payload.get("end_line"),
            "returned_lines": payload.get("returned_lines", 0),
            "returned_chars": payload.get("returned_chars", 0),
            "total_chars": payload.get("total_chars", 0),
            "full_resource_read": payload.get("full_resource_read", False),
            "guidance": payload.get("guidance"),
        }
    if tool_name == "text_search":
        matches = payload.get("matches") or []
        return {
            "query": payload.get("query"),
            "match_count": len(matches),
            "files": _unique_paths_from_matches(matches),
            "truncated": payload.get("truncated", False),
            "truncation_reason": payload.get("truncation_reason"),
            "scanned_entries": payload.get("scanned_entries", 0),
        }
    if tool_name == "run_command":
        return {
            "argv": payload.get("argv") or [],
            "command": payload.get("command"),
            "returncode": payload.get("returncode"),
            "timed_out": payload.get("timed_out", False),
            "cancelled": payload.get("cancelled", False),
        }
    if tool_name == "diff_read":
        diff_files = _changed_files_from_diff(str(payload.get("diff", "")))
        untracked_files = [str(path) for path in payload.get("untracked_files") or []]
        return {
            "command": payload.get("command"),
            "returncode": payload.get("returncode"),
            "files": _dedupe_preserve_order(diff_files + untracked_files),
            "untracked_files": untracked_files,
        }
    if tool_name == "apply_patch":
        return {"files": payload.get("files") or []}
    if tool_name == "write":
        return {
            "path": payload.get("path"),
            "created": payload.get("created", False),
            "overwritten": payload.get("overwritten", False),
            "changed": payload.get("changed", True),
            "bytes_written": payload.get("bytes_written", 0),
        }
    return {}


def _summary_for_result(tool_name: str, payload: Any, reasons: list[str]) -> str:
    if tool_name == "file_search" and isinstance(payload, dict):
        files = payload.get("files") or []
        root = payload.get("root", ".")
        pattern = payload.get("pattern", "**/*")
        exists = bool(payload.get("exists", True))
        is_directory = bool(payload.get("is_directory", True))
        if not exists:
            return f"Found 0 file(s) under {root} matching {pattern!r}; path_exists=false."
        if not is_directory:
            return f"Found 0 file(s) under {root} matching {pattern!r}; is_directory=false."
        sample = _join_sample([str(path) for path in files], limit=5)
        suffix = " Output was truncated." if payload.get("truncated") else ""
        if sample:
            return (
                f"Found {len(files)} file(s) under {root} matching {pattern!r}: "
                f"{sample}.{suffix}"
            )
        return f"Found 0 files under {root} matching {pattern!r}."

    if tool_name in {"artifact_read", "workspace_read"} and isinstance(payload, dict):
        path = payload.get("path", "<unknown>")
        start = payload.get("start_line")
        end = payload.get("end_line")
        total = payload.get("total_lines")
        suffix = " Full output stored as an artifact." if reasons else ""
        cost = (
            f" Raw resource cost: {payload.get('total_lines', 0)} lines, "
            f"{payload.get('total_chars', 0)} chars."
            if payload.get("guidance")
            else ""
        )
        guidance = f" {payload.get('guidance')}" if payload.get("guidance") else ""
        source = "artifact" if tool_name == "artifact_read" else "file"
        return f"Read {source} {path} lines {start}-{end} of {total}.{suffix}{cost}{guidance}"

    if tool_name == "text_search" and isinstance(payload, dict):
        matches = payload.get("matches") or []
        files = _unique_paths_from_matches(matches)
        sample = _join_sample(files, limit=5)
        suffix = " Results were truncated." if payload.get("truncated") else ""
        if sample:
            return (
                f"Search for {payload.get('query')!r} returned {len(matches)} match(es) "
                f"in {sample}.{suffix}"
            )
        return f"Search for {payload.get('query')!r} returned no matches."

    if tool_name == "diff_read" and isinstance(payload, dict):
        diff_files = _changed_files_from_diff(str(payload.get("diff", "")))
        untracked_files = [str(path) for path in payload.get("untracked_files") or []]
        if diff_files and untracked_files:
            return (
                f"Git diff has changes in {_join_sample(diff_files, limit=6)}; "
                f"untracked files: {_join_sample(untracked_files, limit=6)}."
            )
        if diff_files:
            return f"Git diff has changes in {_join_sample(diff_files, limit=8)}."
        if untracked_files:
            return f"Git status shows untracked files: {_join_sample(untracked_files, limit=8)}."
        if payload.get("returncode") not in (0, None):
            return f"Git inspection failed with exit code {payload.get('returncode')}."
        return "Git diff inspected; no tracked or untracked changes detected."

    if tool_name == "apply_patch" and isinstance(payload, dict):
        files = [str(path) for path in payload.get("files") or []]
        return f"Applied patch to {_join_sample(files, limit=8) or 'no reported files'}."

    if tool_name == "write" and isinstance(payload, dict):
        path = payload.get("path", "<unknown>")
        if payload.get("changed", True) is False:
            return f"write made no content change to {path}; requested content already matched the file."
        action = "created" if payload.get("created") else "overwritten"
        return f"Wrote {path} ({action}, {payload.get('bytes_written', 0)} bytes)."

    if tool_name == "run_command" and isinstance(payload, dict):
        runtime_task_id = payload.get("runtime_task_id")
        if runtime_task_id and payload.get("status") == "running":
            return f"Background command started as {runtime_task_id}."
        command = payload.get("command", "<unknown>")
        returncode = payload.get("returncode")
        output = str(payload.get("stderr") or payload.get("stdout") or "")
        evidence = _first_actionable_failure_line(output) if returncode not in (0, None) else _first_non_empty_line(output)
        duration = payload.get("duration_seconds")
        duration_text = (
            f" after {float(duration):.2f}s"
            if isinstance(duration, (int, float))
            else ""
        )
        if payload.get("cancelled"):
            return f"Command {command!r} was cancelled{duration_text}."
        if payload.get("timed_out"):
            return f"Command {command!r} timed out{duration_text}."
        label = " Primary failure" if returncode not in (0, None) else " First output line"
        suffix = f"{label}: {evidence}" if evidence else ""
        return f"Command {command!r} exited {returncode}{duration_text}.{suffix}"

    return f"{tool_name} returned a result."


def _is_important_observation(tool_name: str, payload: Any, reasons: list[str]) -> bool:
    if reasons:
        return True
    if tool_name == "artifact_read":
        return True
    if tool_name in {"apply_patch", "write", "run_command"}:
        return True
    if tool_name == "text_search" and isinstance(payload, dict):
        return bool(payload.get("matches"))
    if tool_name == "diff_read" and isinstance(payload, dict):
        return bool(str(payload.get("diff", "")).strip()) or payload.get("returncode") not in (0, None)
    return False


def _compression_reasons(
    tool_name: str,
    payload: Any,
    raw_text: str,
    token_estimate: int,
) -> list[str]:
    reasons: list[str] = []
    if tool_name in {"artifact_read", "workspace_read"} and isinstance(payload, dict):
        content = str(payload.get("content") or "")
        returned_line_count = len(content.splitlines())
        if returned_line_count > READ_FILE_LINE_LIMIT:
            reasons.append(
                f"{tool_name}_over_{READ_FILE_LINE_LIMIT}_returned_lines"
            )
    if tool_name == "file_search" and isinstance(payload, dict):
        if len(payload.get("files") or []) > FILE_SEARCH_RESULT_LIMIT:
            reasons.append(f"file_search_over_{FILE_SEARCH_RESULT_LIMIT}_files")
    if tool_name == "text_search" and isinstance(payload, dict):
        if len(payload.get("matches") or []) > SEARCH_MATCH_LIMIT:
            reasons.append(f"text_search_over_{SEARCH_MATCH_LIMIT}_matches")
    if tool_name == "run_command":
        failed = (
            isinstance(payload, dict)
            and payload.get("returncode") not in (0, None)
        )
        if failed and len(raw_text) > FAILED_COMMAND_OUTPUT_CHAR_LIMIT:
            reasons.append(
                f"failed_run_command_over_{FAILED_COMMAND_OUTPUT_CHAR_LIMIT}_chars"
            )
        elif len(raw_text) > COMMAND_OUTPUT_CHAR_LIMIT:
            reasons.append(f"run_command_over_{COMMAND_OUTPUT_CHAR_LIMIT}_chars")
    if token_estimate > OBSERVATION_TOKEN_LIMIT:
        reasons.append(f"observation_over_{OBSERVATION_TOKEN_LIMIT}_tokens")
    return reasons


def _artifact_content(tool_name: str, payload: Any, raw_text: str) -> str:
    """Return artifact bytes with source-line semantics for text reads."""

    if tool_name in {"artifact_read", "workspace_read"} and isinstance(payload, dict):
        return str(payload.get("content") or "")
    return raw_text


def _existing_artifact_path(
    tool_arguments: dict[str, Any],
    payload: Any,
) -> str | None:
    """Reuse the source Artifact path instead of creating Artifact chains."""

    candidate = tool_arguments.get("target")
    if candidate in (None, "") and isinstance(payload, dict):
        candidate = payload.get("path")
    normalized = str(candidate or "").strip().replace("\\", "/")
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _compress_text(raw_text: str) -> str:
    if len(raw_text) <= 4000:
        return raw_text
    head = raw_text[:RAW_TEXT_SUMMARY_CHARS]
    tail = raw_text[-int(RAW_TEXT_SUMMARY_CHARS / 2):]
    return f"{head}\n...<context output compressed>...\n{tail}"


def _compress_payload(tool_name: str, payload: Any, raw_text: str) -> str:
    if isinstance(payload, dict):
        if tool_name in {"artifact_read", "workspace_read"}:
            return _compress_read_payload(payload)
        if tool_name == "file_search":
            return _compress_file_search_payload(payload)
        if tool_name == "text_search":
            return _compress_search_payload(payload)
        if tool_name == "run_command":
            return _compress_command_payload(payload)
        if tool_name == "diff_read":
            return _compress_diff_payload(payload)
    return _compress_text(raw_text)


def _structured_compressed_observation(
    *,
    summary: str,
    reasons: list[str],
    artifact_path: str | None,
    evidence_refs: list[str],
    compressed_payload: str,
) -> str:
    try:
        payload: Any = json.loads(compressed_payload)
    except json.JSONDecodeError:
        payload = compressed_payload
    compact = {
        "summary": summary,
        "lossiness": "excerpted",
        "compression_reasons": reasons,
        "evidence_refs": evidence_refs,
        "artifact_path": artifact_path,
        "output_preview": payload,
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compress_read_payload(payload: dict[str, Any]) -> str:
    content = str(payload.get("content", ""))
    lines = content.splitlines()
    compact = {
        "compression": "context output compressed",
        "path": payload.get("path"),
        "start_line": payload.get("start_line"),
        "end_line": payload.get("end_line"),
        "total_lines": payload.get("total_lines"),
        "returned_lines": payload.get("returned_lines", 0),
        "returned_chars": payload.get("returned_chars", 0),
        "total_chars": payload.get("total_chars", 0),
        "full_resource_read": payload.get("full_resource_read", False),
        "guidance": payload.get("guidance"),
        "content_preview": _head_tail_lines(lines, head=120, tail=80),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compress_file_search_payload(payload: dict[str, Any]) -> str:
    files = [str(path) for path in payload.get("files") or []]
    compact = {
        "compression": "context output compressed",
        "root": payload.get("root", "."),
        "pattern": payload.get("pattern", "**/*"),
        "match_count": len(files),
        "files": files[:20],
        "omitted_files": max(0, len(files) - 20),
        "truncated": payload.get("truncated", False),
        "truncation_reason": payload.get("truncation_reason"),
        "scanned_entries": payload.get("scanned_entries", 0),
        "exists": payload.get("exists", True),
        "is_directory": payload.get("is_directory", True),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compress_search_payload(payload: dict[str, Any]) -> str:
    matches = payload.get("matches") or []
    by_file: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        if not isinstance(match, dict):
            continue
        path = str(match.get("path", "<unknown>"))
        bucket = by_file.setdefault(path, [])
        if len(bucket) < 3:
            bucket.append(
                {
                    "line": match.get("line"),
                    "text": _truncate_line(str(match.get("text", "")), limit=220),
                }
            )
    compact_files = [
        {
            "path": path,
            "sample_matches": bucket,
            "omitted_matches_for_file": max(
                0,
                sum(1 for match in matches if isinstance(match, dict) and match.get("path") == path)
                - len(bucket),
            ),
        }
        for path, bucket in list(by_file.items())[:12]
    ]
    compact = {
        "compression": "context output compressed",
        "query": payload.get("query"),
        "match_count": len(matches),
        "truncated": payload.get("truncated", False),
        "truncation_reason": payload.get("truncation_reason"),
        "scanned_entries": payload.get("scanned_entries", 0),
        "files": compact_files,
        "omitted_files": max(0, len(by_file) - len(compact_files)),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compress_command_payload(payload: dict[str, Any]) -> str:
    failed = payload.get("returncode") not in (0, None)
    preview = _failed_command_output_preview if failed else _important_output_preview
    compact = {
        "compression": "context output compressed",
        "argv": payload.get("argv") or [],
        "command": payload.get("command"),
        "returncode": payload.get("returncode"),
        "timed_out": payload.get("timed_out", False),
        "duration_seconds": payload.get("duration_seconds"),
        "stderr_preview": preview(str(payload.get("stderr", ""))),
        "stdout_preview": preview(str(payload.get("stdout", ""))),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compress_diff_payload(payload: dict[str, Any]) -> str:
    diff = str(payload.get("diff", ""))
    diff_files = _changed_files_from_diff(diff)
    untracked_files = [str(path) for path in payload.get("untracked_files") or []]
    compact = {
        "compression": "context output compressed",
        "command": payload.get("command"),
        "returncode": payload.get("returncode"),
        "files": _dedupe_preserve_order(diff_files + untracked_files),
        "untracked_files": untracked_files,
        "status_short": payload.get("status_short", ""),
        "diff_preview": _diff_preview(diff),
        "stderr": payload.get("stderr", ""),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _unique_paths_from_matches(matches: list[Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in matches:
        path = match.get("path") if isinstance(match, dict) else None
        if path and path not in seen:
            paths.append(str(path))
            seen.add(str(path))
    return paths


def _changed_files_from_diff(diff: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path and path not in seen:
            files.append(path)
            seen.add(path)
    return files


def _head_tail_lines(lines: list[str], *, head: int, tail: int) -> str:
    if len(lines) <= head + tail:
        return "\n".join(lines)
    omitted = len(lines) - head - tail
    return "\n".join(
        lines[:head]
        + [f"...<context output compressed: {omitted} lines omitted>..."]
        + lines[-tail:]
    )


def _failed_command_output_preview(text: str) -> str:
    """Keep the primary diagnostic and tail summary within one bounded preview."""

    lines = text.splitlines()
    if not lines:
        return ""
    important = _important_output_lines(lines)
    if important:
        primary = _first_actionable_failure_line(text)
        selected = _dedupe_preserve_order(
            ([primary] if primary else []) + important[:24] + important[-12:]
        )
        preview = "\n".join(selected)
    else:
        preview = _head_tail_lines(lines, head=40, tail=20)
    return _bounded_text_preview(
        preview,
        limit=FAILED_COMMAND_PREVIEW_CHAR_LIMIT,
    )


def _bounded_text_preview(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...<context output compressed>...\n"
    available = max(0, limit - len(marker))
    head_chars = int(available * 2 / 3)
    tail_chars = available - head_chars
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars else ""
    if "\n" in head:
        head = head.rsplit("\n", 1)[0]
    if "\n" in tail:
        tail = tail.split("\n", 1)[-1]
    return f"{head}{marker}{tail}"


def _important_output_preview(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    important = _important_output_lines(lines)
    if important:
        selected = important[:80]
        omitted = max(0, len(important) - len(selected))
        suffix = [f"...<context output compressed: {omitted} important lines omitted>..."] if omitted else []
        return "\n".join(selected + suffix)
    return _head_tail_lines(lines, head=60, tail=40)


def _important_output_lines(lines: list[str]) -> list[str]:
    selected_indexes: set[int] = set()
    for index, line in enumerate(lines):
        if not _looks_important_output_line(line):
            continue
        selected_indexes.add(index)
        # Build tools commonly place the actionable symbol/type details on
        # indented continuation lines after the headline error.
        for continuation in range(index + 1, min(len(lines), index + 5)):
            candidate = lines[continuation]
            if not candidate.strip():
                break
            selected_indexes.add(continuation)
            if continuation > index + 1 and _looks_important_output_line(candidate):
                break
    return _dedupe_preserve_order(
        [lines[index] for index in sorted(selected_indexes)]
    )


def _looks_important_output_line(line: str) -> bool:
    lower = line.lower()
    markers = (
        "exception",
        "error",
        "failed",
        "failure",
        "caused by",
        "assertion",
        "nullpointer",
        "[error]",
        "traceback",
        " at ",
        "\tat ",
    )
    return any(marker in lower for marker in markers)


def _diff_preview(diff: str) -> str:
    selected: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("diff --git ", "index ", "--- ", "+++ ", "@@ ")):
            selected.append(line)
            continue
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            selected.append(line)
    if len(selected) <= IMPORTANT_FAILURE_LINE_LIMIT:
        return "\n".join(selected)
    omitted = len(selected) - IMPORTANT_FAILURE_LINE_LIMIT
    head_count = int(IMPORTANT_FAILURE_LINE_LIMIT * 2 / 3)
    tail_count = IMPORTANT_FAILURE_LINE_LIMIT - head_count
    return "\n".join(
        selected[:head_count]
        + [f"...<diff preview compressed: {omitted} lines omitted>..."]
        + selected[-tail_count:]
    )


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        deduped.append(line)
        seen.add(line)
    return deduped


def _truncate_line(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _join_sample(values: list[str], *, limit: int) -> str:
    selected = values[:limit]
    suffix = f", +{len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(selected) + suffix


def _first_actionable_failure_line(text: str) -> str:
    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(text.splitlines()):
        compact = " ".join(line.split())
        if not compact or not _looks_important_output_line(compact) or _looks_like_noise_line(compact):
            continue
        candidates.append((_failure_line_priority(compact), -index, compact))
    if candidates:
        return _truncate_line(max(candidates)[2], limit=220)
    return _first_non_empty_line(text)


def _failure_line_priority(line: str) -> int:
    lowered = line.lower()
    if "caused by:" in lowered:
        return 100
    if any(
        marker in lowered
        for marker in (
            "cannot find symbol",
            "syntaxerror",
            "typeerror",
            "nameerror",
            "assertionerror",
            "segmentation fault",
            "panic:",
            "unresolved compilation problem",
            "compilation error",
            "type mismatch",
            "not applicable for the arguments",
        )
    ):
        return 90
    if re.search(
        r"\.(?:py|pyi|js|jsx|ts|tsx|java|kt|kts|go|rs|c|cc|cpp|cxx|h|hpp|cs|rb|php|swift|ex|exs|scala|sh|ps1|m|lua|r|dart)(?::\d+|:\[|\(\d+)",
        lowered,
    ):
        return 80
    if "<<< error!" in lowered or "<<< failure!" in lowered:
        return 70
    if "tests run:" in lowered:
        return 10
    return 40


def _looks_like_noise_line(line: str) -> bool:
    lowered = line.lower()
    return (
        "logging initialized using" in lowered
        or "slf4j" in lowered and "debug" in lowered
        or lowered.startswith("debug ")
        or "[main] debug" in lowered
    )


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        compact = " ".join(line.split())
        if compact:
            if len(compact) > 180:
                return compact[:177] + "..."
            return compact
    return ""


def _write_artifact(
    *,
    artifact_dir: Path | None,
    tool_call_id: str,
    tool_name: str,
    content: str,
) -> str | None:
    if artifact_dir is None:
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_tool_call_id = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in tool_call_id
    )
    path = artifact_dir / f"{tool_name}_{safe_tool_call_id}.txt"
    path.write_text(content, encoding="utf-8", newline="")
    return path.relative_to(artifact_dir).as_posix()
