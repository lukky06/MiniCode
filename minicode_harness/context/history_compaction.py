"""Deterministic execution records for compacted canonical history."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import re
from typing import Any

from minicode_harness.policy import render_argv
from minicode_harness.tools.semantics import is_workspace_read, read_target

from .message_groups import MessageGroup


COMPACTED_EXECUTION_HEADING = "[MiniCode compacted execution]"
CURRENT_TOOL_FRONTIER_HEADING = "[MiniCode current tool frontier]"
MAX_EXECUTION_ENTRIES = 64
MAX_EXECUTION_RECORD_CHARS = 12_000
MAX_CURRENT_TOOL_FRONTIER_ENTRIES = 16
MAX_CURRENT_TOOL_FRONTIER_CHARS = 3_000
MAX_RESULT_PREVIEW_CHARS = 320

_WRITE_TOOLS = {"apply_patch", "edit", "write"}
_EXISTING_READ_TOOLS = {"read"}
_REUSE_STATUSES = {"duplicate_reused"}
_SUCCESS_STATUSES = {"ok", "success", "passed"}
_NOOP_STATUSES = {"idempotent_noop", "noop", "unchanged"}
_REVERT_STATUSES = {"reverted", "rolled_back", "restored"}
_FAILURE_WORDS = (
    "error",
    "failed",
    "failure",
    "timed_out",
    "timeout",
    "rejected",
    "denied",
    "blocked",
    "invalid",
)
_ARTIFACT_RE = re.compile(
    r"artifact_path\s*[:=]\s*[\"']?([^\s\"',}\]]+)",
    re.IGNORECASE,
)
_EXISTING_ENTRY_RE = re.compile(r"^-\s+([^|]+?)(?:\s+\|\s+(.*))?$")
_FAILURE_DETAIL_RE = re.compile(
    r"assert(?:ionerror)?\b|expected\b|actual\b|exception\b|traceback\b|!=|==",
    re.IGNORECASE,
)
_FAILURE_SUMMARY_RE = re.compile(
    r"\b(?:error|failed?|failure|denied|rejected|timed?_?out)\b",
    re.IGNORECASE,
)

_CATEGORY_ORDER = {
    "read": 0,
    "modified": 1,
    "write": 2,
    "command": 3,
}
_CATEGORY_PRIORITY = {
    "modified": 100,
    "write": 95,
    "command_failure": 90,
    "command_success": 80,
    "read": 40,
}


@dataclass(frozen=True)
class _ExecutionEntry:
    key: str
    category: str
    priority: int
    order: int
    rendered: str


class _ExecutionState:
    """Merge execution facts into a bounded current-state view."""

    def __init__(self) -> None:
        self._entries: dict[str, _ExecutionEntry] = {}
        self._order = 0

    def add_existing(self, block: str) -> None:
        parsed = _parse_existing_entry(block)
        if parsed is None:
            return
        tool_name, target, fields = parsed
        status = str(fields.get("status") or "success").strip().lower()
        if status in _REUSE_STATUSES or status in _NOOP_STATUSES:
            return

        if tool_name in _EXISTING_READ_TOOLS:
            if not _existing_fields_failed(status, fields, block):
                self._set_read(target)
            return

        if tool_name in _WRITE_TOOLS or tool_name in {"modified", "reverted"}:
            paths = _split_existing_targets(target)
            if tool_name == "reverted" or status in _REVERT_STATUSES:
                for path in paths:
                    self._entries.pop(f"modified:{path}", None)
                return
            if _existing_fields_failed(status, fields, block):
                self._add_generic_existing(
                    tool_name=tool_name,
                    target=target,
                    fields=fields,
                    block=block,
                    category="write",
                )
                return
            for path in paths:
                self._set_modified(path)
            return

        if tool_name == "run_command":
            rendered = _render_existing_entry(tool_name, target, fields)
            failed = _existing_fields_failed(status, fields, block)
            self._set(
                key=f"command:{target}",
                category="command",
                priority=_CATEGORY_PRIORITY[
                    "command_failure" if failed else "command_success"
                ],
                rendered=rendered,
            )

    def add_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        parsed_result: dict[str, Any],
        result_text: str,
        status: str,
    ) -> None:
        if status == "duplicate_reused":
            return
        failed = _status_is_failure(status, parsed_result, result_text)

        if is_workspace_read(tool_name, arguments):
            if not failed:
                payload = parsed_result.get("payload")
                result_path = payload.get("path") if isinstance(payload, dict) else None
                self._set_read(
                    str(result_path or read_target(tool_name, arguments) or "").strip()
                )
            return

        if tool_name in _WRITE_TOOLS:
            paths = _write_paths(arguments, parsed_result)
            if _write_was_reverted(status, parsed_result):
                for path in paths:
                    self._entries.pop(f"modified:{path}", None)
                return
            if _write_was_noop(status, parsed_result):
                return
            if not failed:
                for path in paths:
                    self._set_modified(path)
                if paths:
                    return
            rendered = _render_entry(
                tool_name=tool_name,
                arguments=arguments,
                parsed_result=parsed_result,
                result_text=result_text,
                status=status,
                side_effecting=True,
            )
            if rendered:
                self._set(
                    key=f"write:{tool_name}:{_resource_from_arguments(arguments)}:{rendered}",
                    category="write",
                    priority=_CATEGORY_PRIORITY["write"],
                    rendered=rendered,
                )
            return

        if tool_name != "run_command":
            return

        rendered = _render_entry(
            tool_name=tool_name,
            arguments=arguments,
            parsed_result=parsed_result,
            result_text=result_text,
            status=status,
            side_effecting=False,
        )
        if not rendered:
            return
        command = _command_from_arguments(arguments)
        self._set(
            key=f"command:{command}",
            category="command",
            priority=_CATEGORY_PRIORITY[
                "command_failure" if failed else "command_success"
            ],
            rendered=rendered,
        )

    def entries(self) -> list[_ExecutionEntry]:
        return list(self._entries.values())

    def _set_read(self, path: str) -> None:
        compact = str(path or "").strip()
        if not compact:
            return
        self._set(
            key=f"read:{compact}",
            category="read",
            priority=_CATEGORY_PRIORITY["read"],
            rendered=f"- read | {_single_line(compact, 500)}",
        )

    def _set_modified(self, path: str) -> None:
        compact = str(path or "").strip()
        if not compact:
            return
        self._set(
            key=f"modified:{compact}",
            category="modified",
            priority=_CATEGORY_PRIORITY["modified"],
            rendered=f"- modified | {_single_line(compact, 500)}",
        )

    def _add_generic_existing(
        self,
        *,
        tool_name: str,
        target: str,
        fields: dict[str, Any],
        block: str,
        category: str,
    ) -> None:
        rendered = _render_existing_entry(tool_name, target, fields)
        self._set(
            key=f"{category}:{tool_name}:{target}:{rendered or block}",
            category=category,
            priority=_CATEGORY_PRIORITY[category],
            rendered=rendered or block.strip(),
        )

    def _set(
        self,
        *,
        key: str,
        category: str,
        priority: int,
        rendered: str,
    ) -> None:
        self._order += 1
        self._entries[key] = _ExecutionEntry(
            key=key,
            category=category,
            priority=priority,
            order=self._order,
            rendered=rendered.strip(),
        )


def compact_execution_history(
    groups: list[MessageGroup],
    *,
    tool_effects: dict[str, dict[str, bool]] | None = None,
) -> str:
    """Render only non-replayable execution state from complete message groups."""

    rendered, _ = compact_execution_history_with_details(
        groups,
        tool_effects=tool_effects,
    )
    return rendered


def compact_execution_history_with_details(
    groups: list[MessageGroup],
    *,
    max_chars: int = MAX_EXECUTION_RECORD_CHARS,
    tool_effects: dict[str, dict[str, bool]] | None = None,
) -> tuple[str, int]:
    """Keep only workspace reads, write outcomes, and command outcomes."""

    del tool_effects
    state = _ExecutionState()
    for group in groups:
        if not group:
            continue
        first = group[0]
        existing = _existing_execution_record(first)
        if existing:
            for block in existing:
                state.add_existing(block)
            continue
        if first.get("role") != "assistant" or not first.get("tool_calls"):
            continue

        result_by_id = {
            str(message.get("tool_call_id") or ""): message
            for message in group[1:]
            if message.get("role") == "tool"
        }
        for call in first.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            function = call.get("function") or {}
            tool_name = str(function.get("name") or "unknown")
            result_message = result_by_id.get(call_id)
            if result_message is None:
                continue
            arguments = _parse_arguments(function.get("arguments"))
            result_text = str(result_message.get("content") or "")
            parsed_result = _parse_tool_result(result_text)
            status = _execution_status(parsed_result, result_text)
            state.add_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                parsed_result=parsed_result,
                result_text=result_text,
                status=status,
            )

    entries, omitted = _bounded_entries(state.entries(), max_chars=max_chars)
    if not entries:
        return "", omitted
    return "\n\n".join([COMPACTED_EXECUTION_HEADING, *entries]), omitted


def compact_current_tool_frontier(
    groups: list[MessageGroup],
    *,
    max_chars: int = MAX_CURRENT_TOOL_FRONTIER_CHARS,
) -> tuple[str, int]:
    """Replace recent current-Turn tool protocol with bounded replayable facts.

    The auxiliary semantic model may inspect the exact groups before this runs.
    The canonical history keeps only deterministic read/command summaries after
    successful semantic compaction, so exact source evidence must be reacquired
    through normal tools when needed.
    """

    entries: list[str] = []
    for group in groups:
        if not group:
            continue
        first = group[0]
        existing_entries = _existing_current_tool_frontier_entries(first)
        if existing_entries:
            entries.extend(existing_entries)
            continue
        if first.get("role") != "assistant" or not first.get("tool_calls"):
            continue
        result_by_id = {
            str(message.get("tool_call_id") or ""): message
            for message in group[1:]
            if message.get("role") == "tool"
        }
        for call in first.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            function = call.get("function") or {}
            tool_name = str(function.get("name") or "unknown")
            result_message = result_by_id.get(call_id)
            if result_message is None:
                continue
            arguments = _parse_arguments(function.get("arguments"))
            result_text = str(result_message.get("content") or "")
            parsed_result = _parse_tool_result(result_text)
            status = _execution_status(parsed_result, result_text)
            failed = _status_is_failure(status, parsed_result, result_text)
            if is_workspace_read(tool_name, arguments):
                if failed:
                    continue
            elif tool_name not in _WRITE_TOOLS and tool_name != "run_command":
                continue
            rendered = _render_current_tool_frontier_entry(
                tool_name=tool_name,
                arguments=arguments,
                parsed_result=parsed_result,
                result_text=result_text,
                status=status,
            )
            if rendered:
                entries.append(rendered)

    entries = _deduplicate_current_tool_frontier_entries(entries)
    original_count = len(entries)
    if len(entries) > MAX_CURRENT_TOOL_FRONTIER_ENTRIES:
        entries = entries[-MAX_CURRENT_TOOL_FRONTIER_ENTRIES:]

    char_limit = max(0, max_chars)
    while len(entries) > 1 and len(_render_current_tool_frontier(entries)) > char_limit:
        entries.pop(0)
    if entries and len(_render_current_tool_frontier(entries)) > char_limit:
        shortened = _shorten_frontier_entry(entries[0], char_limit)
        entries = [shortened] if shortened else []

    if not entries:
        return "", original_count
    return (
        _render_current_tool_frontier(entries),
        original_count - len(entries),
    )


def _existing_current_tool_frontier_entries(message: dict[str, Any]) -> list[str]:
    if message.get("role") != "assistant" or message.get("tool_calls"):
        return []
    content = str(message.get("content") or "").strip()
    if not content.startswith(CURRENT_TOOL_FRONTIER_HEADING):
        return []
    entries: list[str] = []
    for block in content.split("\n\n")[1:]:
        compact = block.strip()
        if not compact or compact.startswith("- boundary:"):
            continue
        parsed = _parse_existing_entry(compact)
        if parsed is None:
            continue
        tool_name, target, fields = parsed
        status = str(fields.get("status") or "success").strip().lower()
        if tool_name in _EXISTING_READ_TOOLS:
            if _existing_fields_failed(status, fields, compact) or not target:
                continue
            lines = [f"- read | {_single_line(target, 500)}"]
            read_range = fields.get("range")
            if read_range not in (None, ""):
                lines.append(f"  range: {_single_line(str(read_range), 120)}")
            entries.append("\n".join(lines))
            continue
        if tool_name in _WRITE_TOOLS or tool_name == "run_command":
            entries.append(_render_existing_entry(tool_name, target, fields))
    return entries


def _deduplicate_current_tool_frontier_entries(entries: list[str]) -> list[str]:
    latest: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(entries):
        lines = entry.splitlines()
        first_line = lines[0].strip() if lines else entry
        latest[first_line] = (index, entry)
    return [entry for _, entry in sorted(latest.values(), key=lambda item: item[0])]


def _render_current_tool_frontier_entry(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    parsed_result: dict[str, Any],
    result_text: str,
    status: str,
) -> str:
    if is_workspace_read(tool_name, arguments):
        payload = parsed_result.get("payload")
        result_path = payload.get("path") if isinstance(payload, dict) else None
        target = str(
            result_path or read_target(tool_name, arguments) or "<unknown>"
        ).strip()
        lines = [f"- read | {_single_line(target, 500)}"]
        if isinstance(payload, dict):
            start_line = payload.get("start_line")
            end_line = payload.get("end_line")
            total_lines = payload.get("total_lines")
            if any(value is not None for value in (start_line, end_line, total_lines)):
                lines.append(
                    "  range: "
                    f"{start_line or '?'}-{end_line or '?'} of {total_lines or '?'}"
                )
        return "\n".join(lines)

    target = (
        _command_from_arguments(arguments)
        if tool_name == "run_command"
        else _resource_from_arguments(arguments)
    )
    first_line = f"- {tool_name}"
    if target:
        first_line += f" | {_single_line(target, 500)}"
    lines = [first_line, f"  status: {status}"]

    returncode = _optional_int(parsed_result.get("returncode"))
    if returncode is not None:
        lines.append(f"  returncode: {returncode}")
    timed_out = parsed_result.get("timed_out")
    if isinstance(timed_out, bool):
        lines.append(f"  timed_out: {str(timed_out).lower()}")
    for artifact in _artifact_refs(parsed_result, result_text):
        lines.append(f"  artifact: {_single_line(artifact, 500)}")

    preview = _result_preview(
        parsed_result,
        result_text,
        failure_focused=(
            tool_name == "run_command"
            and _status_is_failure(status, parsed_result, result_text)
        ),
    )
    if preview:
        lines.append(f"  result: {preview}")
    return "\n".join(lines)


def _render_current_tool_frontier(entries: list[str]) -> str:
    boundary = (
        "- boundary: full Tool Call/Tool Result protocol was removed after semantic "
        "compaction; use normal tools to reacquire exact source or command evidence."
    )
    return "\n\n".join([CURRENT_TOOL_FRONTIER_HEADING, *entries, boundary])


def _shorten_frontier_entry(rendered: str, max_chars: int) -> str:
    if max_chars <= len(CURRENT_TOOL_FRONTIER_HEADING) + 32:
        return ""
    available = max_chars - len(CURRENT_TOOL_FRONTIER_HEADING) - 180
    if available <= 0:
        return ""
    lines = rendered.splitlines()
    if not lines:
        return ""
    preferred = [lines[0]]
    for prefix in (
        "  status:",
        "  returncode:",
        "  timed_out:",
        "  range:",
        "  result:",
        "  artifact:",
    ):
        preferred.extend(line for line in lines[1:] if line.startswith(prefix))
    compact = "\n".join(preferred)
    return _single_line(compact, available)


def _render_entry(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    parsed_result: dict[str, Any],
    result_text: str,
    status: str,
    side_effecting: bool,
) -> str:
    if tool_name == "run_command":
        target = _command_from_arguments(arguments)
    else:
        target = _resource_from_arguments(arguments)

    first_line = f"- {tool_name}"
    if target:
        first_line += f" | {_single_line(target, 500)}"
    lines = [first_line, f"  status: {status}"]

    returncode = _optional_int(parsed_result.get("returncode"))
    if returncode is not None:
        lines.append(f"  returncode: {returncode}")
    timed_out = parsed_result.get("timed_out")
    if isinstance(timed_out, bool):
        lines.append(f"  timed_out: {str(timed_out).lower()}")

    for artifact in _artifact_refs(parsed_result, result_text):
        lines.append(f"  artifact: {_single_line(artifact, 500)}")

    failed = _status_is_failure(status, parsed_result, result_text)
    preview = (
        _result_preview(
            parsed_result,
            result_text,
            failure_focused=tool_name == "run_command" and failed,
        )
        if tool_name == "run_command" or failed
        else _side_effect_result_preview(parsed_result)
        if side_effecting
        else ""
    )
    if preview:
        lines.append(f"  result: {preview}")
    return "\n".join(lines)


def _existing_execution_record(message: dict[str, Any]) -> list[str]:
    if message.get("role") != "assistant" or message.get("tool_calls"):
        return []
    content = str(message.get("content") or "").strip()
    if not content.startswith(COMPACTED_EXECUTION_HEADING):
        return []
    body = content[len(COMPACTED_EXECUTION_HEADING) :].strip()
    return [entry.strip() for entry in re.split(r"\n\s*\n(?=- )", body) if entry.strip()]


def _parse_existing_entry(block: str) -> tuple[str, str, dict[str, Any]] | None:
    lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
    if not lines:
        return None
    match = _EXISTING_ENTRY_RE.match(lines[0].strip())
    if match is None:
        return None
    tool_name = match.group(1).strip()
    target = str(match.group(2) or "").strip()
    fields: dict[str, Any] = {}
    artifacts: list[str] = []
    for line in lines[1:]:
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        normalized = key.strip().lower()
        compact = value.strip()
        if normalized == "artifact":
            artifacts.append(compact)
        else:
            fields[normalized] = _coerce_scalar(compact)
    if artifacts:
        fields["artifacts"] = artifacts
    return tool_name, target, fields


def _render_existing_entry(
    tool_name: str,
    target: str,
    fields: dict[str, Any],
) -> str:
    first = f"- {tool_name}"
    if target:
        first += f" | {_single_line(target, 500)}"
    lines = [first]
    status = fields.get("status")
    if status not in (None, ""):
        lines.append(f"  status: {status}")
    returncode = _optional_int(fields.get("returncode"))
    if returncode is not None:
        lines.append(f"  returncode: {returncode}")
    timed_out = fields.get("timed_out")
    if isinstance(timed_out, bool):
        lines.append(f"  timed_out: {str(timed_out).lower()}")
    for artifact in _existing_artifacts(fields):
        lines.append(f"  artifact: {_single_line(artifact, 500)}")
    result = fields.get("result")
    if result not in (None, ""):
        lines.append(f"  result: {_single_line(str(result), MAX_RESULT_PREVIEW_CHARS)}")
    return "\n".join(lines)


def _existing_artifacts(fields: dict[str, Any]) -> list[str]:
    values = fields.get("artifacts")
    if not isinstance(values, list):
        return []
    return _unique(str(value) for value in values)


def _existing_fields_failed(status: str, fields: dict[str, Any], block: str) -> bool:
    return _status_is_failure(
        status,
        {
            "returncode": fields.get("returncode"),
            "timed_out": fields.get("timed_out"),
        },
        block,
    )


def _split_existing_targets(target: str) -> list[str]:
    return _unique(part.strip() for part in target.split(","))


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_tool_result(content: str) -> dict[str, Any]:
    header, marker, result = content.partition("\nresult:\n")
    payload: dict[str, Any] = {}
    for line in header.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {
            "tool",
            "status",
            "resource",
            "artifact_path",
            "summary",
            "returncode",
            "timed_out",
        }:
            payload[key.strip()] = _coerce_scalar(value.strip())
    raw_result = result if marker else content
    stripped = raw_result.strip()
    if stripped:
        try:
            structured = json.loads(stripped)
        except json.JSONDecodeError:
            structured = None
        if isinstance(structured, dict):
            payload["payload"] = structured
            for key in (
                "status",
                "summary",
                "message",
                "error",
                "error_type",
                "returncode",
                "timed_out",
                "path",
                "files",
                "command",
                "argv",
                "artifact_path",
                "changed",
            ):
                if key in structured:
                    payload.setdefault(key, structured[key])
        payload["raw_result"] = stripped
    return payload


def _execution_status(parsed_result: dict[str, Any], result_text: str) -> str:
    status = str(parsed_result.get("status") or "").strip().lower()
    if status:
        return status
    if parsed_result.get("timed_out") is True:
        return "timed_out"
    returncode = _optional_int(parsed_result.get("returncode"))
    if returncode is not None:
        return "passed" if returncode == 0 else "failed"
    payload = parsed_result.get("payload")
    if isinstance(payload, dict):
        if payload.get("success") is False:
            return "failed"
        if any(payload.get(key) not in (None, "", False) for key in ("error", "error_type")):
            return "failed"
        return "success"
    return "failed" if _status_is_failure("", parsed_result, result_text) else "success"


def _status_is_failure(
    status: str,
    parsed_result: dict[str, Any],
    result_text: str,
) -> bool:
    if status in _SUCCESS_STATUSES or status in _REUSE_STATUSES or status in _NOOP_STATUSES:
        return False
    returncode = _optional_int(parsed_result.get("returncode"))
    if returncode is not None:
        return returncode != 0
    if parsed_result.get("timed_out") is True:
        return True
    lowered = f"{status} {result_text}".lower()
    return any(word in lowered for word in _FAILURE_WORDS)


def _write_was_noop(status: str, parsed_result: dict[str, Any]) -> bool:
    if status in _NOOP_STATUSES:
        return True
    payload = parsed_result.get("payload")
    if isinstance(payload, dict) and payload.get("changed") is False:
        return True
    return parsed_result.get("changed") is False


def _write_was_reverted(status: str, parsed_result: dict[str, Any]) -> bool:
    if status in _REVERT_STATUSES:
        return True
    payload = parsed_result.get("payload")
    if isinstance(payload, dict):
        return bool(payload.get("reverted") or payload.get("rolled_back"))
    return False


def _command_from_arguments(arguments: dict[str, Any]) -> str:
    argv = arguments.get("argv")
    if isinstance(argv, list) and all(isinstance(item, str) for item in argv):
        return render_argv(argv)
    return ""


def _write_paths(
    arguments: dict[str, Any],
    parsed_result: dict[str, Any],
) -> list[str]:
    values: list[str] = []
    for key in ("path", "root"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    files = arguments.get("files")
    if isinstance(files, list):
        values.extend(str(item).strip() for item in files if str(item).strip())
    payload = parsed_result.get("payload")
    if isinstance(payload, dict):
        path = payload.get("path")
        if isinstance(path, str) and path.strip():
            values.append(path.strip())
        result_files = payload.get("files")
        if isinstance(result_files, list):
            values.extend(str(item).strip() for item in result_files if str(item).strip())
    resource = parsed_result.get("resource")
    if isinstance(resource, str) and resource.strip():
        values.append(resource.strip())
    return _unique(values)


def _resource_from_arguments(arguments: dict[str, Any]) -> str:
    for key in ("target", "path", "root", "query", "pattern", "name", "title"):
        value = arguments.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _artifact_refs(parsed_result: dict[str, Any], result_text: str) -> list[str]:
    refs: list[str] = []
    direct = parsed_result.get("artifact_path")
    if isinstance(direct, str) and direct.strip():
        refs.append(direct.strip())
    payload = parsed_result.get("payload")
    if isinstance(payload, dict):
        nested = payload.get("artifact_path")
        if isinstance(nested, str) and nested.strip():
            refs.append(nested.strip())
    refs.extend(match.group(1).rstrip(".;") for match in _ARTIFACT_RE.finditer(result_text))
    return _unique(refs)


def _result_preview(
    parsed_result: dict[str, Any],
    result_text: str,
    *,
    failure_focused: bool = False,
) -> str:
    payload = parsed_result.get("payload")
    sources = [payload, parsed_result] if isinstance(payload, dict) else [parsed_result]

    if failure_focused:
        failure_values = _unique(
            str(value)
            for key in ("error", "message", "stderr", "stdout", "summary")
            for source in sources
            if isinstance(source, dict)
            for value in [source.get(key)]
            if value not in (None, "")
        )
        preview = _failure_diagnostic_preview(
            "\n".join(failure_values),
            MAX_RESULT_PREVIEW_CHARS,
        )
        if preview:
            return preview

    for key in ("summary", "error", "message"):
        for source in sources:
            if not isinstance(source, dict):
                continue
            value = source.get(key)
            if value not in (None, ""):
                return _single_line(str(value), MAX_RESULT_PREVIEW_CHARS)
    for key in ("stderr", "stdout"):
        if not isinstance(payload, dict):
            continue
        value = payload.get(key)
        if value not in (None, ""):
            return _single_line(str(value), MAX_RESULT_PREVIEW_CHARS)
    raw_result = str(parsed_result.get("raw_result") or result_text).strip()
    return _single_line(raw_result, MAX_RESULT_PREVIEW_CHARS)


def _failure_diagnostic_preview(value: str, limit: int) -> str:
    lines = _unique(
        _single_line(line, max(1, limit))
        for line in str(value).splitlines()
        if line.strip()
    )
    if not lines:
        return ""

    details = [line for line in lines if _FAILURE_DETAIL_RE.search(line)]
    summaries = [
        line
        for line in lines
        if line not in details and _FAILURE_SUMMARY_RE.search(line)
    ]
    candidates = _unique(
        [
            *details[-3:],
            *summaries[-2:],
            lines[0],
            lines[-1],
        ]
    )
    return _bounded_preview_segments(candidates, limit)


def _bounded_preview_segments(segments: Iterable[str], limit: int) -> str:
    selected: list[str] = []
    used = 0
    for segment in segments:
        compact = _single_line(str(segment), min(180, max(1, limit)))
        separator_chars = 3 if selected else 0
        remaining = limit - used - separator_chars
        if remaining <= 0:
            break
        if len(compact) > remaining:
            if remaining < 20:
                break
            compact = _single_line(compact, remaining)
        selected.append(compact)
        used += separator_chars + len(compact)
    return " | ".join(selected)


def _side_effect_result_preview(parsed_result: dict[str, Any]) -> str:
    """Keep replay-prevention identifiers, not a side-effect tool's full payload."""

    payload = parsed_result.get("payload")
    sources = [payload, parsed_result] if isinstance(payload, dict) else [parsed_result]
    selected: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            normalized = str(key).lower()
            if (
                normalized == "id"
                or normalized.endswith("_id")
                or normalized in {"url", "status", "summary", "message"}
            ) and value not in (None, "", [], {}):
                selected.setdefault(str(key), value)
    if not selected:
        return ""
    return _single_line(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":"), default=str),
        MAX_RESULT_PREVIEW_CHARS,
    )


def _bounded_entries(
    entries: list[_ExecutionEntry],
    *,
    max_chars: int = MAX_EXECUTION_RECORD_CHARS,
) -> tuple[list[str], int]:
    original_count = len(entries)
    selected = list(entries)
    char_limit = max(0, max_chars)

    while len(selected) > MAX_EXECUTION_ENTRIES:
        selected.remove(min(selected, key=lambda entry: (entry.priority, entry.order)))
    while len(selected) > 1 and _entries_chars(selected) > char_limit:
        selected.remove(min(selected, key=lambda entry: (entry.priority, entry.order)))

    if selected and _entries_chars(selected) > char_limit:
        only = selected[0]
        shortened = _shorten_entry(only.rendered, char_limit)
        if shortened:
            selected[0] = _ExecutionEntry(
                key=only.key,
                category=only.category,
                priority=only.priority,
                order=only.order,
                rendered=shortened,
            )
        else:
            selected.clear()

    selected.sort(key=_entry_render_sort_key)
    return [entry.rendered for entry in selected], original_count - len(selected)


def _entry_render_sort_key(entry: _ExecutionEntry) -> tuple[int, Any, int]:
    rank = _CATEGORY_ORDER.get(entry.category, 99)
    if entry.category == "modified":
        return rank, entry.rendered, entry.order
    return rank, entry.order, entry.order


def _entries_chars(entries: list[_ExecutionEntry]) -> int:
    return len("\n\n".join(entry.rendered for entry in entries))


def _shorten_entry(rendered: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(rendered) <= max_chars:
        return rendered
    lines = rendered.splitlines()
    if not lines:
        return ""
    preferred = [lines[0]]
    for prefix in ("  returncode:", "  result:", "  artifact:", "  status:", "  timed_out:"):
        preferred.extend(line for line in lines[1:] if line.startswith(prefix))
    compact = "\n".join(_single_line(line, 180) for line in preferred)
    if len(compact) <= max_chars:
        return compact
    if max_chars < 20:
        return compact[:max_chars]
    return _single_line(compact, max_chars)


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    integer = _optional_int(value)
    return integer if integer is not None else value


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _single_line(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 15)].rstrip() + "...<compacted>"


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        compact = str(value or "").strip()
        if compact and compact not in result:
            result.append(compact)
    return result
