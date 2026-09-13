"""Typed output sinks for MiniCode user-facing output."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Any, Protocol, TextIO
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ContextUsage:
    """Complete prompt-budget facts for one model request."""

    build_duration_ms: int
    token_estimate: int
    context_window: int
    prompt_budget: int
    reserved_output: int

    @property
    def context_remaining(self) -> int:
        return max(self.prompt_budget - self.token_estimate, 0)


class StreamHandler(Protocol):
    """Receives incremental model output deltas."""

    def model_stream_started(self) -> None:
        """Called before a streaming model response starts."""

    def model_text_delta(self, text: str) -> None:
        """Called for each visible model text delta."""

    def model_reasoning_delta(self, text: str) -> None:
        """Called for provider-exposed reasoning suitable for presentation."""


class OutputSink(StreamHandler, Protocol):
    """User-facing output boundary for the agent loop."""

    def context_built(self, *, usage: ContextUsage) -> None:
        """Called after prompt context is built for one model step."""

    def tool_call_started(
        self,
        *,
        step: int,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> None:
        """Called when the model requests a tool call."""

    def tool_call_finished(
        self,
        *,
        step: int,
        tool_name: str,
        status: str,
        tool_call_id: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called when a tool call result is available."""


class NullOutputSink:
    """No-op output sink for tests, benchmarks, and embedded callers."""

    def context_built(self, *, usage: ContextUsage) -> None:
        pass

    def model_stream_started(self) -> None:
        pass

    def model_text_delta(self, text: str) -> None:
        pass

    def model_reasoning_delta(self, text: str) -> None:
        pass

    def tool_call_started(
        self,
        *,
        step: int,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> None:
        pass

    def tool_call_finished(
        self,
        *,
        step: int,
        tool_name: str,
        status: str,
        tool_call_id: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass


class FinalAnswerXmlStreamEmitter:
    """Stream text only after an explicit ``<final_answer>`` boundary.

    Provider commentary before a native tool call remains buffered and hidden.
    Once the final-answer opening tag appears, inner text is emitted
    incrementally while retaining enough trailing characters to detect the
    closing tag without leaking it.
    """

    _OPEN_RE = re.compile(r"<final_answer\b[^>]*>", re.IGNORECASE)
    _CLOSE = "</final_answer>"

    def __init__(self, sink: StreamHandler) -> None:
        self.sink = sink
        self.buffer = ""
        self.root_started = False
        self.root_closed = False
        self.structured_mode: bool | None = None
        self.emitted_chars = 0

    @property
    def emitted(self) -> bool:
        return self.emitted_chars > 0

    def feed(self, text: str) -> None:
        """Consume a raw model text delta and emit bounded final-answer text."""

        if not text or self.root_closed:
            return
        self.buffer += text
        if not self.root_started:
            match = self._OPEN_RE.search(self.buffer)
            if match is None:
                self.buffer = self.buffer[-256:]
                return
            self.root_started = True
            self.buffer = self.buffer[match.end() :]

        if self.structured_mode is None:
            stripped = self.buffer.lstrip()
            if stripped:
                self.structured_mode = stripped.startswith("<")

        lowered = self.buffer.lower()
        close_index = lowered.find(self._CLOSE)
        if close_index >= 0:
            inner = self.buffer[:close_index]
            if self.structured_mode:
                visible = _render_final_answer_xml(
                    "<final_answer>" + inner + self._CLOSE
                )
                self._emit(visible)
            else:
                self._emit(inner)
            self.buffer = ""
            self.root_closed = True
            return

        if self.structured_mode:
            return
        retained = _closing_tag_prefix_suffix(self.buffer, self._CLOSE)
        emit_end = len(self.buffer) - retained
        if emit_end > 0:
            self._emit(self.buffer[:emit_end])
            self.buffer = self.buffer[emit_end:]

    def flush_partial(self) -> None:
        """Flush a plain final-answer tail after the provider ends the stream."""

        if (
            self.root_started
            and not self.root_closed
            and self.structured_mode is False
            and self.buffer
        ):
            self._emit(self.buffer)
            self.buffer = ""

    def _emit(self, text: str) -> None:
        if not text:
            return
        self.sink.model_text_delta(text)
        self.emitted_chars += len(text)


class TextOutputSink:
    """Plain terminal output sink used by the CLI."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        if stream is None:
            reconfigure = getattr(self.stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except (AttributeError, OSError, ValueError):
                    pass

    def context_built(self, *, usage: ContextUsage) -> None:
        self._write_line(
            f"[context] built in {usage.build_duration_ms}ms, "
            f"tokens={usage.token_estimate}, remaining={usage.context_remaining}"
        )

    def model_stream_started(self) -> None:
        pass

    def model_text_delta(self, text: str) -> None:
        self._write(text)

    def model_reasoning_delta(self, text: str) -> None:
        del text

    def tool_call_started(
        self,
        *,
        step: int,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> None:
        self._write_line(f"[tool] {tool_name} {_format_tool_arguments(arguments)}")

    def tool_call_finished(
        self,
        *,
        step: int,
        tool_name: str,
        status: str,
        tool_call_id: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._write_line(f"[tool] {tool_name} {status}")

    def _write_line(self, text: str) -> None:
        self._write(text + "\n")

    def _write(self, text: str) -> None:
        """Write terminal text without crashing on the active console codec."""

        encoding = getattr(self.stream, "encoding", None)
        if encoding:
            text = text.encode(encoding, errors="replace").decode(encoding)
        self.stream.write(text)
        self.stream.flush()


def _format_tool_arguments(arguments: dict[str, Any]) -> str:
    if not arguments:
        return "{}"
    parts: list[str] = []
    for key in sorted(arguments):
        value = arguments[key]
        if isinstance(value, str):
            rendered = repr(value if len(value) <= 80 else value[:77] + "...")
        else:
            rendered = repr(value)
        parts.append(f"{key}={rendered}")
    return "{" + ", ".join(parts) + "}"


_FINAL_ANSWER_RE = re.compile(
    r"<final_answer\b[^>]*>.*?</final_answer>",
    re.IGNORECASE | re.DOTALL,
)


def _closing_tag_prefix_suffix(text: str, closing_tag: str) -> int:
    """Return the suffix length that may still become the closing tag."""

    lowered = text.lower()
    closing = closing_tag.lower()
    limit = min(len(lowered), len(closing) - 1)
    for size in range(limit, 0, -1):
        if lowered.endswith(closing[:size]):
            return size
    return 0


def _render_final_answer_xml(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    if _strip_namespace(root.tag) != "final_answer":
        return ""
    structured = {
        "summary": _child_text(root, "summary"),
        "details": _child_text(root, "details"),
        "key_points": _list_items(root, "key_points", "point"),
        "evidence": _list_items(root, "evidence", "item"),
        "changed_files": _changed_files(root),
        "verification": _child_text(root, "verification"),
        "blockers": _child_text(root, "blockers"),
        "notes": _child_text(root, "notes"),
    }
    rendered = _render_structured_final(structured)
    if rendered:
        return rendered
    return "".join(root.itertext()).strip()


def _render_structured_final(structured: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = str(structured.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    details = str(structured.get("details") or "").strip()
    if details:
        lines.append("详细说明：\n" + details)
    key_points = [str(point).strip() for point in structured.get("key_points") or [] if str(point).strip()]
    if key_points:
        lines.append("要点：\n" + "\n".join(f"- {point}" for point in key_points))
    evidence = [str(item).strip() for item in structured.get("evidence") or [] if str(item).strip()]
    if evidence:
        lines.append("依据：\n" + "\n".join(f"- {item}" for item in evidence))
    changed_files = [str(path).strip() for path in structured.get("changed_files") or [] if str(path).strip()]
    if changed_files:
        lines.append("修改文件：" + ", ".join(changed_files))
    verification = str(structured.get("verification") or "").strip()
    if verification:
        lines.append("验证：" + verification)
    blockers = str(structured.get("blockers") or "").strip()
    if blockers:
        lines.append("阻塞：" + blockers)
    notes = str(structured.get("notes") or "").strip()
    if notes:
        lines.append("备注：" + notes)
    return "\n".join(lines).strip()


def _child_text(root: ET.Element, name: str) -> str:
    for child in root:
        if _strip_namespace(child.tag) == name:
            return " ".join("".join(child.itertext()).split())
    return ""


def _list_items(root: ET.Element, container_name: str, item_name: str) -> list[str]:
    container = None
    for child in root:
        if _strip_namespace(child.tag) == container_name:
            container = child
            break
    if container is None:
        return []
    items: list[str] = []
    for child in container:
        if _strip_namespace(child.tag) == item_name:
            text = " ".join("".join(child.itertext()).split())
            if text:
                items.append(text)
    if items:
        return items
    text = " ".join("".join(container.itertext()).split())
    return [text] if text else []


def _changed_files(root: ET.Element) -> list[str]:
    container = None
    for child in root:
        if _strip_namespace(child.tag) == "changed_files":
            container = child
            break
    if container is None:
        return []
    files: list[str] = []
    for child in container:
        if _strip_namespace(child.tag) == "file":
            text = " ".join("".join(child.itertext()).split())
            if text:
                files.append(text)
    return files


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()
