"""Provider-neutral model response types."""

from __future__ import annotations

from typing import Any, Literal
import html
import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field


class NormalizedToolCall(BaseModel):
    """A provider-neutral tool call requested by a model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = Field(default=None, exclude=True)
    argument_parse_error: str | None = None
    arguments_likely_truncated: bool = False


class ModelUsage(BaseModel):
    """Token usage reported by a provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_miss_input_tokens: int | None = None


class ModelResponse(BaseModel):
    """A provider-neutral model response consumed by the Agent Loop.

    Model output is constrained to a turn contract: one assistant turn is either
    tool calls, a final answer, or invalid. Natural-language content emitted
    together with tool calls is commentary and must stay trace-only.

    Final-answer text is normalized into a stable internal shape. Plain text
    is accepted directly, while streaming runs may use the current
    ``<final_answer>...</final_answer>`` boundary required by the runtime prompt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_text: str | None = None
    tool_calls: list[NormalizedToolCall] = Field(default_factory=list)
    assistant_commentary: str | None = None
    reasoning_content: str | None = None
    invalid_reason: str | None = None
    usage: ModelUsage | None = None
    stop_reason: str | None = None
    raw_response: Any = None
    output_protocol: str | None = None
    structured_final: dict[str, Any] = Field(default_factory=dict)

    def kind(self) -> Literal["tool_calls", "final_text", "invalid"]:
        """Return the normalized model-turn kind."""

        if self.tool_calls:
            return "tool_calls"
        if self.final_text and self.final_text.strip():
            return "final_text"
        return "invalid"

    def is_final(self) -> bool:
        """Return whether this response is a final-answer turn."""

        return self.kind() == "final_text"

    def enforce_turn_contract(self) -> "ModelResponse":
        """Return a copy that follows the final-or-tool-call contract."""

        if self.tool_calls:
            commentary = self.assistant_commentary or self.final_text
            return self.model_copy(
                update={
                    "final_text": None,
                    "assistant_commentary": commentary,
                    "invalid_reason": None,
                    "output_protocol": "native_tool_calls",
                    "structured_final": {},
                }
            )
        if self.final_text and self.final_text.strip():
            final_contract = _normalize_final_answer_contract(self.final_text)
            if final_contract.invalid_reason is not None:
                return self.model_copy(
                    update={
                        "final_text": None,
                        "invalid_reason": final_contract.invalid_reason,
                        "output_protocol": final_contract.output_protocol,
                        "structured_final": final_contract.structured_final,
                    }
                )
            return self.model_copy(
                update={
                    "final_text": final_contract.final_text,
                    "invalid_reason": None,
                    "output_protocol": final_contract.output_protocol,
                    "structured_final": final_contract.structured_final,
                }
            )
        return self.model_copy(
            update={
                "final_text": None,
                "invalid_reason": self.invalid_reason or "empty_response",
                "output_protocol": self.output_protocol or "empty_response",
                "structured_final": {},
            }
        )


class _FinalAnswerContract(BaseModel):
    final_text: str | None
    output_protocol: str
    structured_final: dict[str, Any] = Field(default_factory=dict)
    invalid_reason: str | None = None


_FINAL_ANSWER_RE = re.compile(
    r"<final_answer\b[^>]*>.*?</final_answer>",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_final_answer_contract(text: str) -> _FinalAnswerContract:
    raw_text = text.strip()
    xml_match = _FINAL_ANSWER_RE.search(raw_text)
    if "<final_answer" in raw_text.lower() or "</final_answer>" in raw_text.lower():
        candidate = xml_match.group(0) if xml_match is not None else raw_text
        parsed = _parse_final_answer_xml(candidate)
        if parsed.invalid_reason is None:
            return parsed
        recovered = _recover_malformed_final_answer_xml(candidate)
        if recovered is not None:
            return recovered
        return parsed
    return _FinalAnswerContract(
        final_text=raw_text,
        output_protocol="plain_text",
        structured_final=_empty_structured_final(summary=raw_text),
    )


def _parse_final_answer_xml(xml_text: str) -> _FinalAnswerContract:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return _FinalAnswerContract(
            final_text=None,
            output_protocol="malformed_final_answer_xml",
            invalid_reason="malformed_final_answer_xml",
        )
    if _strip_namespace(root.tag) != "final_answer":
        return _FinalAnswerContract(
            final_text=None,
            output_protocol="malformed_final_answer_xml",
            invalid_reason="malformed_final_answer_xml",
        )

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
    visible_text = _render_structured_final(structured)
    if not visible_text:
        visible_text = " ".join("".join(root.itertext()).split())
        if visible_text:
            structured["summary"] = visible_text
    if not visible_text:
        return _FinalAnswerContract(
            final_text=None,
            output_protocol="final_answer_xml",
            structured_final=structured,
            invalid_reason="empty_final_answer_xml",
        )
    return _FinalAnswerContract(
        final_text=visible_text,
        output_protocol="final_answer_xml",
        structured_final=structured,
    )


def _empty_structured_final(*, summary: str = "") -> dict[str, Any]:
    return {
        "summary": summary,
        "details": "",
        "key_points": [],
        "evidence": [],
        "changed_files": [],
        "verification": "",
        "blockers": "",
        "notes": "",
    }


def _recover_malformed_final_answer_xml(text: str) -> _FinalAnswerContract | None:
    body = _final_answer_body(text)
    if not body:
        return None
    structured = _empty_structured_final()
    structured["summary"] = _tag_text(body, "summary")
    structured["details"] = _tag_text(body, "details")
    structured["verification"] = _tag_text(body, "verification")
    structured["blockers"] = _tag_text(body, "blockers")
    structured["notes"] = _tag_text(body, "notes")
    structured["key_points"] = _nested_tag_items(body, "key_points", "point")
    structured["evidence"] = _nested_tag_items(body, "evidence", "item")
    structured["changed_files"] = _nested_tag_items(body, "changed_files", "file")
    if not _render_structured_final(structured):
        structured["summary"] = _strip_xml_like_tags(body)
    visible_text = _render_structured_final(structured)
    if not visible_text:
        return None
    return _FinalAnswerContract(
        final_text=visible_text,
        output_protocol="malformed_final_answer_xml_recovered",
        structured_final=structured,
    )


def _final_answer_body(text: str) -> str:
    start = re.search(r"<final_answer\b[^>]*>", text, flags=re.IGNORECASE | re.DOTALL)
    if start is not None:
        body = text[start.end() :]
    else:
        body = text
    end = re.search(r"</final_answer>", body, flags=re.IGNORECASE)
    if end is not None:
        body = body[: end.start()]
    return body.strip()


def _tag_text(text: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}\b[^>]*>(.*?)</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _clean_recovered_text(match.group(1)) if match else ""


def _nested_tag_items(text: str, container: str, item: str) -> list[str]:
    container_match = re.search(
        rf"<{container}\b[^>]*>(.*?)</{container}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if container_match is None:
        return []
    body = container_match.group(1)
    items = [
        _clean_recovered_text(match.group(1))
        for match in re.finditer(rf"<{item}\b[^>]*>(.*?)</{item}>", body, flags=re.IGNORECASE | re.DOTALL)
    ]
    items = [value for value in items if value]
    if items:
        return items
    fallback = _strip_xml_like_tags(body)
    return [fallback] if fallback else []


def _strip_xml_like_tags(text: str) -> str:
    return _clean_recovered_text(re.sub(r"</?\w+\b[^>]*>", " ", text))


def _clean_recovered_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


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
    if files:
        return files
    text = " ".join("".join(container.itertext()).split())
    return [text] if text else []


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
