"""Bounded read-only subagent execution.

A subagent receives a fresh message history and a read-only tool registry. Its
intermediate transcript is isolated from the parent; only the final summary is
returned as the parent tool result. One runner is sequential and non-recursive;
the parent may execute multiple independent runners in bounded foreground parallelism.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Literal

from pydantic import BaseModel

from minicode_harness.context import (
    ContextObservation,
    ContextPreparer,
    LLMSemanticHistoryCompactor,
    TokenBudget,
    build_observation,
    render_tool_result_message,
)
from minicode_harness.models import ModelClient, ModelRequest, ModelResponse
from minicode_harness.policy import RiskLevel
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.recovery import ModelRecoveryPolicy
from minicode_harness.tools import ToolRegistry
from minicode_harness.trace import TraceWriter


SUBAGENT_MAX_CONTEXT_WINDOW = 64_000
SUBAGENT_MAX_RESERVED_OUTPUT = 8_192


class SubagentResult(BaseModel):
    status: Literal["completed", "partial"]
    summary: str


def build_subagent_token_budget(model_client: ModelClient) -> TokenBudget:
    """Derive a bounded child budget from provider/model capabilities."""

    capabilities = model_client.capabilities
    context_window = min(
        max(1, int(capabilities.context_window)),
        SUBAGENT_MAX_CONTEXT_WINDOW,
    )
    reserved_output = min(
        max(1, int(capabilities.reserved_output_tokens)),
        SUBAGENT_MAX_RESERVED_OUTPUT,
        max(1, context_window - 1),
    )
    return TokenBudget(
        context_budget=context_window,
        reserved_output=reserved_output,
        soft_limit=0.8,
        hard_limit=0.95,
    )


class ReadonlySubagentRunner:
    """Execute one delegated analysis task with a small fixed budget."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        model_client: ModelClient,
        trace_writer: TraceWriter,
        artifact_dir: Path | str,
        recovery_policy: ModelRecoveryPolicy | None = None,
        mcp_manager: Any | None = None,
        max_steps: int = 8,
        max_tool_calls: int = 12,
        cancellation_token: CancellationToken | None = None,
        max_duration_seconds: float = 180.0,
    ) -> None:
        self.workspace = str(Path(workspace).resolve())
        self.model_client = model_client
        self.trace_writer = trace_writer
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_policy = recovery_policy or ModelRecoveryPolicy()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.cancellation_token = cancellation_token or CancellationToken()
        self.max_duration_seconds = max(1.0, float(max_duration_seconds))
        self.tools = ToolRegistry(
            self.workspace,
            enable_write=False,
            artifact_dir=str(self.artifact_dir),
            skill_loader=None,
            skill_names=[],
            mcp_manager=mcp_manager,
            subagent_handler=None,
        )
        token_budget = build_subagent_token_budget(self.model_client)
        self.preparer = ContextPreparer(
            token_budget,
            semantic_compactor=LLMSemanticHistoryCompactor(
                self.model_client,
                trace_writer=self.trace_writer,
            ),
        )
        self.finalizer_preparer = ContextPreparer(token_budget)

    def run(self, task: str) -> SubagentResult:
        clean_task = task.strip()
        if not clean_task:
            raise ValueError("subagent task must not be empty.")
        messages: list[dict[str, Any]] = [{"role": "user", "content": clean_task}]
        evidence_summaries: list[str] = []
        tool_calls = 0
        model_calls = 0
        completed_steps = 0
        stop_reason = "max_steps"
        started = time.monotonic()
        self.trace_writer.write_event("subagent_started", task=clean_task)

        for step in range(1, self.max_steps + 1):
            boundary_reason = self._boundary_stop_reason(started)
            if boundary_reason is not None:
                stop_reason = boundary_reason
                break
            completed_steps = step
            excluded_control_tools = {
                "delegate_task",
                "delegate_worktree",
                "runtime_task_status",
                "runtime_task_stop",
            }
            schemas = [
                schema
                for schema in self.tools.schemas()
                if str(schema["function"]["name"]) not in excluded_control_tools
                and self.tools.risk_level(str(schema["function"]["name"])) == RiskLevel.LOW
            ]
            prepared = self.preparer.prepare(
                system_messages=[
                    {
                        "role": "system",
                        "content": _SUBAGENT_SYSTEM_PROMPT
                        + f"\n\n工作目录：{self.workspace}",
                    }
                ],
                messages=messages,
                tools=schemas,
                metadata={"subagent": True, "step": step},
            )
            model_calls += 1
            response = self._call_model(
                prepared.request,
                step=step,
                model_call_index=model_calls,
                phase="explore",
            )
            boundary_reason = self._boundary_stop_reason(started)
            if boundary_reason is not None:
                stop_reason = boundary_reason
                break
            if response.is_final():
                summary = response.final_text or ""
                self.trace_writer.write_event(
                    "subagent_finished",
                    status="completed",
                    steps=step,
                    tool_calls=tool_calls,
                    summary=summary,
                    stop_reason="completed",
                )
                return SubagentResult(status="completed", summary=summary)
            if not response.tool_calls:
                stop_reason = "no_action"
                break
            if tool_calls + len(response.tool_calls) > self.max_tool_calls:
                stop_reason = "tool_budget_exhausted"
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                boundary_reason = self._boundary_stop_reason(started)
                if boundary_reason is not None:
                    stop_reason = boundary_reason
                    break
                tool_calls += 1
                self.trace_writer.write_event(
                    "subagent_tool_called",
                    step=step,
                    tool=call.name,
                    args=call.arguments,
                )
                try:
                    admission = self.tools.admit(call.name, call.arguments)
                    if not admission.allowed or admission.risk_level != RiskLevel.LOW:
                        raise PermissionError("Subagents may use read-only tools only.")
                    result = self.tools.execute_admitted(admission)
                    observation, compression_events = build_observation(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        result=result,
                        artifact_dir=self.artifact_dir,
                        tool_arguments=call.arguments,
                    )
                    for event in compression_events:
                        self.trace_writer.write_event(
                            "subagent_context_compressed",
                            step=step,
                            **event.model_dump(mode="json"),
                        )
                    content = render_tool_result_message(observation)
                    status = str(observation.metadata.get("status") or "ok")
                    evidence_summaries.append(
                        _summarize_subagent_observation(
                            call.name,
                            call.arguments,
                            observation,
                        )
                    )
                except Exception as exc:
                    content = f"Tool {call.name} failed: {type(exc).__name__}: {exc}"
                    status = "error"
                    evidence_summaries.append(
                        _summarize_subagent_error(call.name, call.arguments, exc)
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": content,
                    }
                )
                self.trace_writer.write_event(
                    "subagent_tool_result",
                    step=step,
                    tool=call.name,
                    status=status,
                    truncated=(observation.is_truncated if status != "error" else False),
                    artifact_path=(observation.artifact_path if status != "error" else None),
                )
            if stop_reason in {"cancelled", "timeout"}:
                break

        summary = None
        summary_source = "none"
        if stop_reason not in {"cancelled", "timeout"} and self._boundary_stop_reason(started) is None:
            summary = self._finalize_partial_summary(
                messages,
                step=completed_steps,
                model_call_index=model_calls + 1,
                stop_reason=stop_reason,
            )
            if summary:
                summary_source = "model"
            final_boundary_reason = self._boundary_stop_reason(started)
            if final_boundary_reason is not None:
                stop_reason = final_boundary_reason
                summary = None
                summary_source = "none"
        if not summary:
            summary = _deterministic_evidence_summary(evidence_summaries)
            if evidence_summaries:
                summary_source = "deterministic_evidence"
                self.trace_writer.write_event(
                    "subagent_summary_deterministic",
                    step=completed_steps,
                    chars=len(summary),
                    stop_reason=stop_reason,
                )
            else:
                summary_source = "fixed_stop"
        self.trace_writer.write_event(
            "subagent_finished",
            status="partial",
            steps=completed_steps,
            tool_calls=tool_calls,
            summary=summary,
            stop_reason=stop_reason,
            summary_finalized=summary_source == "model",
            summary_source=summary_source,
        )
        return SubagentResult(status="partial", summary=summary)

    def _call_model(
        self,
        request: ModelRequest,
        *,
        step: int,
        model_call_index: int,
        phase: str,
    ) -> ModelResponse:
        started = time.monotonic()
        estimated_prompt_tokens = request.metadata.get("estimated_prompt_tokens")
        self.trace_writer.write_event(
            "model_call_started",
            step=step,
            model_call_index=model_call_index,
            phase=phase,
            subagent=True,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )

        def on_retry(attempt: int, failure: Any, delay: float) -> None:
            self.trace_writer.write_event(
                "model_retry_scheduled",
                step=step,
                model_call_index=model_call_index,
                phase=phase,
                attempt=attempt,
                kind=failure.kind,
                status_code=failure.status_code,
                retry_after_seconds=failure.retry_after_seconds,
                delay_seconds=delay,
                error=failure.message,
            )

        try:
            response = self.recovery_policy.invoke(
                lambda: self.model_client.call_request(request),
                on_retry=on_retry,
            ).enforce_turn_contract()
        except Exception as exc:
            self.trace_writer.write_event(
                "model_call_finished",
                step=step,
                model_call_index=model_call_index,
                phase=phase,
                subagent=True,
                status="error",
                duration_ms=int((time.monotonic() - started) * 1000),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        duration_ms = int((time.monotonic() - started) * 1000)
        usage = response.usage.model_dump() if response.usage is not None else None
        provider_prompt_tokens = (
            response.usage.input_tokens if response.usage is not None else None
        )
        if (
            isinstance(estimated_prompt_tokens, int)
            and not isinstance(estimated_prompt_tokens, bool)
            and estimated_prompt_tokens > 0
            and isinstance(provider_prompt_tokens, int)
            and not isinstance(provider_prompt_tokens, bool)
            and provider_prompt_tokens > 0
        ):
            signed_error = provider_prompt_tokens - estimated_prompt_tokens
            self.trace_writer.write_event(
                "token_estimate_calibration",
                step=step,
                model_call_index=model_call_index,
                phase=phase,
                subagent=True,
                estimated_prompt_tokens=estimated_prompt_tokens,
                provider_prompt_tokens=provider_prompt_tokens,
                ratio=round(provider_prompt_tokens / estimated_prompt_tokens, 6),
                signed_error_tokens=signed_error,
                absolute_error_tokens=abs(signed_error),
                relative_error=round(abs(signed_error) / provider_prompt_tokens, 6),
                token_estimator_version=request.metadata.get("token_estimator_version"),
            )
        self.trace_writer.write_event(
            "model_call_finished",
            step=step,
            model_call_index=model_call_index,
            phase=phase,
            subagent=True,
            status="ok",
            duration_ms=duration_ms,
            response_kind=response.kind(),
            final_text_present=bool(response.final_text),
            tool_calls=len(response.tool_calls),
            stop_reason=response.stop_reason,
            input_tokens=(response.usage.input_tokens if response.usage else None),
            output_tokens=(response.usage.output_tokens if response.usage else None),
            total_tokens=(response.usage.total_tokens if response.usage else None),
            cached_input_tokens=(
                response.usage.cached_input_tokens if response.usage else None
            ),
        )
        self.trace_writer.write_event(
            "model_response",
            step=step,
            model_call_index=model_call_index,
            phase=phase,
            subagent=True,
            response_kind=response.kind(),
            final_text_present=bool(response.final_text),
            invalid_reason=response.invalid_reason,
            stop_reason=response.stop_reason,
            tool_calls=[call.model_dump(mode="json") for call in response.tool_calls],
            usage=usage,
        )
        return response

    def _finalize_partial_summary(
        self,
        messages: list[dict[str, Any]],
        *,
        step: int,
        model_call_index: int,
        stop_reason: str,
    ) -> str | None:
        try:
            prepared = self.finalizer_preparer.prepare(
                system_messages=[
                    {
                        "role": "system",
                        "content": (
                            _SUBAGENT_SYSTEM_PROMPT
                            + f"\n\n工作目录：{self.workspace}"
                            + "\n\n"
                            + _SUBAGENT_FINAL_ONLY_PROMPT
                        ),
                    }
                ],
                messages=messages,
                tools=[],
                metadata={
                    "subagent": True,
                    "step": step,
                    "phase": "finalize",
                    "final_only": True,
                },
            )
            response = self._call_model(
                prepared.request,
                step=step,
                model_call_index=model_call_index,
                phase="finalize",
            )
        except Exception as exc:
            self.trace_writer.write_event(
                "subagent_summary_failed",
                step=step,
                reason="model_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None
        if not response.is_final():
            self.trace_writer.write_event(
                "subagent_summary_failed",
                step=step,
                reason="non_final_response",
                response_kind=response.kind(),
                requested_tool_calls=len(response.tool_calls),
            )
            return None
        summary = (response.final_text or "").strip()
        if not summary:
            return None
        self.trace_writer.write_event(
            "subagent_summary_finalized",
            step=step,
            chars=len(summary),
            stop_reason=stop_reason,
        )
        return summary

    def _boundary_stop_reason(self, started: float) -> str | None:
        if self.cancellation_token.is_cancelled:
            return "cancelled"
        if time.monotonic() - started >= self.max_duration_seconds:
            return "timeout"
        return None


def _deterministic_evidence_summary(evidence: list[str]) -> str:
    retained = [item for item in evidence if item][:12]
    if not retained:
        return "子任务未完整完成，当前没有足够证据形成结论。"
    if len(evidence) > len(retained):
        retained.append(f"- 另有 {len(evidence) - len(retained)} 条工具结果未展开。")
    return "\n".join(
        [
            "子任务未完整完成。基于已有工具结果：",
            *retained,
        ]
    )[:3000]


def _summarize_subagent_observation(
    tool_name: str,
    arguments: dict[str, Any],
    observation: ContextObservation,
) -> str:
    metadata = observation.metadata
    status = str(metadata.get("status") or "ok")
    if tool_name == "search":
        return _summarize_search_observation(arguments, metadata, status)
    if tool_name == "read":
        source = str(arguments.get("source") or "workspace")
        target = str(arguments.get("target") or "")
        label = f'read(source={source}) target="{target}"'
        if status != "ok":
            return f"- {label}：失败，{_single_line(observation.content, 240)}"
        path = str(metadata.get("path") or target)
        start = metadata.get("start_line")
        end = metadata.get("end_line")
        total = metadata.get("total_lines")
        total_chars = metadata.get("total_chars")
        range_text = f"，已读取行 {start}-{end}" if start is not None and end is not None else ""
        total_text = f"，共 {total} 行" if total is not None else ""
        char_text = f"、{total_chars} 字符" if total_chars is not None else ""
        guidance = str(metadata.get("guidance") or "").strip()
        guidance_text = f"；{guidance}" if guidance else ""
        return (
            f'- {label}：已确认路径 "{path}" 存在{range_text}{total_text}{char_text}'
            f"{guidance_text}。"
        )

    label = _tool_call_label(tool_name, arguments)
    detail = observation.summary or observation.output_preview or observation.content
    return f"- {label} [{status}]：{_single_line(detail, 240)}"


def _summarize_search_observation(
    arguments: dict[str, Any],
    metadata: dict[str, Any],
    status: str,
) -> str:
    kind = str(arguments.get("kind") or "files")
    path = str(arguments.get("path") or ".")
    query = str(arguments.get("query") or "*")
    label = f'search(kind={kind}) path="{path}" query="{query}"'
    if status != "ok":
        return f"- {label}：失败。"
    files = [str(item) for item in metadata.get("files") or []]
    if kind == "files":
        if metadata.get("exists") is False:
            return f"- {label}：路径不存在。"
        if metadata.get("is_directory") is False:
            return f"- {label}：路径存在，但不是目录。"
        if files:
            details = {
                str(item.get("path")): item
                for item in metadata.get("file_details") or []
                if isinstance(item, dict) and item.get("path")
            }
            samples = []
            for file_path in files[:6]:
                detail = details.get(file_path) or {}
                size_bytes = detail.get("size_bytes")
                estimated_lines = detail.get("estimated_lines")
                suffix = (
                    f" ({size_bytes} bytes, ~{estimated_lines} lines)"
                    if size_bytes is not None and estimated_lines is not None
                    else ""
                )
                samples.append(file_path + suffix)
            sample_text = ", ".join(samples)
            if len(files) > len(samples):
                sample_text += f"，另有 {len(files) - len(samples)} 项"
            return f"- {label}：找到 {len(files)} 个文件：{sample_text}。"
        return f"- {label}：路径存在，当前范围内无匹配。"

    match_count = int(metadata.get("match_count") or 0)
    if match_count:
        suffix = f"，涉及路径：{_bounded_values(files)}" if files else ""
        return f"- {label}：找到 {match_count} 处匹配{suffix}。"
    return f"- {label}：未找到文本匹配。"


def _summarize_subagent_error(
    tool_name: str,
    arguments: dict[str, Any],
    error: Exception,
) -> str:
    label = _tool_call_label(tool_name, arguments)
    return f"- {label}：失败，{type(error).__name__}: {_single_line(str(error), 220)}"


def _tool_call_label(tool_name: str, arguments: dict[str, Any]) -> str:
    compact = ", ".join(
        f"{key}={value}"
        for key, value in list(arguments.items())[:4]
        if value not in (None, "", [], {})
    )
    return f"{tool_name}({compact})" if compact else tool_name


def _bounded_values(values: list[str], limit: int = 6) -> str:
    rendered = ", ".join(values[:limit])
    if len(values) > limit:
        rendered += f"，另有 {len(values) - limit} 项"
    return rendered or "无"


def _single_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


_SUBAGENT_SYSTEM_PROMPT = """你是 MiniCode 的只读代码分析子 Agent。

完成父 Agent 委派的一个独立分析问题。你的目标是回答这个问题，不是理解整个相关模块或穷举仓库。
使用只读工具核对事实，不修改文件，不运行命令，不继续委派。文件或实现位置未知时先搜索定位，再读取必要范围；已知短文件或明确位置可以直接读取。
优先寻找能直接回答问题的定义、调用点和配置，不为完整性扫描整个目录。每获得一批证据后判断是否已经足够回答，足够时立即停止探索并返回结论。
如果当前边界不足以完成全部内容，优先保留最高价值的已确认事实，明确尚未完成的部分，不继续扩大范围。
基于实际工具结果给出简洁、可直接使用的结论，不要输出过程性计划。"""


_SUBAGENT_FINAL_ONLY_PROMPT = """停止继续探索，不得请求或假装使用任何工具。
仅根据当前已有 Tool Result 给出可直接供父 Agent 使用的部分总结：先写已经确认的语义结论，再写仍未完成或无法确认的部分。不要只罗列调用过的工具、读取过的路径或搜索数量；这些只能作为结论的证据来源。"""
