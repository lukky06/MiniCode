"""Single-model-step orchestration for the bounded Agent Loop.

The runner owns request preparation, provider invocation, streaming coordination,
and bounded model-side recovery for one loop step. It deliberately receives the
owning loop as a narrow callback/state surface instead of becoming a second Run
controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import TYPE_CHECKING, Any

from minicode_harness.context import PromptBudgetExceeded
from minicode_harness.context.builder import PROMPT_BUILDER_VERSION
from minicode_harness.context.token import estimate_tokens
from minicode_harness.models import ModelProviderError, ModelRequest, ModelResponse
from minicode_harness.output import ContextUsage, FinalAnswerXmlStreamEmitter
from minicode_harness.runtime.recovery import ModelCallFailed, PromptTooLongFailure

if TYPE_CHECKING:
    from minicode_harness.loop import AgentLoop


@dataclass(frozen=True)
class PreparedModelStep:
    request: ModelRequest
    tool_names: tuple[str, ...]
    forced_final_only: bool


@dataclass(frozen=True)
class ModelStepOutcome:
    response: ModelResponse | None = None
    result: Any | None = None
    should_continue: bool = False


class ModelStepRunner:
    """Prepare and request exactly one model step for an AgentLoop."""

    def prepare(self, loop: AgentLoop, step: int) -> PreparedModelStep | Any:
        forced_final_only = loop._force_final_only_next_call
        final_only_step = forced_final_only or (
            loop.config.reserve_final_step and step == loop.config.max_steps
        )
        if forced_final_only:
            loop._force_final_only_next_call = False
            loop.trace_writer.write_event(
                "final_answer_rewrite_step",
                step=step,
                modified_files=list(loop.modified_files),
            )
        elif final_only_step:
            loop._append_runtime_recovery(
                "工具阶段已经结束。这是最后一个模型调用，不要再调用任何工具。"
                "请基于现有源码证据和已经完成的修改直接给出最终答案；"
                "如无法验证，请明确说明验证未运行。"
            )
            loop.trace_writer.write_event(
                "final_answer_step_reserved",
                step=step,
                modified_files=list(loop.modified_files),
            )

        model_call_index = loop.user_turn.begin_model_call()
        build_started = time.monotonic()
        if final_only_step:
            tool_schemas: list[dict[str, Any]] = []
        else:
            hidden_tool_names = set(loop.config.hidden_tool_names)
            collaboration_tool_names = (
                set(loop.tools.readonly_tool_names())
                if loop.collaboration_mode.value == "plan"
                else None
            )
            tool_schemas = [
                schema
                for schema in loop.tools.schemas()
                if str(schema["function"]["name"]) not in hidden_tool_names
                and (
                    collaboration_tool_names is None
                    or str(schema["function"]["name"]) in collaboration_tool_names
                )
            ]
        tool_names = tuple(
            str(schema["function"]["name"]) for schema in tool_schemas
        )
        (
            tool_schema_chars,
            tool_schema_tokens,
            largest_tool_schema,
        ) = _tool_schema_metrics(tool_schemas)
        schema_hash, schema_snapshot = loop._trace_tool_schema_if_changed(
            step,
            tool_schemas,
        )
        loop._emit_lifecycle_hook(
            "before_context_build",
            step=step,
            payload={"tool_names": list(tool_names)},
        )

        repository_rules = loop._repository_rules_snapshot()
        built = loop.context_builder.build(
            available_skills=loop.available_skills,
            workspace=loop.workspace,
            repository_rules=repository_rules.content,
            long_term_context=loop.long_term_context,
            repository_structure_card=loop._repository_structure_card(),
            streaming_enabled=loop.stream_model,
            current_step=step,
            max_steps=loop.config.max_steps,
            current_tool_calls=loop.tool_call_count,
            max_tool_calls=loop.config.max_tool_calls,
            collaboration_mode=loop.collaboration_mode.value,
        )
        system_messages = [
            message for message in built.messages if message.get("role") == "system"
        ]
        try:
            prepared = loop.context_preparer.prepare(
                system_messages=system_messages,
                messages=loop.user_turn.snapshot_messages(),
                tools=tool_schemas,
                compaction_state=loop.compaction_state,
                tool_effects=loop.tools.history_effects(),
                task_projection_provider=lambda: (
                    loop.task,
                    loop.task_store.open_rows(),
                ),
                metadata={
                    "run_id": loop.run_id,
                    "step": step,
                    "user_turn_id": loop.user_turn.turn_id,
                    "model_call_index": model_call_index,
                    "max_output_tokens": loop.output_budget.current,
                    "reasoning_effort": loop.config.reasoning_effort,
                },
            )
        except PromptBudgetExceeded as exc:
            loop.trace_writer.write_event(
                "context_budget_exceeded",
                step=step,
                token_estimate=exc.token_estimate,
                hard_token_limit=exc.hard_token_limit,
                source_tokens=exc.source_tokens,
                error=str(exc),
            )
            loop._emit_lifecycle_hook(
                "error",
                step=step,
                payload={"error": exc, "kind": "prompt_budget_exceeded"},
            )
            return loop._finish_stopped_run(
                step=step,
                reason="prompt_budget_exceeded",
                final_text=None,
            )

        loop.compaction_state = prepared.compaction_update.model_copy(deep=True)
        loop.tool_runtime.synchronize_reuse_with_messages(
            prepared.request.messages,
            observations=loop.observations,
            run_state=loop.run_state,
            workspace_generation=loop.workspace_generation,
        )
        duration_ms = int((time.monotonic() - build_started) * 1000)
        loop.output_sink.context_built(
            usage=ContextUsage(
                build_duration_ms=duration_ms,
                token_estimate=prepared.token_estimate,
                context_window=prepared.context_window,
                prompt_budget=prepared.prompt_budget,
                reserved_output=prepared.reserved_output,
            )
        )
        compression_events = [
            *built.compression_events,
            *prepared.compression_events,
        ]
        for event in compression_events:
            loop.trace_writer.write_event(
                "context_compressed",
                step=step,
                **event.model_dump(mode="json"),
            )
        loop._emit_lifecycle_hook(
            "after_context_build",
            step=step,
            payload={
                "token_estimate": prepared.token_estimate,
                "message_count": prepared.message_count,
            },
        )
        loop.trace_writer.write_event(
            "context_built",
            step=step,
            user_turn_id=loop.user_turn.turn_id,
            model_call_index=model_call_index,
            message_count=prepared.message_count,
            token_estimate=prepared.token_estimate,
            context_window=prepared.context_window,
            prompt_budget=prepared.prompt_budget,
            reserved_output=prepared.reserved_output,
            request_tokens_before_compaction=prepared.request_tokens_before_compaction,
            request_tokens_after_argument_compaction=(
                prepared.request_tokens_after_argument_compaction
            ),
            budget_usage_ratio=prepared.budget_usage_ratio,
            history_groups_compacted=prepared.history_groups_compacted,
            source_tokens=prepared.source_tokens,
            reactive_compaction_count=loop._reactive_compactions,
            token_estimator_version=prepared.token_estimator_version,
            compression_count=len(compression_events),
            available_skills=[skill.name for skill in loop.available_skills],
            prompt_prefix_hash=built.prompt_prefix_hash,
            prompt_prefix_tokens=built.prompt_prefix_tokens,
            prompt_builder_version=PROMPT_BUILDER_VERSION,
            context_architecture="canonical_messages",
            tool_count=len(tool_schemas),
            tool_schema_chars=tool_schema_chars,
            tool_schema_tokens=tool_schema_tokens,
            largest_tool_schema=largest_tool_schema,
            tool_schema_hash=schema_hash,
            tool_schema_snapshot=schema_snapshot,
            available_tools=list(tool_names),
            collaboration_mode=loop.collaboration_mode.value,
            duration_ms=duration_ms,
            tool_calls_used=loop.tool_call_count,
            remaining_tool_calls=max(
                0,
                loop.config.max_tool_calls - loop.tool_call_count,
            ),
        )
        return PreparedModelStep(
            request=prepared.request,
            tool_names=tool_names,
            forced_final_only=forced_final_only,
        )

    def request(
        self,
        loop: AgentLoop,
        step: int,
        prepared: PreparedModelStep,
    ) -> ModelStepOutcome:
        loop._emit_lifecycle_hook(
            "before_model_call",
            step=step,
            payload={"request": prepared.request},
        )
        if loop.cancellation_token.is_cancelled:
            return ModelStepOutcome(result=loop._finish_cancelled_run(step=step))
        try:
            response = self._call_model(loop, step, prepared.request)
        except PromptTooLongFailure as exc:
            if (
                loop._reactive_compactions
                >= loop.recovery_policy.config.max_reactive_compactions
            ):
                loop._emit_lifecycle_hook(
                    "error",
                    step=step,
                    payload={"error": exc, "kind": "prompt_too_long"},
                )
                return ModelStepOutcome(
                    result=loop._finish_stopped_run(
                        step=step,
                        reason="prompt_too_long",
                        final_text=None,
                    )
                )
            compacted_state, event = loop.context_preparer.reactive_compact(
                loop.user_turn.snapshot_messages(),
                compaction_state=loop.compaction_state,
                tool_effects=loop.tools.history_effects(),
                task_projection_provider=lambda: (
                    loop.task,
                    loop.task_store.open_rows(),
                ),
            )
            loop.compaction_state = compacted_state.model_copy(deep=True)
            loop._reactive_compactions += 1
            if event is not None:
                loop.trace_writer.write_event(
                    "context_compressed",
                    step=step,
                    **event.model_dump(mode="json"),
                )
            loop.trace_writer.write_event(
                "model_recovery",
                step=step,
                action="reactive_compact",
                attempt=loop._reactive_compactions,
                error=str(exc),
            )
            loop._persist_and_checkpoint(step, reason="reactive_prompt_compact")
            return ModelStepOutcome(should_continue=True)
        except ModelCallFailed as exc:
            loop._emit_lifecycle_hook(
                "error",
                step=step,
                payload={"error": exc, "kind": exc.failure.kind},
            )
            loop.trace_writer.write_event(
                "model_call_failed",
                step=step,
                kind=exc.failure.kind,
                status_code=exc.failure.status_code,
                error=exc.failure.message,
            )
            return ModelStepOutcome(
                result=loop._finish_stopped_run(
                    step=step,
                    reason=f"model_{exc.failure.kind}",
                    final_text=None,
                )
            )

        loop._emit_lifecycle_hook(
            "after_model_response",
            step=step,
            payload={"response": response},
        )
        if loop.cancellation_token.is_cancelled:
            return ModelStepOutcome(result=loop._finish_cancelled_run(step=step))
        if prepared.forced_final_only and self.response_was_truncated(response):
            return ModelStepOutcome(
                result=loop._finish_stopped_run(
                    step=step,
                    reason="final_answer_rewrite_truncated",
                    final_text=response.final_text,
                    rollback=False,
                )
            )
        if self.response_was_truncated(response):
            if loop._recover_truncated_output(step, response):
                loop._persist_and_checkpoint(step, reason="output_truncation_recovery")
                return ModelStepOutcome(should_continue=True)
            return ModelStepOutcome(
                result=loop._finish_stopped_run(
                    step=step,
                    reason="output_truncated",
                    final_text=response.final_text,
                    rollback=False,
                )
            )
        if prepared.forced_final_only and not response.is_final():
            return ModelStepOutcome(
                result=loop._finish_stopped_run(
                    step=step,
                    reason="final_answer_rewrite_failed",
                    final_text=response.final_text,
                    rollback=False,
                )
            )
        if response.kind() == "invalid":
            loop._record_invalid_model_response(step, response)
            loop._append_runtime_recovery(
                "模型响应无效。请返回原生工具调用，或在任务完成后返回最终答案。"
            )
            loop._persist_and_checkpoint(step, reason="invalid_model_response")
            return ModelStepOutcome(should_continue=True)
        return ModelStepOutcome(response=response)

    def _call_model(
        self,
        loop: AgentLoop,
        step: int,
        request: ModelRequest,
    ) -> ModelResponse:
        started = time.monotonic()
        streamed_parts: list[str] = []
        loop.trace_writer.write_event(
            "model_call_started",
            step=step,
            user_turn_id=loop.user_turn.turn_id,
            model_call_index=loop.user_turn.model_call_count,
            streaming=loop.stream_model,
        )
        loop._current_stream_emitted_final_text = False
        emitter: FinalAnswerXmlStreamEmitter | None = None
        if loop.stream_model:
            loop.output_sink.model_stream_started()

        def operation() -> ModelResponse:
            nonlocal emitter
            if loop.stream_model:
                emitter = FinalAnswerXmlStreamEmitter(loop.output_sink)

                def on_text_delta(text: str) -> None:
                    streamed_parts.append(text)
                    assert emitter is not None
                    emitter.feed(text)
                    loop._current_stream_emitted_final_text = emitter.emitted

                try:
                    return loop.model_client.stream_request(
                        request,
                        on_text_delta=on_text_delta,
                        on_reasoning_delta=loop.output_sink.model_reasoning_delta,
                    )
                except Exception as exc:
                    if emitter.emitted:
                        raise ModelProviderError(
                            "Streaming failed after visible final-answer text was emitted; "
                            "automatic retry was suppressed to prevent duplicate output.",
                            kind="unknown",
                            retryable=False,
                        ) from exc
                    raise
            return loop.model_client.call_request(request)

        def on_retry(attempt: int, failure: Any, delay: float) -> None:
            loop.trace_writer.write_event(
                "model_retry_scheduled",
                step=step,
                attempt=attempt,
                kind=failure.kind,
                status_code=failure.status_code,
                retry_after_seconds=failure.retry_after_seconds,
                delay_seconds=delay,
                error=failure.message,
            )

        response = loop.recovery_policy.invoke(operation, on_retry=on_retry)
        response = response.enforce_turn_contract()
        if emitter is not None and (
            response.kind() == "final_text" or self.response_was_truncated(response)
        ):
            emitter.flush_partial()
            loop._current_stream_emitted_final_text = emitter.emitted
        streamed_final_chars = emitter.emitted_chars if emitter is not None else 0
        duration_ms = int((time.monotonic() - started) * 1000)
        estimated_prompt_tokens = request.metadata.get("estimated_prompt_tokens")
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
            loop.trace_writer.write_event(
                "token_estimate_calibration",
                step=step,
                provider=loop.provider,
                model=loop.model,
                estimated_prompt_tokens=estimated_prompt_tokens,
                provider_prompt_tokens=provider_prompt_tokens,
                ratio=round(provider_prompt_tokens / estimated_prompt_tokens, 6),
                signed_error_tokens=signed_error,
                absolute_error_tokens=abs(signed_error),
                relative_error=round(abs(signed_error) / provider_prompt_tokens, 6),
                token_estimator_version=request.metadata.get("token_estimator_version"),
            )
        loop.trace_writer.write_event(
            "model_call_finished",
            step=step,
            user_turn_id=loop.user_turn.turn_id,
            model_call_index=loop.user_turn.model_call_count,
            duration_ms=duration_ms,
            final_text_present=bool(response.final_text),
            tool_calls=len(response.tool_calls),
            streamed_text_chars=sum(len(part) for part in streamed_parts),
            streamed_text_suppressed=response.kind() == "tool_calls",
            streamed_final_text_chars=streamed_final_chars,
            stop_reason=response.stop_reason,
        )
        loop.trace_writer.write_event(
            "model_response",
            step=step,
            response_kind=response.kind(),
            final_text_present=bool(response.final_text),
            assistant_commentary_present=bool(response.assistant_commentary),
            reasoning_content_chars=len(response.reasoning_content or ""),
            invalid_reason=response.invalid_reason,
            stop_reason=response.stop_reason,
            output_protocol=response.output_protocol,
            structured_final_fields=sorted(response.structured_final.keys()),
            tool_calls=[call.model_dump(mode="json") for call in response.tool_calls],
            usage=response.usage.model_dump() if response.usage else None,
        )
        if response.assistant_commentary:
            loop.trace_writer.write_event(
                "assistant_commentary_suppressed",
                step=step,
                chars=len(response.assistant_commentary),
            )
        return response

    @staticmethod
    def response_was_truncated(response: ModelResponse) -> bool:
        reason = (response.stop_reason or "").strip().lower()
        return reason in {"length", "max_tokens", "max_output_tokens"}


def _tool_schema_metrics(
    tool_schemas: list[dict[str, Any]],
) -> tuple[int, int, dict[str, Any] | None]:
    payload = json.dumps(
        tool_schemas,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    largest: dict[str, Any] | None = None
    for schema in tool_schemas:
        schema_payload = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        item = {
            "name": str(schema["function"]["name"]),
            "chars": len(schema_payload),
            "tokens": estimate_tokens(schema_payload),
        }
        if largest is None or item["tokens"] > largest["tokens"]:
            largest = item
    return len(payload), estimate_tokens(payload), largest
