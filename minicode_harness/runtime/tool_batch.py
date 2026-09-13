"""One-response Tool Call batch orchestration for AgentLoop."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any

from minicode_harness.models import ModelResponse, NormalizedToolCall
from minicode_harness.runtime.progress_policy import ProgressGuidance
from minicode_harness.runtime.tool_runtime import ToolExecutionOutcome
from minicode_harness.tools.semantics import is_task_tool


class ToolBatchExecutor:
    """Schedule one model response's Tool Calls without duplicating ToolRuntime policy."""

    def execute(
        self,
        loop: Any,
        step: int,
        response: ModelResponse,
        *,
        tool_names: tuple[str, ...],
    ) -> Any | None:
        if not response.tool_calls:
            return loop._finish_stopped_run(
                step=step,
                reason="no_tool_calls",
                final_text=response.final_text,
            )

        remaining = loop.config.max_tool_calls - loop.tool_call_count
        if loop.cancellation_token.is_cancelled:
            return loop._finish_cancelled_run(step=step)

        loop._append_assistant_tool_calls(response)
        executable_tool_calls: list[NormalizedToolCall] = []
        skipped_tool_calls: list[NormalizedToolCall] = []
        budget_left = max(0, remaining)
        for tool_call in response.tool_calls:
            if not loop.tools.counts_against_tool_budget(tool_call.name):
                executable_tool_calls.append(tool_call)
            elif budget_left > 0:
                executable_tool_calls.append(tool_call)
                budget_left -= 1
            else:
                skipped_tool_calls.append(tool_call)

        if not executable_tool_calls and skipped_tool_calls:
            loop._append_tool_budget_results(step, skipped_tool_calls)
            loop._persist_and_checkpoint(
                step,
                reason="max_tool_calls",
                status="stopped",
            )
            return loop._finish_stopped_run(
                step=step,
                reason="max_tool_calls",
                final_text=None,
                rollback=False,
            )

        parallel_tool_outcomes = (
            self._execute_parallel(
                loop,
                step,
                executable_tool_calls,
                available_tool_names=tool_names,
            )
            if len(executable_tool_calls) > 1
            else None
        )
        pending_guidance: list[ProgressGuidance] = []
        batch_stop_reason: str | None = None
        batch_cancelled = False
        for tool_index, tool_call in enumerate(executable_tool_calls):
            if loop.tools.counts_against_tool_budget(tool_call.name):
                loop.tool_call_count += 1
            outcome = (
                parallel_tool_outcomes[tool_index]
                if parallel_tool_outcomes is not None
                else loop.tool_runtime.execute(
                    step=step,
                    tool_call=tool_call,
                    available_tool_names=tool_names,
                    workspace_generation=loop.workspace_generation,
                )
            )
            loop._emit_lifecycle_hook(
                "post_tool_use",
                step=step,
                payload={
                    "tool_call": tool_call,
                    "observation": outcome.observation,
                },
            )
            loop._emit_tool_call_finished(step, outcome.observation)
            loop.observations.append(outcome.observation)
            loop._append_tool_result_message(tool_call, outcome.observation)
            loop._record_modified_files(outcome.modified_files or [])
            loop._advance_verification_state(tool_call, outcome)
            guidance = loop.progress_policy.after_tool(
                step=step,
                tool_call=tool_call,
                outcome=outcome,
                run_state=loop.run_state,
                workspace_generation=loop.workspace_generation,
                modified_files=loop.modified_files,
            )
            loop._record_run_state(step, tool_call, outcome)
            if guidance is not None:
                pending_guidance.append(guidance)
            loop._persist_and_checkpoint(
                step,
                reason=outcome.stop_reason or f"tool:{tool_call.name}",
                status="stopped" if outcome.stop_reason else "running",
            )
            if outcome.stop_reason:
                if parallel_tool_outcomes is None:
                    return loop._finish_stopped_run(
                        step=step,
                        reason=outcome.stop_reason,
                        final_text=None,
                        rollback=False,
                    )
                if batch_stop_reason is None:
                    batch_stop_reason = outcome.stop_reason
            if loop.cancellation_token.is_cancelled:
                if parallel_tool_outcomes is None:
                    loop._append_cancelled_tool_results(
                        step,
                        executable_tool_calls[tool_index + 1 :] + skipped_tool_calls,
                    )
                    return loop._finish_cancelled_run(step=step)
                batch_cancelled = True

        if skipped_tool_calls:
            loop._append_tool_budget_results(step, skipped_tool_calls)
            loop._persist_and_checkpoint(
                step,
                reason="tool_budget_partial_group",
            )

        if batch_stop_reason is not None:
            return loop._finish_stopped_run(
                step=step,
                reason=batch_stop_reason,
                final_text=None,
                rollback=False,
            )
        if batch_cancelled:
            return loop._finish_cancelled_run(step=step)

        for guidance in pending_guidance:
            loop.user_turn.append_message(
                {"role": "user", "content": guidance.message}
            )
            loop.trace_writer.write_event(
                "progress_stagnation_nudge",
                step=step,
                level=guidance.level,
                guidance=guidance.message,
                non_write_calls_since_progress=(
                    loop.progress_policy.non_write_calls_since_progress
                ),
                remaining_steps=max(0, loop.config.max_steps - step),
            )
        if pending_guidance:
            loop._persist_and_checkpoint(step, reason="progress_guidance")
        return None

    def _execute_parallel(
        self,
        loop: Any,
        step: int,
        tool_calls: list[NormalizedToolCall],
        *,
        available_tool_names: tuple[str, ...],
    ) -> list[ToolExecutionOutcome]:
        started = time.monotonic()
        loop.trace_writer.write_event(
            "tool_batch_started",
            step=step,
            count=len(tool_calls),
            tools=[tool_call.name for tool_call in tool_calls],
        )
        workspace_generation = loop.workspace_generation
        memory_read_admission = loop.tool_runtime.plan_memory_read_batch(tool_calls)
        outcomes: list[ToolExecutionOutcome | None] = [None] * len(tool_calls)
        admitted_indexes = [
            index
            for index, tool_call in enumerate(tool_calls)
            if memory_read_admission.get(tool_call.id) is not False
        ]
        serial_indexes = [
            index
            for index in admitted_indexes
            if is_task_tool(tool_calls[index].name)
        ]
        parallel_indexes = [
            index
            for index in admitted_indexes
            if index not in serial_indexes
        ]
        if parallel_indexes:
            with ThreadPoolExecutor(
                max_workers=len(parallel_indexes),
                thread_name_prefix="minicode-tool",
            ) as pool:
                futures = {
                    index: pool.submit(
                        loop.tool_runtime.execute,
                        step=step,
                        tool_call=tool_calls[index],
                        available_tool_names=available_tool_names,
                        workspace_generation=workspace_generation,
                        memory_read_allowed=memory_read_admission.get(
                            tool_calls[index].id
                        ),
                    )
                    for index in parallel_indexes
                }
                for index in serial_indexes:
                    outcomes[index] = loop.tool_runtime.execute(
                        step=step,
                        tool_call=tool_calls[index],
                        available_tool_names=available_tool_names,
                        workspace_generation=workspace_generation,
                    )
                for index, future in futures.items():
                    outcomes[index] = future.result()
        else:
            for index in serial_indexes:
                outcomes[index] = loop.tool_runtime.execute(
                    step=step,
                    tool_call=tool_calls[index],
                    available_tool_names=available_tool_names,
                    workspace_generation=workspace_generation,
                )
        for index, tool_call in enumerate(tool_calls):
            if memory_read_admission.get(tool_call.id) is not False:
                continue
            outcomes[index] = loop.tool_runtime.execute(
                step=step,
                tool_call=tool_call,
                available_tool_names=available_tool_names,
                workspace_generation=workspace_generation,
                memory_read_allowed=False,
            )
        loop.trace_writer.write_event(
            "tool_batch_finished",
            step=step,
            count=len(tool_calls),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return [outcome for outcome in outcomes if outcome is not None]
