"""Optional bounded execution-progress guidance for AgentLoop."""

from __future__ import annotations

from dataclasses import dataclass

from minicode_harness.context import ContextObservation, RunState
from minicode_harness.models import NormalizedToolCall
from minicode_harness.runtime.tool_runtime import ToolExecutionOutcome


_STAGNATION_WRITE_TOOLS = {
    "edit",
    "write",
    "apply_patch",
    "delegate_worktree",
}
_STAGNATION_IGNORED_TOOLS = {
    "task",
    "runtime_task_status",
    "runtime_task_stop",
}
_PROGRESS_STAGNATION_FIRST_MESSAGE = """[Runtime guidance: progress stagnation 1]
你已经连续执行多次非修改操作，但当前任务要求修改代码。

请先明确预期不变量、该不变量第一次被破坏的位置，以及一个能够直接证伪首要根因假设的诊断。已有证据能够完整解释现象时，立即执行最小修改，不要继续调查错误发生后的下游调用链。"""
_FAILED_VERIFICATION_REASSESSMENT_MESSAGE = """[Runtime guidance: failed verification reassessment]
当前补丁未通过验证。先检查失败状态是否在进入修改函数前就已存在，以及当前补丁修复的是最早错误点还是下游表现。只获取一项能够证伪当前根因的关键证据，然后调整最小补丁；不要通过放宽或改写测试证明修复正确。"""
_BUDGET_PROGRESS_NO_EDIT_MESSAGE = """[Runtime guidance: execution budget 80%]
当前任务已使用大部分模型调用预算，但尚未修改文件。停止新的旁支调查；如果已有函数级复现证明数据在某个转换中丢失或错误变化，下一步直接修改该转换并运行对应层级的聚焦测试。"""
_BUDGET_PROGRESS_FAILED_VERIFICATION_MESSAGE = """[Runtime guidance: execution budget 80%]
当前任务已使用大部分模型调用预算，且补丁仍未通过验证。不要继续增加测试或修补下游异常；重新运行最初失败的不变量，确认错误在当前修改点之前还是之后发生，然后完成最小修正。"""


@dataclass(frozen=True)
class ProgressGuidance:
    level: int
    message: str
    reason: str


class DisabledProgressPolicy:
    """No-op policy used by ordinary runs."""

    @property
    def non_write_calls_since_progress(self) -> int:
        return 0

    def after_tool(
        self,
        *,
        step: int,
        tool_call: NormalizedToolCall,
        outcome: ToolExecutionOutcome,
        run_state: RunState,
        workspace_generation: int,
        modified_files: list[str],
    ) -> ProgressGuidance | None:
        _ = (
            step,
            tool_call,
            outcome,
            run_state,
            workspace_generation,
            modified_files,
        )
        return None



class BoundedProgressPolicy:
    """Emit at most two stagnation nudges plus bounded recovery guidance."""

    def __init__(
        self,
        *,
        max_steps: int,
        start_step: int,
        observations: list[ContextObservation],
    ) -> None:
        self.max_steps = max_steps
        self._consecutive_read_only_calls = 0
        self._stagnation_nudges = 0
        self._failed_verification_nudge_generation: int | None = None
        self._budget_progress_nudge_emitted = start_step * 5 >= max_steps * 4
        self._restore(observations)

    @property
    def non_write_calls_since_progress(self) -> int:
        return self._consecutive_read_only_calls

    def after_tool(
        self,
        *,
        step: int,
        tool_call: NormalizedToolCall,
        outcome: ToolExecutionOutcome,
        run_state: RunState,
        workspace_generation: int,
        modified_files: list[str],
    ) -> ProgressGuidance | None:
        guidance = self._stagnation_guidance(
            step=step,
            tool_call=tool_call,
            outcome=outcome,
            run_state=run_state,
            workspace_generation=workspace_generation,
            has_modified_files=bool(modified_files),
        )
        if guidance is not None:
            return guidance
        return self._budget_guidance(
            step=step,
            run_state=run_state,
            has_modified_files=bool(modified_files),
        )

    def _restore(self, observations: list[ContextObservation]) -> None:
        count = 0
        for observation in observations:
            if observation.tool_name in _STAGNATION_WRITE_TOOLS:
                count = 0
            elif observation.tool_name not in _STAGNATION_IGNORED_TOOLS:
                count += 1
        self._consecutive_read_only_calls = count
        self._stagnation_nudges = 2 if count >= 12 else 1 if count >= 8 else 0

    def _stagnation_guidance(
        self,
        *,
        step: int,
        tool_call: NormalizedToolCall,
        outcome: ToolExecutionOutcome,
        run_state: RunState,
        workspace_generation: int,
        has_modified_files: bool,
    ) -> ProgressGuidance | None:
        if tool_call.name in _STAGNATION_WRITE_TOOLS:
            self._consecutive_read_only_calls = 0
            return None
        if tool_call.name not in _STAGNATION_IGNORED_TOOLS:
            self._consecutive_read_only_calls += 1

        returncode = outcome.observation.metadata.get("returncode")
        if (
            has_modified_files
            and tool_call.name == "run_command"
            and run_state.verification.status == "failed"
            and returncode not in (None, 0)
            and self._failed_verification_nudge_generation != workspace_generation
        ):
            self._failed_verification_nudge_generation = workspace_generation
            return ProgressGuidance(
                level=3,
                message=_FAILED_VERIFICATION_REASSESSMENT_MESSAGE,
                reason="failed_verification",
            )

        if self._stagnation_nudges >= 2:
            return None
        if self._stagnation_nudges == 0 and self._consecutive_read_only_calls >= 8:
            self._stagnation_nudges = 1
            return ProgressGuidance(
                level=1,
                message=_PROGRESS_STAGNATION_FIRST_MESSAGE,
                reason="read_stagnation_first",
            )
        if self._stagnation_nudges == 1 and self._consecutive_read_only_calls >= 12:
            self._stagnation_nudges = 2
            return ProgressGuidance(
                level=2,
                message=_second_stagnation_message(
                    step=step,
                    max_steps=self.max_steps,
                ),
                reason="read_stagnation_second",
            )
        return None

    def _budget_guidance(
        self,
        *,
        step: int,
        run_state: RunState,
        has_modified_files: bool,
    ) -> ProgressGuidance | None:
        if (
            self._budget_progress_nudge_emitted
            or step * 5 < self.max_steps * 4
        ):
            return None
        self._budget_progress_nudge_emitted = True
        if not has_modified_files:
            return ProgressGuidance(
                level=4,
                message=_BUDGET_PROGRESS_NO_EDIT_MESSAGE,
                reason="budget_without_edit",
            )
        if run_state.verification.status == "failed":
            return ProgressGuidance(
                level=4,
                message=_BUDGET_PROGRESS_FAILED_VERIFICATION_MESSAGE,
                reason="budget_failed_verification",
            )
        return None


def _second_stagnation_message(*, step: int, max_steps: int) -> str:
    return f"""[Runtime guidance: progress stagnation 2]
你已经连续执行至少 12 次非修改操作。
当前模型调用：{step} / {max_steps}。

停止扩散探索。下一步优先选择：执行一个能够证伪首要假设的聚焦诊断；或对最早破坏预期不变量的位置做最小修改并立即验证。不要继续搜索新的旁支候选。"""
