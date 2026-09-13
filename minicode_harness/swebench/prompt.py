"""Prompt construction for SWE-bench instances."""

from __future__ import annotations

from .models import SweBenchInstance


DEFAULT_PREFIX = (
    "请根据以下问题修改当前仓库。定位根因，完成最小必要修改，并在可用时运行聚焦验证。"
)

INVESTIGATION_GUIDANCE = """调查要求：

- 先用最小复现明确预期不变量，再沿数据流定位该不变量第一次被破坏的位置。
- 优先执行一个能够直接证伪首要根因假设的诊断，不要顺序浏览文件或下游调用链。
- 只有直接实现不能解释现象时，才扩展到父类、Mixin、Metaclass、装饰器、调用方、导入来源或并行实现。
- 证据能够完整解释现象时立即做最小修改，不继续扩大探索范围。
- 修改后先重跑同一层级的失败断言；若仍失败，重新检查最早错误点，不得只修补下游表现或修改测试证明修复正确。
- 不要重复读取已经覆盖的相同源码区域。"""

NO_VERIFICATION_GUIDANCE = """当前没有命令执行工具：

- 不要创建临时验证脚本、临时测试文件或一次性调试文件。
- 不要寻找其他执行通道，也不要尝试委派任务来运行命令。
- 完成最小修改后，只做必要的源码或 Diff 复核，然后直接给出最终答案。
- 最终答案中明确说明验证未运行。"""


def build_task_prompt(
    instance: SweBenchInstance,
    *,
    include_prefix: bool = True,
    verification_enabled: bool = True,
) -> str:
    """Return a Gold-free task prompt for the normal AgentLoop."""

    statement = instance.problem_statement.strip()
    if not include_prefix:
        return statement
    guidance = [DEFAULT_PREFIX, INVESTIGATION_GUIDANCE]
    if not verification_enabled:
        guidance.append(NO_VERIFICATION_GUIDANCE)
    guidance.append(statement)
    return "\n\n".join(guidance)
