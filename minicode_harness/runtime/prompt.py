"""Dynamic system-prompt assembly.

Every model call rebuilds a compact system prompt from live runtime facts while
repository evidence stays in canonical provider-native messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol


PROMPT_RUNTIME_VERSION = "claude-style-runtime-v24"


class SkillLike(Protocol):
    name: str
    description: str
    source: str


_CORE_RULES = """你是本地代码智能体 MiniCode，使用工具在工作区完成用户任务。

一、证据
* 仓库、源码、命令、测试和修改事实必须来自工具，不得凭名称或猜测声明。
* 复用仍有效的历史证据；信息缺失、范围改变或写入使证据失效时再读取。探索以回答当前问题所需的充分证据为目标，证据已经足够时停止扩大范围，不以穷举仓库为目标。

二、定位与修改
* 用户要求修改、编写、修复、重构或补测试时必须落地，信息明确后立即最小修改。
* 文件未知时在最窄目录做 Files Search；文件已知但定义、引用或调用点未知时，对精确文件做 Text Search；位置明确后读取局部实现。短小资源、明确全文请求可直接读取。
* 不得以依次全文读取候选文件代替搜索。仅在证据不足或聚焦验证暴露新位置时扩大范围；每轮最多调用 6 个直接相关工具。

三、约束与测试
* 严格遵守用户禁止项，不得反向解释为要求；无新证据不得破坏已满足约束的实现。
* 默认运行现有最小相关测试；现有测试不足时才新增，并复用项目测试结构和 Mock API。
* 代码变更后至少运行一次最小相关验证；无法验证须说明原因。

四、失败
* 工具失败、拒绝和策略结果都是证据；条件未变时不得机械重试。

五、任务与子 Agent
* `task` 仅用于多步骤任务；状态不是完成证据。
* `delegate_task` 只用于边界明确、可通过有界只读探索独立回答的问题，例如一条完整调用链或一个相对独立的问题；不要在一个子任务中组合多个模块、多条调用链或多个独立检查目标。
* 不要委派读取一个已知短文件、查找一个已知定义等简单任务；正常完成的子 Agent结论可直接使用，不重复执行相同探索。
* 子 Agent 不得写入、运行命令或递归委派；修改任务只能交给隔离 Worktree Worker。

六、协议与回答
* 工具调用轮只返回原生工具调用；同一响应中的多个工具调用默认并行执行，存在前置依赖时必须拆到后续响应。
* 完成或明确受阻时再给最终答案。执行命令后报告完整命令、退出码和结果；最终答案只陈述已完成事实、真实验证与阻塞，语言跟随用户。"""


def build_stable_system_prefix(
    *,
    available_skills: Iterable[SkillLike],
    repository_rules: str = "",
    long_term_context: str = "",
    repository_structure_card: str = "",
) -> str:
    """Build the cacheable portion of the system prompt."""

    sections = [_CORE_RULES]
    rules = _strip_section_heading(
        repository_rules,
        headings=("Repository Rules:", "仓库规则：", "仓库规则:"),
    )
    if rules:
        sections.append("仓库规则：\n" + rules)

    memory_index = _strip_section_heading(
        long_term_context,
        headings=("Repository Memory Index:",),
    )
    if memory_index:
        sections.append("长期记忆：\n" + memory_index)

    skills = list(available_skills)
    if skills:
        rendered = [
            f"- `{skill.name}`：{skill.description.strip()}"
            for skill in skills
        ]
        sections.append(
            "可用技能目录（按需调用 `read(source=skill, target=<name>)` 读取全文）：\n"
            + "\n".join(rendered)
        )

    repository = _strip_section_heading(
        repository_structure_card,
        headings=("Repository Structure Card:", "仓库概览：", "仓库提示："),
    )
    if repository:
        sections.append("仓库提示：\n" + repository)

    return "\n\n".join(section for section in sections if section.strip())


def build_dynamic_system_suffix(
    *,
    workspace: str | Path,
    streaming_enabled: bool = False,
    current_step: int | None = None,
    max_steps: int | None = None,
    current_tool_calls: int | None = None,
    max_tool_calls: int | None = None,
    collaboration_mode: str = "default",
) -> str:
    """Build only the live final-answer protocol."""

    sections: list[str] = []
    if collaboration_mode == "plan":
        sections.append(
            "Plan Mode：当前 Run 只做只读探索和实施规划。"
            "不得修改工作区、运行命令、启动 Worktree Worker 或调用有副作用的扩展工具。"
            "可使用只读工具收集必要证据，并可用 task 维护 Run 内计划步骤。"
            "可从仓库发现的事实必须先通过只读工具探索；只有无法从仓库推导且会改变方案的用户偏好或取舍才需要询问。"
            "若 request_user_input 可用，使用它提供 2-4 个清晰选项；若不可用，在最终计划中明确标记待确认项。"
            "最终回答应给出有顺序、可验证的实施计划，包含关键文件/模块、修改步骤、验证方式和已知风险。"
            "不得声称文件已经修改、命令已经执行或测试已经通过。"
        )
    if streaming_enabled:
        sections.append(
            "流式最终答案协议：最终回答必须包裹在 <final_answer> 与 </final_answer> 中；"
            "工具调用轮不要输出该标签。"
        )
    return "\n\n".join(sections)


def _strip_section_heading(value: str, *, headings: tuple[str, ...]) -> str:
    text = value.strip()
    for heading in headings:
        if text.startswith(heading):
            return text[len(heading) :].lstrip()
    return text
