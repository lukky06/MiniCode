"""Build the dynamic system prompt for one model call.

Tool evidence and user tasks are never rebuilt here. They remain in the
provider-native canonical message history owned by ``UserTurnState``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from minicode_harness.runtime import (
    PROMPT_RUNTIME_VERSION,
    build_dynamic_system_suffix,
    build_stable_system_prefix,
)

from .token import estimate_tokens
from .types import BuiltContext, ContextCompressionEvent, ContextSkill, TokenBudget


PROMPT_BUILDER_VERSION = PROMPT_RUNTIME_VERSION
@dataclass(frozen=True)
class PromptPrefixMetadata:
    prefix_hash: str | None = None
    prefix_tokens: int = 0


class ContextBuilder:
    """Assemble the model-facing system prompt only.

    Conversation history, observations, controller state, session transcripts,
    test-context packs, and artifact recall belong in canonical messages or
    explicit tools and are intentionally not accepted here.
    """

    def __init__(self, budget: TokenBudget | None = None) -> None:
        self.budget = budget or TokenBudget()

    def build(
        self,
        *,
        available_skills: list[ContextSkill],
        workspace: str | Path = ".",
        repository_rules: str = "",
        long_term_context: str = "",
        repository_structure_card: str = "",
        streaming_enabled: bool = False,
        current_step: int | None = None,
        max_steps: int | None = None,
        current_tool_calls: int | None = None,
        max_tool_calls: int | None = None,
        collaboration_mode: str = "default",
    ) -> BuiltContext:
        """Build one bounded system message under the configured prompt budget."""

        stable_prefix, prefix = self._stable_prefix(
            available_skills,
            repository_rules,
            long_term_context,
            repository_structure_card=repository_structure_card,
        )
        dynamic_suffix = build_dynamic_system_suffix(
            workspace=workspace,
            streaming_enabled=streaming_enabled,
            current_step=current_step,
            max_steps=max_steps,
            current_tool_calls=current_tool_calls,
            max_tool_calls=max_tool_calls,
            collaboration_mode=collaboration_mode,
        )
        system = _combine_system(stable_prefix, dynamic_suffix)
        messages = _system_messages(system)
        token_estimate = estimate_tokens(system)
        compression_events: list[ContextCompressionEvent] = []

        if token_estimate > self.budget.soft_token_limit:
            before = token_estimate
            compact_prefix = build_stable_system_prefix(
                available_skills=_compact_skills(available_skills),
                repository_rules=_clip(repository_rules, 8000),
                long_term_context=_clip(long_term_context, 1600),
                repository_structure_card=_clip(repository_structure_card, 1800),
            )
            system = _combine_system(compact_prefix, dynamic_suffix)
            messages = _system_messages(system)
            token_estimate = estimate_tokens(system)
            prefix = _prefix_metadata(compact_prefix)
            compression_events.append(
                ContextCompressionEvent(
                    reason="system_prompt_soft_limit",
                    before_tokens=before,
                    after_tokens=token_estimate,
                    details={"strategy": "compact_optional_system_sections"},
                )
            )

        if token_estimate > self.budget.hard_token_limit:
            before = token_estimate
            minimal_system = _combine_system(
                "你是 MiniCode。使用当前消息历史和提供的工具直接完成代码任务；"
                "不要猜测仓库事实，不要机械重复相同调用；写入后优先做最小验证。",
                dynamic_suffix,
            )
            messages = _system_messages(minimal_system)
            token_estimate = estimate_tokens(minimal_system)
            prefix = _prefix_metadata(minimal_system)
            compression_events.append(
                ContextCompressionEvent(
                    reason="system_prompt_hard_limit",
                    before_tokens=before,
                    after_tokens=token_estimate,
                    details={"strategy": "minimal_runtime_prompt"},
                )
            )

        return BuiltContext(
            messages=messages,
            token_estimate=token_estimate,
            compression_events=compression_events,
            prompt_prefix_hash=prefix.prefix_hash,
            prompt_prefix_tokens=prefix.prefix_tokens,
        )

    def _stable_prefix(
        self,
        available_skills: list[ContextSkill],
        repository_rules: str,
        long_term_context: str,
        *,
        repository_structure_card: str,
    ) -> tuple[str, PromptPrefixMetadata]:
        content = build_stable_system_prefix(
            available_skills=available_skills,
            repository_rules=repository_rules,
            long_term_context=long_term_context,
            repository_structure_card=repository_structure_card,
        )
        return content, _prefix_metadata(content)


def _combine_system(stable_prefix: str, dynamic_suffix: str) -> str:
    return "\n\n".join(
        section.strip()
        for section in (stable_prefix, dynamic_suffix)
        if section.strip()
    )


def _system_messages(system: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system.strip()}]


def _compact_skills(skills: list[ContextSkill]) -> list[ContextSkill]:
    return [
        ContextSkill(
            name=skill.name,
            source=skill.source,
            description=_clip(skill.description, 400),
        )
        for skill in skills[:3]
    ]


def _clip(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)].rstrip() + "\n...<system compacted>"


def _prefix_metadata(content: str) -> PromptPrefixMetadata:
    return PromptPrefixMetadata(
        prefix_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        prefix_tokens=estimate_tokens(content),
    )
