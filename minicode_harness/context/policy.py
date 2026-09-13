"""集中定义上下文压缩的轻量内部策略。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactionPolicy:
    """压缩阈值集合，不承担动态路由或用户配置职责。"""

    recent_semantic_turns: int = 3
    current_frontier_token_limit: int = 4_000
    semantic_summary_max_tokens: int = 1_600
