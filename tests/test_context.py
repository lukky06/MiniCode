import json

import pytest

from minicode_harness.context import (
    ContextBuilder,
    ContextObservation,
    ContextSkill,
    PromptSectionCache,
    RunState,
    TokenBudget,
    initialize_run_state,
    mark_verification_failed,
    mark_verification_not_run,
    mark_verification_passed,
    mark_verification_rolled_back,
    record_inspected_file,
    render_repository_structure_card,
)
from minicode_harness.context.token import estimate_tokens


def _rendered(context) -> str:
    return "\n".join(str(message.get("content") or "") for message in context.messages)


def _observation(
    *,
    tool: str = "read_file",
    status: str = "ok",
    path: str = "src/a.py",
    start: int = 1,
    end: int = 10,
    total: int = 10,
) -> ContextObservation:
    return ContextObservation(
        tool_call_id="call_1",
        tool_name=tool,
        content="VALUE = 1",
        output_preview="VALUE = 1",
        token_estimate=4,
        summary="Read a.py.",
        metadata={
            "status": status,
            "path": path,
            "start_line": start,
            "end_line": end,
            "total_lines": total,
        },
    )


def test_mixed_language_token_estimator_is_conservative_for_cjk() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("中文任务") == 4
    assert estimate_tokens("ab中文🙂") == 4


def test_ascii_source_estimate_remains_close_to_four_characters_per_token() -> None:
    source = "def add(a, b):\n    return a + b\n"

    assert estimate_tokens(source) == 8


def test_context_builder_renders_system_only_without_redundant_runtime_fields(tmp_path) -> None:
    context = ContextBuilder(prompt_cache_enabled=False).build(
        available_skills=[
            ContextSkill(
                name="repo-explain",
                description="Use source evidence.",
                source="builtin",
            )
        ],
        workspace=tmp_path,
        long_term_context="Repository Memory Index:\n- 输出中文",
        repository_structure_card="Repository Structure Card:\n- build_files:\n  - pyproject.toml",
    )

    rendered = _rendered(context)
    assert len(context.messages) == 1
    assert context.messages[0]["role"] == "system"
    assert "解释关注功能" not in rendered
    assert rendered.count("长期记忆：") == 1
    assert "Repository Memory Index" not in rendered
    assert rendered.count("仓库提示：") == 1
    assert "Repository Structure Card" not in rendered
    assert "pyproject.toml" in rendered
    assert "可用工具：" not in rendered
    assert "运行模式：" not in rendered
    assert "当前时间：" not in rendered
    assert "最终答案协议" not in rendered
    assert "<final_answer>" not in rendered
    assert "Recent Dialogue" not in rendered
    assert "Test Context Pack" not in rendered
    assert "Sub-Agent Digest" not in rendered
    assert "evidence_sufficiency" not in rendered
    assert "active_edit_target" not in rendered
    assert "Task Memory" not in rendered


def test_context_builder_guides_focused_dependency_search_and_verification() -> None:
    rendered = _rendered(
        ContextBuilder(prompt_cache_enabled=False).build(available_skills=[])
    )

    assert "文件未知时在最窄目录做 Files Search" in rendered
    assert "文件已知但定义、引用或调用点未知时" in rendered
    assert "对精确文件做 Text Search" in rendered
    assert "不得以依次全文读取候选文件代替搜索" in rendered
    assert "充分证据" in rendered
    assert "不以穷举仓库为目标" in rendered
    assert "边界明确" in rendered
    assert "一条完整调用链" in rendered
    assert "不要在一个子任务中组合多个模块" in rendered
    assert "正常完成的子 Agent结论可直接使用" in rendered
    assert "evidence_coordinates" not in rendered
    assert "implementation-confirmed" not in rendered
    assert "代码变更后至少运行一次最小相关验证" in rendered
    assert "同一响应中的多个工具调用默认并行执行" in rendered
    assert "存在前置依赖时必须拆到后续响应" in rendered
    assert "执行命令后报告完整命令、退出码和结果" in rendered
    assert "记忆首步门" not in rendered
    assert "首个模型动作必须且只能" not in rendered
    assert "decision=skip" not in rendered
    assert 600 <= estimate_tokens(rendered) <= 900


def test_context_builder_adds_final_answer_boundary_only_for_streaming() -> None:
    context = ContextBuilder(prompt_cache_enabled=False).build(
        available_skills=[],
        streaming_enabled=True,
    )

    rendered = _rendered(context)
    assert "<final_answer>" in rendered
    assert "工具调用轮不要输出该标签" in rendered


def test_context_builder_rejects_removed_compatibility_arguments() -> None:
    builder = ContextBuilder()
    with pytest.raises(TypeError):
        builder.build(
            task="explain",
            available_skills=[],
        )
    with pytest.raises(TypeError):
        builder.build(
            available_skills=[],
            observations=[],
        )
    with pytest.raises(TypeError):
        builder.build(
            available_skills=[],
            task_memory=RunState(),
        )
    with pytest.raises(TypeError):
        builder.build(
            available_skills=[],
            task_state={},
        )


def test_context_builder_prompt_cache_reuses_stable_system_prefix(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = PromptSectionCache(
        workspace=workspace,
        provider="test",
        model="model",
        data_dir=tmp_path / "data",
    )
    builder = ContextBuilder(prompt_cache=cache)

    first = builder.build(available_skills=[], long_term_context="rule")
    second = builder.build(available_skills=[], long_term_context="rule")

    assert first.prompt_cache_enabled is True
    assert first.prompt_cache_hit is False
    assert second.prompt_cache_hit is True
    assert second.prompt_cache_key == first.prompt_cache_key


def test_context_builder_omits_runtime_path_and_budgets_from_prompt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = PromptSectionCache(
        workspace=workspace,
        provider="test",
        model="model",
        data_dir=tmp_path / "data",
    )
    builder = ContextBuilder(prompt_cache=cache)

    first = builder.build(
        available_skills=[],
        current_step=15,
        max_steps=20,
        current_tool_calls=24,
        max_tool_calls=30,
    )
    final = builder.build(
        available_skills=[],
        current_step=20,
        max_steps=20,
        current_tool_calls=30,
        max_tool_calls=30,
    )

    assert "工作目录：" not in _rendered(first)
    assert "模型调用预算" not in _rendered(first)
    assert "模型调用预算" not in _rendered(final)
    assert "工具调用预算" not in _rendered(first)
    assert "工具调用预算" not in _rendered(final)
    assert final.prompt_cache_hit is True
    assert final.prompt_cache_key == first.prompt_cache_key
    assert final.prompt_prefix_hash == first.prompt_prefix_hash


def test_repository_structure_card_contains_only_high_density_hints(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_app.py").write_text("def test_app(): pass\n", encoding="utf-8")
    (workspace / "node_modules").mkdir()

    card = render_repository_structure_card(workspace)

    assert "build_files" in card
    assert "pyproject.toml" in card
    assert "source_roots" in card
    assert "test_roots" in card
    assert "workspace_root" not in card
    assert "primary_language" not in card
    assert "detected_languages" not in card
    assert "generated_or_ignored_dirs" not in card
    assert ": none" not in card


def test_token_budget_orders_soft_semantic_and_emergency_hard_limits() -> None:
    budget = TokenBudget(context_budget=32_000, reserved_output=6_000)

    assert budget.prompt_budget == 26_000
    assert budget.soft_token_limit == 20_800
    assert budget.semantic_token_limit == 22_880
    assert budget.hard_token_limit == 24_700

    constrained = TokenBudget(
        context_budget=10_000,
        reserved_output=0,
        soft_limit=0.20,
        semantic_limit=0.88,
        hard_limit=0.50,
    )
    assert constrained.soft_token_limit < constrained.semantic_token_limit
    assert constrained.semantic_token_limit < constrained.hard_token_limit


def test_context_builder_packs_large_optional_system_context(tmp_path) -> None:
    budget = TokenBudget(context_budget=400, reserved_output=100, soft_limit=0.5, hard_limit=0.8)
    context = ContextBuilder(budget=budget, prompt_cache_enabled=False).build(
        available_skills=[
            ContextSkill(name="large", description="skill detail\n" * 200, source="test")
        ],
        workspace=tmp_path,
        long_term_context="project rule\n" * 200,
        repository_structure_card="project source\n" * 200,
    )

    assert context.token_estimate <= budget.hard_token_limit
    assert context.compression_events


def test_run_state_keeps_strongest_file_coverage() -> None:
    state = initialize_run_state()
    record_inspected_file(
        state=state,
        tool_name="read",
        tool_call_id="full",
        arguments={"source": "workspace", "target": "src/a.py"},
        observation=_observation(),
        step=1,
    )
    partial = _observation(start=4, end=6, total=10)
    partial.tool_call_id = "partial"
    record_inspected_file(
        state=state,
        tool_name="read",
        tool_call_id="partial",
        arguments={
            "source": "workspace",
            "target": "src/a.py",
            "start_line": 4,
            "end_line": 6,
        },
        observation=partial,
        step=2,
    )

    assert len(state.inspected_files) == 1
    assert state.inspected_files[0].last_tool_call_id == "full"
    assert state.inspected_files[0].line_start == 1
    assert state.inspected_files[0].line_end == 10


def test_verification_state_updates_are_structured() -> None:
    state = RunState()

    mark_verification_passed(state, command="pytest tests/test_x.py -q")
    assert state.verification.status == "passed"
    assert state.verification.command == "pytest tests/test_x.py -q"
    assert state.verification.returncode == 0

    mark_verification_failed(
        state,
        command="pytest tests/test_x.py -q",
        returncode=1,
        reason="assertion failed",
    )
    assert state.verification.status == "failed"
    assert state.verification.returncode == 1
    assert state.verification.reason == "assertion failed"

    mark_verification_not_run(state)
    assert state.verification.status == "not_run"
    assert state.verification.command is None

    mark_verification_rolled_back(state, reason="max_steps")
    assert state.verification.status == "rolled_back"
    assert state.verification.reason == "max_steps"


def test_run_state_serialization_is_minimal() -> None:
    payload = json.loads(RunState().model_dump_json())
    assert set(payload) == {"inspected_files", "verification"}
    assert payload["verification"] == {
        "status": "not_run",
        "command": None,
        "returncode": None,
        "reason": None,
    }
