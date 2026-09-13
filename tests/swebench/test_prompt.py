from minicode_harness.swebench import SweBenchInstance
from minicode_harness.swebench.prompt import build_task_prompt


def test_prompt_guides_root_cause_investigation_without_gold_data() -> None:
    instance = SweBenchInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="abc123",
        problem_statement="A direct subclass behaves incorrectly.",
    )

    prompt = build_task_prompt(instance)

    assert "最小复现明确预期不变量" in prompt
    assert "第一次被破坏的位置" in prompt
    assert "直接证伪首要根因假设" in prompt
    assert "只有直接实现不能解释现象时" in prompt
    assert "父类、Mixin、Metaclass" in prompt
    assert "调用方、导入来源或并行实现" in prompt
    assert "证据能够完整解释现象时立即做最小修改" in prompt
    assert "重跑同一层级的失败断言" in prompt
    assert "不得只修补下游表现或修改测试证明修复正确" in prompt
    assert "不要重复读取已经覆盖的相同源码区域" in prompt
    assert "Text Search" not in prompt
    assert "仓库级搜索必须使用" not in prompt
    assert instance.problem_statement in prompt
    assert instance.base_commit not in prompt
    assert instance.instance_id not in prompt
    assert "FAIL_TO_PASS" not in prompt
    assert "PASS_TO_PASS" not in prompt
    assert "Gold Patch" not in prompt


def test_prompt_explains_no_verification_mode_and_direct_completion() -> None:
    instance = SweBenchInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="abc123",
        problem_statement="Fix the issue.",
    )

    prompt = build_task_prompt(instance, verification_enabled=False)

    assert "当前没有命令执行工具" in prompt
    assert "不要创建临时验证脚本" in prompt
    assert "不要寻找其他执行通道" in prompt
    assert "直接给出最终答案" in prompt


def test_prompt_can_return_problem_statement_without_adapter_guidance() -> None:
    instance = SweBenchInstance(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="abc123",
        problem_statement="Fix the issue.",
    )

    assert build_task_prompt(instance, include_prefix=False) == "Fix the issue."
