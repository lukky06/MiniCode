import json
from minicode_harness.context import ContextObservation
from minicode_harness.context.token import estimate_tokens
from minicode_harness.hooks import HookDecision, HookEvent, HookManager
from minicode_harness.trace import TraceWriter
from minicode_harness.hooks.builtins import default_hook_manager


def test_default_hook_manager_keeps_read_and_verification_retries_open() -> None:
    hook_names = [hook.name for hook in default_hook_manager().hooks]

    assert "repeated_tool_error_guard" not in hook_names
    assert "post_write_verification_guard" not in hook_names
    assert "duplicate_failed_verification_command_guard" not in hook_names
    assert hook_names == ["invalid_write_target_guard", "command_policy_guard"]


class AllowHook:
    name = "allow_hook"
    events = ("pre_tool_use",)

    def __init__(self) -> None:
        self.called = False

    def handle(self, event: HookEvent) -> HookDecision:
        self.called = True
        return HookDecision.allow(hook_name=self.name)


class BlockHook:
    name = "block_hook"
    events = ("pre_tool_use",)

    def __init__(self) -> None:
        self.called = False

    def handle(self, event: HookEvent) -> HookDecision:
        self.called = True
        content = "Blocked by deterministic hook."
        return HookDecision.block(
            hook_name=self.name,
            reason="test_block",
            observation=ContextObservation(
                tool_call_id="call_1",
                tool_name="read",
                content=content,
                output_preview=content,
                token_estimate=estimate_tokens(content),
                summary=content,
                metadata={"status": "blocked_by_test_hook"},
            ),
        )


class LaterHook:
    name = "later_hook"
    events = ("pre_tool_use",)

    def __init__(self) -> None:
        self.called = False

    def handle(self, event: HookEvent) -> HookDecision:
        self.called = True
        return HookDecision.allow(hook_name=self.name)


class FailingHook:
    name = "failing_hook"
    events = ("pre_tool_use",)

    def handle(self, event: HookEvent) -> HookDecision:
        raise RuntimeError("boom")


def test_hook_manager_short_circuits_and_traces_non_allow_decision(tmp_path) -> None:
    allow_hook = AllowHook()
    block_hook = BlockHook()
    later_hook = LaterHook()
    trace_path = tmp_path / "trace.jsonl"
    manager = HookManager(
        [allow_hook, block_hook, later_hook],
        trace_writer=TraceWriter(trace_path),
    )

    decision = manager.emit(
        HookEvent(
            name="pre_tool_use",
            run_id="run_1",
            task="Read file",
            workspace=str(tmp_path),
            step=3,
            payload={},
        )
    )

    assert allow_hook.called is True
    assert block_hook.called is True
    assert later_hook.called is False
    assert decision.action == "block"
    assert decision.observation is not None
    assert decision.observation.metadata["status"] == "blocked_by_test_hook"

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        {
            "type": "hook_decision",
            "time": events[0]["time"],
            "step": 3,
            "hook": "block_hook",
            "lifecycle_event": "pre_tool_use",
            "action": "block",
            "reason": "test_block",
            "observation_status": "blocked_by_test_hook",
            "tool": None,
        }
    ]


def test_hook_manager_records_hook_error_and_continues(tmp_path) -> None:
    allow_hook = AllowHook()
    trace_path = tmp_path / "trace.jsonl"
    manager = HookManager(
        [FailingHook(), allow_hook],
        trace_writer=TraceWriter(trace_path),
    )

    decision = manager.emit(
        HookEvent(
            name="pre_tool_use",
            run_id="run_1",
            task="Read file",
            workspace=str(tmp_path),
            step=1,
            payload={},
        )
    )

    assert decision.action == "allow"
    assert allow_hook.called is True
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "hook_error"
    assert events[0]["hook"] == "failing_hook"
    assert events[0]["error_type"] == "RuntimeError"
