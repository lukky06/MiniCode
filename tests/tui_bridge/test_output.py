import json
from io import StringIO
from threading import Thread

from minicode_harness.output import ContextUsage
from minicode_harness.tui_bridge.output import JsonlOutputSink
from minicode_harness.tui_bridge.protocol import parse_server_message
from minicode_harness.tui_bridge.writer import JsonlEventWriter


def _sink() -> tuple[JsonlOutputSink, StringIO]:
    stream = StringIO()
    return JsonlOutputSink(JsonlEventWriter(stream)), stream


def test_jsonl_output_sink_emits_existing_runtime_callbacks_in_order() -> None:
    sink, stream = _sink()

    sink.session_started("session_1")
    sink.run_started("run_1")
    sink.context_built(
        usage=ContextUsage(
            build_duration_ms=12,
            token_estimate=7200,
            context_window=64000,
            prompt_budget=56000,
            reserved_output=8000,
        )
    )
    sink.model_reasoning_delta("先读取入口。\n")
    sink.tool_call_started(
        step=1,
        tool_call_id="call_1",
        tool_name="read",
        arguments={
            "source": "workspace",
            "target": "src/loop.py",
            "secret": "must-not-cross-protocol",
        },
    )
    sink.tool_call_finished(
        step=1,
        tool_call_id="call_1",
        tool_name="read",
        status="ok",
        summary="read completed",
        metadata={"secret": "must-not-cross-protocol"},
    )
    sink.model_text_delta("完成\n")
    sink.run_finished(status="completed", run_id="run_1")

    lines = stream.getvalue().splitlines()
    events = [parse_server_message(line) for line in lines]

    assert [event.type for event in events] == [
        "session_started",
        "run_started",
        "context",
        "reasoning_delta",
        "tool_started",
        "tool_finished",
        "assistant_delta",
        "run_finished",
    ]
    assert events[4].target == "src/loop.py"
    assert events[3].text == "先读取入口。\n"
    assert "must-not-cross-protocol" not in stream.getvalue()
    assert json.loads(lines[6])["text"] == "完成\n"


def test_jsonl_output_sink_emits_context_compaction_lifecycle() -> None:
    sink, stream = _sink()

    sink.context_compaction_started(kind="semantic")
    sink.context_compaction_finished(
        kind="semantic",
        duration_ms=245,
        success=True,
    )

    events = [
        parse_server_message(line)
        for line in stream.getvalue().splitlines()
    ]
    assert [event.type for event in events] == [
        "context_compaction",
        "context_compaction",
    ]
    assert events[0].phase == "started"
    assert events[0].kind == "semantic"
    assert events[0].duration_ms is None
    assert events[1].phase == "completed"
    assert events[1].duration_ms == 245


def test_jsonl_output_sink_bounds_activity_text_but_preserves_assistant_delta() -> None:
    sink, stream = _sink()
    long_target = "x" * 500

    sink.tool_call_started(
        step=3,
        tool_call_id="call_3",
        tool_name="run_command",
        arguments={"argv": ["pytest", long_target]},
    )
    sink.tool_call_finished(
        step=3,
        tool_call_id="call_3",
        tool_name="run_command",
        status="command_failed",
        summary="failure\n" + long_target,
        metadata={
            "command_status": "failed",
            "returncode": 1,
            "duration_ms": 1450,
            "runtime_task_id": "cmd_0001",
            "secret": "must-not-cross-protocol",
        },
    )
    sink.model_text_delta("a\nb")

    events = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    assert len(events[0].target or "") <= 240
    assert "\n" not in (events[1].summary or "")
    assert len(events[1].summary or "") <= 240
    assert events[1].command_status == "failed"
    assert events[1].returncode == 1
    assert events[1].duration_ms == 1450
    assert events[1].runtime_task_id == "cmd_0001"
    assert "must-not-cross-protocol" not in stream.getvalue()
    assert events[2].text == "a\nb"


def test_jsonl_output_sink_exposes_only_bounded_mutation_diff_fields() -> None:
    sink, stream = _sink()
    diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new"

    sink.tool_call_finished(
        step=2,
        tool_call_id="edit_1",
        tool_name="edit",
        status="ok",
        summary="edited app.py",
        metadata={
            "diff_preview": diff,
            "diff_truncated": True,
            "secret": "must-not-cross-protocol",
        },
    )

    [event] = [parse_server_message(line) for line in stream.getvalue().splitlines()]
    assert event.diff_preview == diff
    assert event.diff_truncated is True
    assert "must-not-cross-protocol" not in stream.getvalue()


def test_jsonl_writer_does_not_interleave_concurrent_records() -> None:
    sink, stream = _sink()

    threads = [
        Thread(target=sink.model_text_delta, args=(f"消息 {index} 😀",))
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 20
    assert all(parse_server_message(line).type == "assistant_delta" for line in lines)


def test_jsonl_output_sink_error_is_still_protocol_json() -> None:
    sink, stream = _sink()

    sink.error("配置失败", fatal=True)

    [line] = stream.getvalue().splitlines()
    event = parse_server_message(line)
    assert event.type == "error"
    assert event.message == "配置失败"
    assert event.fatal is True


def test_jsonl_output_sink_emits_exit_request() -> None:
    sink, stream = _sink()

    sink.exit_requested()

    [event] = [
        parse_server_message(line)
        for line in stream.getvalue().splitlines()
    ]
    assert event.type == "exit_requested"
