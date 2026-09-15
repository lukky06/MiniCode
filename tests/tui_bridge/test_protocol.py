import json

import pytest

from minicode_harness.tui_bridge.protocol import (
    ApprovalRequired,
    ApprovalResponseMessage,
    AssistantDelta,
    CancelMessage,
    CommandMessage,
    ContextEvent,
    ErrorEvent,
    ExitRequested,
    JsonlProtocolError,
    PanelEvent,
    RunFinished,
    RunStarted,
    SessionStarted,
    SteerMessage,
    TaskMessage,
    ToolFinished,
    ToolStarted,
    UserInputRequired,
    UserInputResponseMessage,
    encode_message,
    parse_client_message,
    parse_server_message,
)


@pytest.mark.parametrize(
    "message",
    [
        SessionStarted(session_id="session_中文"),
        RunStarted(run_id="run_1"),
        ContextEvent(
            used=7200,
            window=64000,
            prompt_budget=56000,
            reserved_output=8000,
        ),
        ToolStarted(
            id="call_1",
            step=1,
            tool="read",
            target=r"C:\workspace\project\很长的路径\loop.py",
        ),
        ToolFinished(
            id="call_1",
            step=1,
            tool="read",
            status="ok",
            summary="读取完成 ✅",
        ),
        ToolFinished(
            id="cmd_1",
            step=2,
            tool="run_command",
            status="command_timed_out",
            summary="focused test timed out",
            command_status="timed_out",
            returncode=124,
            duration_ms=30000,
        ),
        ToolFinished(
            id="edit_1",
            step=3,
            tool="edit",
            status="ok",
            summary="edited app.py",
            diff_preview="@@ -1 +1 @@\n-old\n+new",
            diff_truncated=True,
        ),
        AssistantDelta(text="第一行\n第二行 😀"),
        ApprovalRequired(
            id="approval_1",
            tool_call_id="call_1",
            tool="run_command",
            summary="运行聚焦测试",
            details="pytest tests/test_x.py -q",
        ),
        UserInputRequired(
            id="input_1",
            question="选择兼容策略",
            options=[
                {"label": "strict", "description": "移除旧入口"},
                {"label": "compat", "description": "保留单向兼容"},
            ],
        ),
        RunFinished(status="completed", run_id="run_1"),
        ErrorEvent(message="配置错误", fatal=True),
        PanelEvent(name="help", title="Help", content="第一行\n第二行"),
        ExitRequested(),
    ],
)
def test_server_messages_round_trip_as_one_jsonl_record(message) -> None:
    encoded = encode_message(message)

    assert encoded.endswith("\n")
    assert encoded.count("\n") == 1
    assert "\r" not in encoded
    parsed = parse_server_message(encoded)

    assert parsed == message


@pytest.mark.parametrize(
    "message",
    [
        TaskMessage(text="修复终端渲染"),
        SteerMessage(text="只修改 terminal"),
        CommandMessage(text="/context"),
        ApprovalResponseMessage(id="approval_1", decision="approve"),
        ApprovalResponseMessage(id="approval_1", decision="approve_session"),
        ApprovalResponseMessage(id="approval_1", decision="reject"),
        ApprovalResponseMessage(id="approval_1", decision="skip"),
        ApprovalResponseMessage(id="approval_1", decision="abort"),
        UserInputResponseMessage(id="input_1", selected_index=1),
        CancelMessage(),
    ],
)
def test_client_messages_round_trip(message) -> None:
    assert parse_client_message(encode_message(message)) == message


def test_encode_preserves_unicode_without_ascii_expansion() -> None:
    encoded = encode_message(AssistantDelta(text="中文 😀"))

    assert "中文" in encoded
    assert "😀" in encoded
    assert "\\u4e2d" not in encoded


def test_embedded_newline_is_json_escaped_not_a_second_record() -> None:
    encoded = encode_message(AssistantDelta(text="a\nb"))

    payload = json.loads(encoded)
    assert payload["text"] == "a\nb"
    assert encoded.count("\n") == 1


@pytest.mark.parametrize(
    "line",
    [
        "",
        "\n",
        "[]\n",
        "{not-json}\n",
        '{"type":"unknown"}\n',
        '{"type":"cancel","extra":1}\n',
        '{"type":"approval_response","id":"a","decision":"yes"}\n',
        '{"type":"user_input_response","id":"a","selected_index":-1}\n',
        '{"type":"task","text":""}\n',
        '{"type":"cancel"}\n{"type":"cancel"}\n',
    ],
)
def test_invalid_client_records_fail_deterministically(line: str) -> None:
    with pytest.raises(JsonlProtocolError):
        parse_client_message(line)
