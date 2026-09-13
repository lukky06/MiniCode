from io import BytesIO, StringIO, TextIOWrapper

from minicode_harness.output import FinalAnswerXmlStreamEmitter, TextOutputSink


def test_text_output_sink_does_not_print_model_streaming_banner() -> None:
    stream = StringIO()
    sink = TextOutputSink(stream=stream)

    sink.model_stream_started()

    assert stream.getvalue() == ""


class RecordingStreamHandler:
    def __init__(self) -> None:
        self.text: list[str] = []

    def model_stream_started(self) -> None:
        pass

    def model_text_delta(self, text: str) -> None:
        self.text.append(text)


def test_final_answer_xml_stream_emitter_waits_for_complete_parseable_xml() -> None:
    sink = RecordingStreamHandler()
    emitter = FinalAnswerXmlStreamEmitter(sink)

    emitter.feed("<final_answer><summary>还没闭合")

    assert sink.text == []
    assert not emitter.emitted

    emitter.feed("。</summary><verification>只读。</verification></final_answer>")

    assert sink.text == ["还没闭合。\n验证：只读。"]
    assert emitter.emitted


def test_final_answer_xml_stream_emitter_streams_plain_root_text_after_boundary() -> None:
    sink = RecordingStreamHandler()
    emitter = FinalAnswerXmlStreamEmitter(sink)

    emitter.feed("hidden commentary<final_answer>plain answer chunk one ")
    emitter.feed("chunk two</final_answer>")

    assert "".join(sink.text) == "plain answer chunk one chunk two"
    assert len(sink.text) >= 2
    assert emitter.emitted


def test_final_answer_xml_stream_emitter_emits_each_plain_character_immediately() -> None:
    sink = RecordingStreamHandler()
    emitter = FinalAnswerXmlStreamEmitter(sink)
    text = "中文 😀 **bold** `code`\n```python\npass\n```"

    emitter.feed("<final_answer>")
    for index, char in enumerate(text, start=1):
        emitter.feed(char)
        assert "".join(sink.text) == text[:index]
    for char in "</final_answer>":
        emitter.feed(char)

    assert sink.text == list(text)
    assert emitter.root_closed


def test_final_answer_xml_stream_emitter_flushes_plain_truncated_tail() -> None:
    sink = RecordingStreamHandler()
    emitter = FinalAnswerXmlStreamEmitter(sink)

    emitter.feed("<final_answer>plain answer without a closing tag")
    emitter.flush_partial()

    assert "".join(sink.text) == "plain answer without a closing tag"
    assert emitter.emitted


def test_final_answer_xml_stream_emitter_suppresses_malformed_xml() -> None:
    sink = RecordingStreamHandler()
    emitter = FinalAnswerXmlStreamEmitter(sink)

    emitter.feed("<final_answer><summary>非法 & XML</summary></final_answer>")

    assert sink.text == []
    assert not emitter.emitted


def test_text_output_sink_replaces_characters_unsupported_by_console_codec() -> None:
    raw = BytesIO()
    stream = TextIOWrapper(raw, encoding="gbk", errors="strict")
    sink = TextOutputSink(stream=stream)

    sink.model_text_delta("支持中文，unsupported bullet: •")
    stream.flush()

    assert raw.getvalue().decode("gbk") == "支持中文，unsupported bullet: ?"


def test_text_output_sink_prints_compact_tool_status_lines() -> None:
    stream = StringIO()
    sink = TextOutputSink(stream=stream)

    sink.tool_call_started(
        step=1,
        tool_name="read",
        arguments={
            "source": "workspace",
            "target": "src/main/java/com/example/Example.java",
        },
    )
    sink.tool_call_finished(step=1, tool_name="read", status="ok")

    assert stream.getvalue() == (
        "[tool] read {source='workspace', target='src/main/java/com/example/Example.java'}\n"
        "[tool] read ok\n"
    )
