from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json

import pytest

from minicode_harness.trace import TraceWriter


def test_trace_writer_appends_jsonl_events(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = TraceWriter(trace_path)

    first_event = writer.write_event(
        "run_started",
        run_id="run_20260630_001",
        task="noop",
        time=datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
    )
    writer.write_event(
        "tool_called",
        step=1,
        tool="search",
        time=datetime(2026, 6, 30, 10, 0, 1, tzinfo=timezone.utc),
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert first_event["type"] == "run_started"
    assert len(lines) == 2

    run_started = json.loads(lines[0])
    tool_called = json.loads(lines[1])
    assert run_started == {
        "type": "run_started",
        "time": "2026-06-30T10:00:00+00:00",
        "run_id": "run_20260630_001",
        "task": "noop",
    }
    assert tool_called == {
        "type": "tool_called",
        "time": "2026-06-30T10:00:01+00:00",
        "step": 1,
        "tool": "search",
    }


def test_trace_writer_serializes_concurrent_events(tmp_path) -> None:
    trace_path = tmp_path / "concurrent.jsonl"
    writer = TraceWriter(trace_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(writer.write_event, "worker_event", index=index)
            for index in range(200)
        ]
        for future in futures:
            future.result()

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 200
    assert {event["index"] for event in events} == set(range(200))


def test_trace_writer_rejects_empty_event_type(tmp_path) -> None:
    with pytest.raises(ValueError, match="event type"):
        TraceWriter(tmp_path / "trace.jsonl").write_event("")
