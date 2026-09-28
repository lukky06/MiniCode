from minicode_harness.context import ContextObservation, InspectedFile, RunState
from minicode_harness.models import NormalizedToolCall
from minicode_harness.runtime.tool_reuse import ReadReuseMatch, ToolReuseTracker


def _read_call(call_id: str, start: int, end: int) -> NormalizedToolCall:
    return NormalizedToolCall(
        id=call_id,
        name="read",
        arguments={
            "source": "workspace",
            "target": "source.py",
            "start_line": start,
            "end_line": end,
        },
    )


def test_tracker_matches_contained_reads_and_marks_partial_overlap() -> None:
    tracker = ToolReuseTracker()
    original = _read_call("read_big", 1, 200)
    observation = ContextObservation(
        tool_call_id="read_big",
        tool_name="read",
        content="source",
        output_preview="source",
        token_estimate=1,
        summary="source.py lines 1-200",
        metadata={
            "status": "ok",
            "read_source": "workspace",
            "path": "source.py",
            "start_line": 1,
            "end_line": 200,
            "total_lines": 300,
        },
    )

    tracker.record(original, observation, workspace_generation=0)

    contained = tracker.lookup(_read_call("read_small", 100, 150), workspace_generation=0)
    assert isinstance(contained, ReadReuseMatch)
    assert contained.path == "source.py"
    assert contained.requested == (100, 150)
    assert contained.covered_by == (1, 200)
    assert tracker.overlap_detected(
        _read_call("read_overlap", 180, 250),
        workspace_generation=0,
    ) is True


def test_restore_uses_run_state_as_read_coverage_source() -> None:
    tracker = ToolReuseTracker()
    observation = ContextObservation(
        tool_call_id="observation_read",
        tool_name="read",
        content="source",
        output_preview="source",
        token_estimate=1,
        summary="source.py lines 1-200",
        metadata={
            "status": "ok",
            "read_source": "workspace",
            "path": "source.py",
            "start_line": 1,
            "end_line": 200,
            "total_lines": 300,
            "workspace_generation": 4,
        },
    )
    run_state = RunState(
        workspace_generation=4,
        inspected_files=[
            InspectedFile(
                path="source.py",
                summary="source.py lines 1-200",
                last_tool_call_id="run_state_read",
                last_step=1,
                line_start=1,
                line_end=200,
                total_lines=300,
                content_status="partial_content_available_in_context",
                workspace_generation=4,
            )
        ],
    )

    tracker.restore(
        observations=[observation],
        run_state=run_state,
        workspace_generation=4,
    )

    match = tracker.lookup(_read_call("read_small", 100, 150), workspace_generation=4)
    assert isinstance(match, ReadReuseMatch)
    assert match.source_tool_call_id == "run_state_read"
