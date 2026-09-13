from minicode_harness.swebench.metrics import _collect_convergence_metrics


def test_convergence_metrics_capture_edit_and_verification_behavior() -> None:
    target = ["python", "-m", "pytest", "-q", "tests/test_app.py::test_case"]
    events = [
        {"type": "tool_called", "step": 1, "tool_call_id": "read_1", "tool": "read", "args": {}},
        {
            "type": "tool_called",
            "step": 2,
            "tool_call_id": "test_before",
            "tool": "run_command",
            "args": {"argv": target},
        },
        {
            "type": "tool_arguments_validated",
            "step": 2,
            "tool_call_id": "test_before",
            "command_category": "verification",
        },
        {
            "type": "run_state_updated",
            "step": 2,
            "tool_call_id": "test_before",
            "tool": "run_command",
            "modified_files": 0,
            "verification_status": "failed",
        },
        {
            "type": "tool_called",
            "step": 3,
            "tool_call_id": "edit_1",
            "tool": "edit",
            "args": {"path": "app.py", "old_text": "old branch"},
        },
        {
            "type": "run_state_updated",
            "step": 3,
            "tool_call_id": "edit_1",
            "tool": "edit",
            "modified_files": 1,
            "verification_status": "not_run",
        },
        {"type": "tool_called", "step": 4, "tool_call_id": "read_2", "tool": "read", "args": {}},
        {
            "type": "tool_called",
            "step": 5,
            "tool_call_id": "test_after_1",
            "tool": "run_command",
            "args": {"argv": target},
        },
        {
            "type": "tool_arguments_validated",
            "step": 5,
            "tool_call_id": "test_after_1",
            "command_category": "verification",
        },
        {
            "type": "run_state_updated",
            "step": 5,
            "tool_call_id": "test_after_1",
            "tool": "run_command",
            "modified_files": 1,
            "verification_status": "failed",
        },
        {"type": "tool_called", "step": 6, "tool_call_id": "search_1", "tool": "search", "args": {}},
        {
            "type": "tool_called",
            "step": 7,
            "tool_call_id": "edit_2",
            "tool": "edit",
            "args": {"path": "parser.py", "old_text": "earlier invariant"},
        },
        {
            "type": "run_state_updated",
            "step": 7,
            "tool_call_id": "edit_2",
            "tool": "edit",
            "modified_files": 2,
            "verification_status": "not_run",
        },
        {
            "type": "tool_called",
            "step": 8,
            "tool_call_id": "test_after_2",
            "tool": "run_command",
            "args": {"argv": target},
        },
        {
            "type": "tool_arguments_validated",
            "step": 8,
            "tool_call_id": "test_after_2",
            "command_category": "verification",
        },
        {
            "type": "run_state_updated",
            "step": 8,
            "tool_call_id": "test_after_2",
            "tool": "run_command",
            "modified_files": 2,
            "verification_status": "passed",
        },
    ]

    metrics = _collect_convergence_metrics(events)

    assert metrics == {
        "first_edit_step": 3,
        "non_write_calls_before_first_edit": 2,
        "reproducer_to_edit_delay": 1,
        "target_test_reuse_rate": 1.0,
        "post_edit_diagnostic_calls": 2,
        "root_cause_switch_after_failed_verification": True,
    }


def test_convergence_metrics_report_no_edit_without_inventing_semantics() -> None:
    metrics = _collect_convergence_metrics(
        [
            {"type": "tool_called", "step": 1, "tool_call_id": "read", "tool": "read", "args": {}},
            {
                "type": "tool_called",
                "step": 2,
                "tool_call_id": "diagnose",
                "tool": "run_command",
                "args": {"argv": ["python", "-c", "print(1)"]},
            },
        ]
    )

    assert metrics["first_edit_step"] is None
    assert metrics["non_write_calls_before_first_edit"] == 2
    assert metrics["reproducer_to_edit_delay"] is None
    assert metrics["target_test_reuse_rate"] is None
    assert metrics["post_edit_diagnostic_calls"] == 0
    assert metrics["root_cause_switch_after_failed_verification"] is None
