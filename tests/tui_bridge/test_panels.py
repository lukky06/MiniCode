from __future__ import annotations

from minicode_harness.tui_bridge.panels import panel_title


def test_panel_titles_are_presentation_only() -> None:
    assert panel_title("help") == "Help"
    assert panel_title("recover") == "Recover"
    assert panel_title("custom_command") == "Custom Command"
