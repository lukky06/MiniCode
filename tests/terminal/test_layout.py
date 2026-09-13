from importlib.util import find_spec


def test_legacy_python_fullscreen_layout_is_removed() -> None:
    assert find_spec("minicode_harness.terminal.layout") is None
