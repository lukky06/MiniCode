from importlib.util import find_spec


def test_legacy_terminal_application_is_removed() -> None:
    assert find_spec("minicode_harness.terminal.application") is None
