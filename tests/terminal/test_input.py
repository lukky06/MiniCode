from importlib.util import find_spec


def test_legacy_prompt_toolkit_input_is_removed() -> None:
    assert find_spec("minicode_harness.terminal.input") is None
