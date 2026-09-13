from typer.main import get_command
from typer.testing import CliRunner

from minicode_harness.cli import app


runner = CliRunner()


def test_swebench_command_exposes_adapter_options() -> None:
    result = runner.invoke(app, ["bench", "swebench", "--help"])
    root = get_command(app)
    swebench = root.commands["bench"].commands["swebench"]
    options = {
        option
        for parameter in swebench.params
        for option in getattr(parameter, "opts", [])
    }
    max_steps = next(
        parameter
        for parameter in swebench.params
        if parameter.name == "max_steps"
    )

    assert result.exit_code == 0
    assert max_steps.default == 40
    assert {
        "--dataset",
        "--instance-id",
        "--repo-cache",
        "--resume",
        "--retry-failed",
        "--evaluator-python",
        "--docker-image",
    }.issubset(options)
