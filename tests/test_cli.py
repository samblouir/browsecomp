from typer.testing import CliRunner

from browsecomp250.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_subset_command() -> None:
    result = CliRunner().invoke(app, ["subset"])
    assert result.exit_code == 0
    assert "250" in result.stdout
