"""Tests for the Ragas CLI module."""

from typer.testing import CliRunner

from ragas.cli import app


def test_cli_help():
    """Test that the CLI help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Ragas CLI for running LLM evaluations" in result.stdout


def test_hello_world_help():
    """Test that the hello-world help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["hello-world", "--help"])
    assert result.exit_code == 0
    assert "Directory to run the hello world example in" in result.stdout


def test_evals_help():
    """Test that the evals help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["evals", "--help"])
    assert result.exit_code == 0
    assert "Run evaluations on a dataset" in result.stdout


if __name__ == "__main__":
    print("Running CLI tests...")
    test_cli_help()
    print("✓ CLI help test passed")
    test_hello_world_help()
    print("✓ Hello world help test passed")
    test_evals_help()
    print("✓ Evals help test passed")
    print("All CLI tests passed!")
