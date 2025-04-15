"""
Tests for quality_tools module.
"""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

# Direct import for coverage tracking
import src.cli_code.tools.quality_tools
from src.cli_code.tools.quality_tools import FormatterTool, LinterCheckerTool, _run_quality_command


def test_linter_checker_tool_init():
    """Test LinterCheckerTool initialization."""
    tool = LinterCheckerTool()
    assert tool.name == "linter_checker"
    assert "Runs a code linter" in tool.description


def test_formatter_tool_init():
    """Test FormatterTool initialization."""
    tool = FormatterTool()
    assert tool.name == "formatter"
    assert "Runs a code formatter" in tool.description


@patch("subprocess.run")
def test_run_quality_command_success(mock_run):
    """Test _run_quality_command with successful command execution."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Command output"
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    # Execute function
    result = _run_quality_command(["test", "command"], "TestTool")

    # Verify results
    assert "TestTool Result (Exit Code: 0)" in result
    assert "Command output" in result
    assert "-- Errors --" not in result
    mock_run.assert_called_once_with(["test", "command"], capture_output=True, text=True, check=False, timeout=120)


@patch("subprocess.run")
def test_run_quality_command_with_errors(mock_run):
    """Test _run_quality_command with command that outputs errors."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = "Command output"
    mock_process.stderr = "Error message"
    mock_run.return_value = mock_process

    # Execute function
    result = _run_quality_command(["test", "command"], "TestTool")

    # Verify results
    assert "TestTool Result (Exit Code: 1)" in result
    assert "Command output" in result
    assert "-- Errors --" in result
    assert "Error message" in result


@patch("subprocess.run")
def test_run_quality_command_no_output(mock_run):
    """Test _run_quality_command with command that produces no output."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = ""
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    # Execute function
    result = _run_quality_command(["test", "command"], "TestTool")

    # Verify results
    assert "TestTool Result (Exit Code: 0)" in result
    assert "(No output)" in result


@patch("subprocess.run")
def test_run_quality_command_long_output(mock_run):
    """Test _run_quality_command with command that produces very long output."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "A" * 3000  # Longer than 2000 char limit
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    # Execute function
    result = _run_quality_command(["test", "command"], "TestTool")

    # Verify results
    assert "... (output truncated)" in result
    assert len(result) < 3000


def test_run_quality_command_file_not_found():
    """Test _run_quality_command with non-existent command."""
    # Set up side effect
    with patch("subprocess.run", side_effect=FileNotFoundError("No such file or directory: 'nonexistent'")):
        # Execute function
        result = _run_quality_command(["nonexistent"], "TestTool")

        # Verify results
        assert "Error: Command 'nonexistent' not found" in result
        assert "Is 'nonexistent' installed and in PATH?" in result


def test_run_quality_command_timeout():
    """Test _run_quality_command with command that times out."""
    # Set up side effect
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="slow_command", timeout=120)):
        # Execute function
        result = _run_quality_command(["slow_command"], "TestTool")

        # Verify results
        assert "Error: TestTool run timed out" in result
        assert "2 minutes" in result


def test_run_quality_command_unexpected_error():
    """Test _run_quality_command with unexpected error."""
    # Set up side effect
    with patch("subprocess.run", side_effect=Exception("Unexpected error")):
        # Execute function
        result = _run_quality_command(["command"], "TestTool")

        # Verify results
        assert "Error running TestTool" in result
        assert "Unexpected error" in result


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_linter_checker_with_defaults(mock_run_command):
    """Test LinterCheckerTool with default parameters."""
    # Setup mock
    mock_run_command.return_value = "Linter output"

    # Execute tool
    tool = LinterCheckerTool()
    result = tool.execute()

    # Verify results
    assert result == "Linter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["ruff", "check", os.path.abspath(".")]
    assert args[1] == "Linter"


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_linter_checker_with_custom_path(mock_run_command):
    """Test LinterCheckerTool with custom path."""
    # Setup mock
    mock_run_command.return_value = "Linter output"

    # Execute tool
    tool = LinterCheckerTool()
    result = tool.execute(path="src")

    # Verify results
    assert result == "Linter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["ruff", "check", os.path.abspath("src")]


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_linter_checker_with_custom_command(mock_run_command):
    """Test LinterCheckerTool with custom linter command."""
    # Setup mock
    mock_run_command.return_value = "Linter output"

    # Execute tool
    tool = LinterCheckerTool()
    result = tool.execute(linter_command="flake8")

    # Verify results
    assert result == "Linter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["flake8", os.path.abspath(".")]


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_linter_checker_with_complex_command(mock_run_command):
    """Test LinterCheckerTool with complex command including arguments."""
    # Setup mock
    mock_run_command.return_value = "Linter output"

    # Execute tool
    tool = LinterCheckerTool()
    result = tool.execute(linter_command="flake8 --max-line-length=100")

    # Verify results
    assert result == "Linter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["flake8", "--max-line-length=100", os.path.abspath(".")]


def test_linter_checker_with_parent_directory_traversal():
    """Test LinterCheckerTool with path containing parent directory traversal."""
    tool = LinterCheckerTool()
    result = tool.execute(path="../dangerous")

    # Verify results
    assert "Error: Invalid path" in result
    assert "Cannot access parent directories" in result


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_formatter_with_defaults(mock_run_command):
    """Test FormatterTool with default parameters."""
    # Setup mock
    mock_run_command.return_value = "Formatter output"

    # Execute tool
    tool = FormatterTool()
    result = tool.execute()

    # Verify results
    assert result == "Formatter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["black", os.path.abspath(".")]
    assert args[1] == "Formatter"


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_formatter_with_custom_path(mock_run_command):
    """Test FormatterTool with custom path."""
    # Setup mock
    mock_run_command.return_value = "Formatter output"

    # Execute tool
    tool = FormatterTool()
    result = tool.execute(path="src")

    # Verify results
    assert result == "Formatter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["black", os.path.abspath("src")]


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_formatter_with_custom_command(mock_run_command):
    """Test FormatterTool with custom formatter command."""
    # Setup mock
    mock_run_command.return_value = "Formatter output"

    # Execute tool
    tool = FormatterTool()
    result = tool.execute(formatter_command="prettier")

    # Verify results
    assert result == "Formatter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["prettier", os.path.abspath(".")]


@patch("src.cli_code.tools.quality_tools._run_quality_command")
def test_formatter_with_complex_command(mock_run_command):
    """Test FormatterTool with complex command including arguments."""
    # Setup mock
    mock_run_command.return_value = "Formatter output"

    # Execute tool
    tool = FormatterTool()
    result = tool.execute(formatter_command="prettier --write")

    # Verify results
    assert result == "Formatter output"
    mock_run_command.assert_called_once()
    args, kwargs = mock_run_command.call_args
    assert args[0] == ["prettier", "--write", os.path.abspath(".")]


def test_formatter_with_parent_directory_traversal():
    """Test FormatterTool with path containing parent directory traversal."""
    tool = FormatterTool()
    result = tool.execute(path="../dangerous")

    # Verify results
    assert "Error: Invalid path" in result
    assert "Cannot access parent directories" in result
