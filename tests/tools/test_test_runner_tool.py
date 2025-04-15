"""
Tests for the TestRunnerTool class.
"""

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.cli_code.tools.test_runner import TestRunnerTool


@pytest.fixture
def test_runner_tool():
    """Provides an instance of TestRunnerTool."""
    return TestRunnerTool()


def test_initialization():
    """Test that the tool initializes correctly with the right name and description."""
    tool = TestRunnerTool()
    assert tool.name == "test_runner"
    assert "test runner" in tool.description.lower()
    assert "pytest" in tool.description


def test_successful_test_run(test_runner_tool):
    """Test executing a successful test run."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock to simulate a successful test run
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "All tests passed!"
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        # Execute the tool
        result = test_runner_tool.execute(test_path="tests/")

        # Verify the command that was run
        mock_run.assert_called_once_with(
            ["pytest", "tests/"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        # Check the result
        assert "SUCCESS" in result
        assert "All tests passed!" in result


def test_failed_test_run(test_runner_tool):
    """Test executing a failed test run."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock to simulate a failed test run
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = "1 test failed"
        mock_process.stderr = "Error details"
        mock_run.return_value = mock_process

        # Execute the tool
        result = test_runner_tool.execute()

        # Verify the command that was run
        mock_run.assert_called_once_with(
            ["pytest"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        # Check the result
        assert "FAILED" in result
        assert "1 test failed" in result
        assert "Error details" in result


def test_with_options(test_runner_tool):
    """Test executing tests with additional options."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "All tests passed with options!"
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        # Execute the tool with options
        result = test_runner_tool.execute(options="-v --cov=src --junit-xml=results.xml")

        # Verify the command that was run with all the options
        mock_run.assert_called_once_with(
            ["pytest", "-v", "--cov=src", "--junit-xml=results.xml"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        # Check the result
        assert "SUCCESS" in result
        assert "All tests passed with options!" in result


def test_with_different_runner(test_runner_tool):
    """Test using a different test runner than pytest."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Tests passed with unittest!"
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        # Execute the tool with a different runner command
        result = test_runner_tool.execute(runner_command="python -m unittest")

        # Verify the command that was run
        mock_run.assert_called_once_with(
            ["python -m unittest"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        # Check the result
        assert "SUCCESS" in result
        assert "using 'python -m unittest'" in result
        assert "Tests passed with unittest!" in result


def test_command_not_found(test_runner_tool):
    """Test handling of command not found error."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock to raise FileNotFoundError
        mock_run.side_effect = FileNotFoundError("No such file or directory")

        # Execute the tool with a command that doesn't exist
        result = test_runner_tool.execute(runner_command="nonexistent_command")

        # Check the result
        assert "Error" in result
        assert "not found" in result
        assert "nonexistent_command" in result


def test_timeout_error(test_runner_tool):
    """Test handling of timeout error."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock to raise TimeoutExpired
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=300)

        # Execute the tool
        result = test_runner_tool.execute()

        # Check the result
        assert "Error" in result
        assert "exceeded the timeout limit" in result


def test_general_error(test_runner_tool):
    """Test handling of general unexpected errors."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock to raise a general exception
        mock_run.side_effect = Exception("Something went wrong")

        # Execute the tool
        result = test_runner_tool.execute()

        # Check the result
        assert "Error" in result
        assert "Something went wrong" in result


def test_invalid_options_parsing(test_runner_tool):
    """Test handling of invalid options string."""
    with (
        patch("subprocess.run") as mock_run,
        patch("shlex.split") as mock_split,
        patch("src.cli_code.tools.test_runner.log") as mock_log,
    ):
        # Configure shlex.split to raise ValueError
        mock_split.side_effect = ValueError("Invalid option string")

        # Configure subprocess.run for normal execution after the error
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Tests passed anyway"
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        # Execute the tool with invalid options
        result = test_runner_tool.execute(options="--invalid='unclosed quote")

        # Verify warning was logged
        mock_log.warning.assert_called_once()

        # Verify run was called without the options
        mock_run.assert_called_once_with(
            ["pytest"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        # Check the result
        assert "SUCCESS" in result


def test_no_tests_collected(test_runner_tool):
    """Test handling of pytest exit code 5 (no tests collected)."""
    with patch("subprocess.run") as mock_run:
        # Configure the mock
        mock_process = MagicMock()
        mock_process.returncode = 5
        mock_process.stdout = "No tests collected"
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        # Execute the tool
        result = test_runner_tool.execute()

        # Check the result
        assert "FAILED" in result
        assert "exit code 5" in result.lower()
        assert "no tests were found" in result.lower()
