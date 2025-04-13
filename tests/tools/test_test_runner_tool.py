import pytest
from unittest import mock
import subprocess
import shlex
import os
import logging

# Import directly to ensure coverage
from src.cli_code.tools.test_runner import TestRunnerTool, log

# Create an instance to force coverage to collect data
_ensure_coverage = TestRunnerTool()

@pytest.fixture
def test_runner_tool():
    """Provides an instance of TestRunnerTool."""
    return TestRunnerTool()


def test_direct_initialization():
    """Test direct initialization of TestRunnerTool to ensure coverage."""
    tool = TestRunnerTool()
    assert tool.name == "test_runner"
    assert "test" in tool.description.lower()
    
    # Create a simple command to execute a branch of the code
    # This gives us some coverage without actually running subprocesses
    with mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("Command not found")
        result = tool.execute(options="--version", test_path="tests/", runner_command="fake_runner") 
        assert "not found" in result


def test_execute_successful_run(test_runner_tool):
    """Test execute method with a successful test run."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "All tests passed successfully."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute()
        
        # Verify subprocess was called with correct arguments
        mock_run.assert_called_once_with(
            ["pytest"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        
        # Check the output
        assert "Test run using 'pytest' completed" in result
        assert "Exit Code: 0" in result
        assert "Status: SUCCESS" in result
        assert "All tests passed successfully." in result


def test_execute_failed_run(test_runner_tool):
    """Test execute method with a failed test run."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 1
    mock_completed_process.stdout = "Test failures occurred."
    mock_completed_process.stderr = "Error details."
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute()
        
        # Verify subprocess was called correctly
        mock_run.assert_called_once()
        
        # Check the output
        assert "Test run using 'pytest' completed" in result
        assert "Exit Code: 1" in result
        assert "Status: FAILED" in result
        assert "Test failures occurred." in result
        assert "Error details." in result


def test_execute_with_test_path(test_runner_tool):
    """Test execute method with a specific test path."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "Tests in specific path passed."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute(test_path="tests/specific_test.py")
        
        # Verify subprocess was called with correct arguments including the test path
        mock_run.assert_called_once_with(
            ["pytest", "tests/specific_test.py"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        
        assert "SUCCESS" in result


def test_execute_with_options(test_runner_tool):
    """Test execute method with command line options."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "Tests with options passed."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        with mock.patch("shlex.split", return_value=["-v", "--cov"]) as mock_split:
            result = test_runner_tool.execute(options="-v --cov")
            
            # Verify shlex.split was called with the options string
            mock_split.assert_called_once_with("-v --cov")
            
            # Verify subprocess was called with correct arguments including the options
            mock_run.assert_called_once_with(
                ["pytest", "-v", "--cov"],
                capture_output=True,
                text=True,
                check=False,
                timeout=300
            )
            
            assert "SUCCESS" in result


def test_execute_with_custom_runner(test_runner_tool):
    """Test execute method with a custom runner command."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "Tests with custom runner passed."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute(runner_command="nose2")
        
        # Verify subprocess was called with the custom runner
        mock_run.assert_called_once_with(
            ["nose2"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        
        assert "Test run using 'nose2' completed" in result
        assert "SUCCESS" in result


def test_execute_with_invalid_options(test_runner_tool):
    """Test execute method with invalid options that cause a ValueError in shlex.split."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "Tests run without options."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        with mock.patch("shlex.split", side_effect=ValueError("Invalid options")) as mock_split:
            with mock.patch("src.cli_code.tools.test_runner.log") as mock_log:
                result = test_runner_tool.execute(options="invalid\"options")
                
                # Verify shlex.split was called with the options string
                mock_split.assert_called_once_with("invalid\"options")
                
                # Verify warning was logged
                mock_log.warning.assert_called_once()
                
                # Verify subprocess was called without the options
                mock_run.assert_called_once_with(
                    ["pytest"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300
                )
                
                assert "SUCCESS" in result


def test_execute_command_not_found(test_runner_tool):
    """Test execute method when the runner command is not found."""
    with mock.patch("subprocess.run", side_effect=FileNotFoundError("Command not found")) as mock_run:
        result = test_runner_tool.execute()
        
        # Verify error message
        assert "Error: Test runner command 'pytest' not found" in result


def test_execute_timeout(test_runner_tool):
    """Test execute method when the command times out."""
    with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pytest", 300)) as mock_run:
        result = test_runner_tool.execute()
        
        # Verify error message
        assert "Error: Test run exceeded the timeout limit" in result


def test_execute_unexpected_error(test_runner_tool):
    """Test execute method with an unexpected exception."""
    with mock.patch("subprocess.run", side_effect=Exception("Unexpected error")) as mock_run:
        result = test_runner_tool.execute()
        
        # Verify error message
        assert "Error: An unexpected error occurred" in result


def test_execute_no_tests_collected(test_runner_tool):
    """Test execute method when no tests are collected (exit code 5)."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 5
    mock_completed_process.stdout = "No tests collected."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute()
        
        # Check that the specific note about exit code 5 is included
        assert "Exit Code: 5" in result
        assert "FAILED" in result
        assert "Pytest exit code 5 often means no tests were found or collected" in result


def test_actual_execution_for_coverage(test_runner_tool):
    """Test to trigger actual code execution for coverage purposes."""
    # This test actually executes code paths, not just mocks
    # Mock only the subprocess.run to avoid actual subprocess execution
    with mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("Command not found")
        result = test_runner_tool.execute(options="--version", test_path="tests/", runner_command="fake_runner")
        assert "not found" in result 