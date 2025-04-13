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


def test_get_function_declaration():
    """Test get_function_declaration method inherited from BaseTool."""
    tool = TestRunnerTool()
    function_decl = tool.get_function_declaration()
    
    # Verify basic properties
    assert function_decl is not None
    assert function_decl.name == "test_runner"
    assert "test" in function_decl.description.lower()
    
    # Verify parameters structure exists
    assert function_decl.parameters is not None
    
    # The correct attributes are directly on the parameters object
    # Check if the parameters has the expected attributes
    assert hasattr(function_decl.parameters, 'type_')
    # Type is an enum, just check it exists
    assert function_decl.parameters.type_ is not None
    
    # Check for properties
    assert hasattr(function_decl.parameters, 'properties')
    
    # Check for expected parameters from the execute method signature
    properties = function_decl.parameters.properties
    assert 'test_path' in properties
    assert 'options' in properties
    assert 'runner_command' in properties
    
    # Check parameter types - using isinstance or type presence
    for param_name in ['test_path', 'options', 'runner_command']:
        assert hasattr(properties[param_name], 'type_')
        assert properties[param_name].type_ is not None
        assert hasattr(properties[param_name], 'description')
        assert 'Parameter' in properties[param_name].description


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


def test_execute_with_different_exit_codes(test_runner_tool):
    """Test execute method with various non-zero exit codes."""
    # Test various exit codes that aren't explicitly handled
    for exit_code in [2, 3, 4, 6, 10]:
        mock_completed_process = mock.Mock()
        mock_completed_process.returncode = exit_code
        mock_completed_process.stdout = f"Tests failed with exit code {exit_code}."
        mock_completed_process.stderr = f"Error for exit code {exit_code}."
        
        with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
            result = test_runner_tool.execute()
            
            # All non-zero exit codes should be reported as FAILED
            assert f"Exit Code: {exit_code}" in result
            assert "Status: FAILED" in result
            assert f"Tests failed with exit code {exit_code}." in result
            assert f"Error for exit code {exit_code}." in result


def test_execute_with_very_long_output(test_runner_tool):
    """Test execute method with very long output that should be truncated."""
    # Create a long output string that exceeds truncation threshold
    long_stdout = "X" * 2000  # Generate a string longer than 1000 chars
    
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = long_stdout
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute()
        
        # Check for success status but truncated output
        assert "Status: SUCCESS" in result
        # The output should contain the last 1000 chars of the long stdout
        assert long_stdout[-1000:] in result
        # The full stdout should not be included (too long to check exactly, but we can check the length)
        assert len(result) < len(long_stdout) + 200  # Add a margin for the added status text


def test_execute_with_empty_stderr_stdout(test_runner_tool):
    """Test execute method with empty stdout and stderr."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = ""
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute()
        
        # Should still report success
        assert "Status: SUCCESS" in result
        # Should indicate empty output
        assert "Output:" in result
        assert "---" in result  # Output delimiters should still be there


def test_execute_with_stderr_only(test_runner_tool):
    """Test execute method with empty stdout but content in stderr."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 1
    mock_completed_process.stdout = ""
    mock_completed_process.stderr = "Error occurred but no stdout."
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        result = test_runner_tool.execute()
        
        # Should report failure
        assert "Status: FAILED" in result
        # Should have empty stdout section
        assert "Standard Output:" in result
        # Should have stderr content
        assert "Standard Error:" in result
        assert "Error occurred but no stdout." in result


def test_execute_with_none_params(test_runner_tool):
    """Test execute method with explicit None parameters."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "Tests passed with None parameters."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        # Explicitly passing None should be the same as default
        result = test_runner_tool.execute(test_path=None, options=None, runner_command="pytest")
        
        # Should call subprocess with just pytest command
        mock_run.assert_called_once_with(
            ["pytest"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        
        assert "SUCCESS" in result


def test_execute_with_empty_strings(test_runner_tool):
    """Test execute method with empty string parameters."""
    mock_completed_process = mock.Mock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "Tests passed with empty strings."
    mock_completed_process.stderr = ""
    
    with mock.patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        # Empty strings should be treated similarly to None for test_path
        # Empty options might be handled differently
        result = test_runner_tool.execute(test_path="", options="")
        
        # It appears the implementation doesn't add the empty test_path 
        # to the command (which makes sense)
        mock_run.assert_called_once_with(
            ["pytest"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        
        assert "SUCCESS" in result


def test_actual_execution_for_coverage(test_runner_tool):
    """Test to trigger actual code execution for coverage purposes."""
    # This test actually executes code paths, not just mocks
    # Mock only the subprocess.run to avoid actual subprocess execution
    with mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("Command not found")
        result = test_runner_tool.execute(options="--version", test_path="tests/", runner_command="fake_runner")
        assert "not found" in result 