"""
Tests for system tools.
"""
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from cli_code.tools.system_tools import BashTool


def test_bash_tool_init():
    """Test BashTool initialization."""
    tool = BashTool()
    assert tool.name == "bash"
    assert tool.description == "Execute a bash command"
    assert isinstance(tool.BANNED_COMMANDS, list)
    assert len(tool.BANNED_COMMANDS) > 0


@patch('subprocess.Popen')
def test_bash_tool_execute_success(mock_popen):
    """Test successful command execution with BashTool."""
    # Setup the mock
    process_mock = MagicMock()
    process_mock.returncode = 0
    process_mock.communicate.return_value = ("Command output", "")
    mock_popen.return_value = process_mock
    
    # Execute the tool
    tool = BashTool()
    result = tool.execute("echo 'Hello World'")
    
    # Verify results
    assert result == "Command output"
    mock_popen.assert_called_once()
    process_mock.communicate.assert_called_once()


@patch('subprocess.Popen')
def test_bash_tool_execute_error(mock_popen):
    """Test command execution error with BashTool."""
    # Setup the mock
    process_mock = MagicMock()
    process_mock.returncode = 1
    process_mock.communicate.return_value = ("", "Command error")
    mock_popen.return_value = process_mock
    
    # Execute the tool
    tool = BashTool()
    result = tool.execute("invalid_command")
    
    # Verify results
    assert "Command exited with status 1" in result
    assert "Command error" in result
    mock_popen.assert_called_once()
    process_mock.communicate.assert_called_once()


def test_bash_tool_execute_banned_command():
    """Test execution of banned commands."""
    tool = BashTool()
    
    # Try to execute each banned command
    for banned_cmd in tool.BANNED_COMMANDS:
        result = tool.execute(banned_cmd)
        assert f"Error: The command '{banned_cmd}' is not allowed for security reasons." in result


@patch('subprocess.Popen')
def test_bash_tool_execute_timeout(mock_popen):
    """Test command timeout with BashTool."""
    # Setup the mock
    process_mock = MagicMock()
    process_mock.communicate.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=1)
    mock_popen.return_value = process_mock
    
    # Execute the tool with a short timeout
    tool = BashTool()
    result = tool.execute("sleep 10", timeout=1000)
    
    # Verify results
    assert "Error: Command timed out after" in result
    mock_popen.assert_called_once()
    process_mock.communicate.assert_called_once()
    process_mock.kill.assert_called_once()


@patch('subprocess.Popen')
def test_bash_tool_execute_invalid_timeout(mock_popen):
    """Test execution with invalid timeout value."""
    # Setup the mock
    process_mock = MagicMock()
    process_mock.returncode = 0
    process_mock.communicate.return_value = ("Command output", "")
    mock_popen.return_value = process_mock
    
    # Execute the tool with an invalid timeout
    tool = BashTool()
    result = tool.execute("echo 'Hello'", timeout="invalid")
    
    # Verify results
    assert result == "Command output"
    # Should use default timeout (30s)
    process_mock.communicate.assert_called_once_with(timeout=30)


@patch('subprocess.Popen')
def test_bash_tool_execute_general_exception(mock_popen):
    """Test general exception handling during execution."""
    # Setup the mock to raise an exception
    mock_popen.side_effect = Exception("Test exception")
    
    # Execute the tool
    tool = BashTool()
    result = tool.execute("echo 'Hello'")
    
    # Verify results
    assert "Error executing command: Test exception" in result 