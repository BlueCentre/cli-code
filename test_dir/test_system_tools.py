"""
Tests for system tools.
"""
import subprocess
import pytest
from unittest.mock import patch, MagicMock

# Direct import for coverage tracking
import src.cli_code.tools.system_tools
from src.cli_code.tools.system_tools import BashTool


def test_bash_tool_init():
    """Test BashTool initialization."""
    tool = BashTool()
    assert tool.name == "bash"
    assert tool.description == "Execute a bash command"
    assert isinstance(tool.BANNED_COMMANDS, list)
    assert len(tool.BANNED_COMMANDS) > 0


def test_bash_tool_banned_command():
    """Test BashTool rejects banned commands."""
    tool = BashTool()
    
    # Try a banned command (using the first one in the list)
    banned_cmd = tool.BANNED_COMMANDS[0]
    result = tool.execute(f"{banned_cmd} some_args")
    
    assert "not allowed for security reasons" in result
    assert banned_cmd in result


@patch("subprocess.Popen")
def test_bash_tool_successful_command(mock_popen):
    """Test BashTool executes commands successfully."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = ("Command output", "")
    mock_popen.return_value = mock_process
    
    # Execute a simple command
    tool = BashTool()
    result = tool.execute("echo 'hello world'")
    
    # Verify results
    assert result == "Command output"
    mock_popen.assert_called_once()
    mock_process.communicate.assert_called_once()


@patch("subprocess.Popen")
def test_bash_tool_command_error(mock_popen):
    """Test BashTool handling of command errors."""
    # Setup mock to simulate command failure
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = ("", "Command failed")
    mock_popen.return_value = mock_process
    
    # Execute a command that will fail
    tool = BashTool()
    result = tool.execute("invalid_command")
    
    # Verify error handling
    assert "exited with status 1" in result
    assert "STDERR:\nCommand failed" in result
    mock_popen.assert_called_once()


@patch("subprocess.Popen")
def test_bash_tool_timeout(mock_popen):
    """Test BashTool handling of timeouts."""
    # Setup mock to simulate timeout
    mock_process = MagicMock()
    mock_process.communicate.side_effect = subprocess.TimeoutExpired("cmd", 1)
    mock_popen.return_value = mock_process
    
    # Execute command with short timeout
    tool = BashTool()
    result = tool.execute("sleep 10", timeout=1000)  # 1 second timeout
    
    # Verify timeout handling
    assert "Command timed out" in result
    mock_process.kill.assert_called_once()


def test_bash_tool_invalid_timeout():
    """Test BashTool with invalid timeout value."""
    with patch("subprocess.Popen") as mock_popen:
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("Command output", "")
        mock_popen.return_value = mock_process
        
        # Execute with invalid timeout
        tool = BashTool()
        result = tool.execute("echo test", timeout="not-a-number")
        
        # Verify default timeout was used
        mock_process.communicate.assert_called_once_with(timeout=30)
        assert result == "Command output"


@patch("subprocess.Popen")
def test_bash_tool_general_exception(mock_popen):
    """Test BashTool handling of general exceptions."""
    # Setup mock to raise an exception
    mock_popen.side_effect = Exception("Something went wrong")
    
    # Execute command
    tool = BashTool()
    result = tool.execute("some command")
    
    # Verify exception handling
    assert "Error executing command" in result
    assert "Something went wrong" in result 