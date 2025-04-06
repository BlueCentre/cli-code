"""
Comprehensive tests for the system_tools module.
"""

import pytest
from unittest.mock import patch, MagicMock
import subprocess
import time

from cli_code.tools.system_tools import BashTool


class TestBashTool:
    """Test cases for the BashTool class."""
    
    def test_init(self):
        """Test initialization of BashTool."""
        tool = BashTool()
        assert tool.name == "bash"
        assert tool.description == "Execute a bash command"
        assert isinstance(tool.BANNED_COMMANDS, list)
        assert len(tool.BANNED_COMMANDS) > 0
    
    def test_banned_commands(self):
        """Test that banned commands are rejected."""
        tool = BashTool()
        
        # Test each banned command
        for banned_cmd in tool.BANNED_COMMANDS:
            result = tool.execute(f"{banned_cmd} some_args")
            assert "not allowed for security reasons" in result
            assert banned_cmd in result
    
    def test_execute_simple_command(self):
        """Test executing a simple command."""
        tool = BashTool()
        result = tool.execute("echo 'hello world'")
        assert "hello world" in result
    
    def test_execute_with_error(self):
        """Test executing a command that returns an error."""
        tool = BashTool()
        result = tool.execute("ls /nonexistent_directory")
        assert "Command exited with status" in result
        assert "STDERR" in result
    
    @patch('subprocess.Popen')
    def test_timeout_handling(self, mock_popen):
        """Test handling of command timeouts."""
        # Setup mock to simulate timeout
        mock_process = MagicMock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=0.1)
        mock_popen.return_value = mock_process
        
        tool = BashTool()
        result = tool.execute("sleep 100", timeout=100)  # 100ms timeout
        
        assert "Command timed out" in result
    
    @patch('subprocess.Popen')
    def test_exception_handling(self, mock_popen):
        """Test general exception handling."""
        # Setup mock to raise exception
        mock_popen.side_effect = Exception("Test exception")
        
        tool = BashTool()
        result = tool.execute("echo test")
        
        assert "Error executing command" in result
        assert "Test exception" in result
    
    def test_timeout_conversion(self):
        """Test conversion of timeout parameter."""
        tool = BashTool()
        
        # Test with invalid timeout
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = ("output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            tool.execute("echo test", timeout="invalid")
            
            # Should use default timeout (30 seconds)
            mock_process.communicate.assert_called_with(timeout=30)
    
    def test_long_output_handling(self):
        """Test handling of commands with large output."""
        tool = BashTool()
        
        # Generate a large output
        result = tool.execute("python -c \"print('x' * 10000)\"")
        
        # Verify the tool can handle large outputs
        assert len(result) >= 10000
        assert result.count('x') >= 10000
    
    def test_command_with_arguments(self):
        """Test executing a command with arguments."""
        tool = BashTool()
        
        # Test with multiple arguments
        result = tool.execute("echo arg1 arg2 arg3")
        assert "arg1 arg2 arg3" in result
        
        # Test with quoted arguments
        result = tool.execute("echo 'argument with spaces'")
        assert "argument with spaces" in result
        
        # Test with environment variables
        result = tool.execute("echo $HOME")
        assert len(result.strip()) > 0  # Should have some content 