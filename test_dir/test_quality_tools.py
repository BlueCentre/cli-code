"""
Tests for code quality tools.
"""
import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from cli_code.tools.quality_tools import _run_quality_command, LinterCheckerTool, FormatterTool


class TestRunQualityCommand:
    """Tests for the _run_quality_command helper function."""

    @patch("subprocess.run")
    def test_run_quality_command_success(self, mock_run):
        """Test successful command execution."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Successful output"
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        # Execute function
        result = _run_quality_command(["test", "command"], "TestTool")
        
        # Verify results
        assert "TestTool Result (Exit Code: 0)" in result
        assert "Successful output" in result
        assert "-- Errors --" not in result
        mock_run.assert_called_once_with(
            ["test", "command"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120
        )

    @patch("subprocess.run")
    def test_run_quality_command_with_errors(self, mock_run):
        """Test command execution with errors."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = "Output"
        mock_process.stderr = "Error message"
        mock_run.return_value = mock_process
        
        # Execute function
        result = _run_quality_command(["test", "command"], "TestTool")
        
        # Verify results
        assert "TestTool Result (Exit Code: 1)" in result
        assert "Output" in result
        assert "-- Errors --" in result
        assert "Error message" in result

    @patch("subprocess.run")
    def test_run_quality_command_no_output(self, mock_run):
        """Test command execution with no output."""
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
    def test_run_quality_command_long_output(self, mock_run):
        """Test command execution with output that exceeds length limit."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "A" * 3000  # More than the 2000 character limit
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        # Execute function
        result = _run_quality_command(["test", "command"], "TestTool")
        
        # Verify results
        assert "... (output truncated)" in result
        assert len(result) < 3000

    def test_run_quality_command_file_not_found(self):
        """Test when the command is not found."""
        # Setup side effect
        with patch("subprocess.run", side_effect=FileNotFoundError("No such file or directory: 'nonexistent'")):
            # Execute function
            result = _run_quality_command(["nonexistent"], "TestTool")
            
            # Verify results
            assert "Error: Command 'nonexistent' not found" in result
            assert "Is 'nonexistent' installed and in PATH?" in result

    def test_run_quality_command_timeout(self):
        """Test when the command times out."""
        # Setup side effect
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="slow_command", timeout=120)):
            # Execute function
            result = _run_quality_command(["slow_command"], "TestTool")
            
            # Verify results
            assert "Error: TestTool run timed out" in result

    def test_run_quality_command_unexpected_error(self):
        """Test when an unexpected error occurs."""
        # Setup side effect
        with patch("subprocess.run", side_effect=Exception("Unexpected error")):
            # Execute function
            result = _run_quality_command(["command"], "TestTool")
            
            # Verify results
            assert "Error running TestTool" in result
            assert "Unexpected error" in result


class TestLinterCheckerTool:
    """Tests for the LinterCheckerTool class."""

    def test_init(self):
        """Test initialization of LinterCheckerTool."""
        tool = LinterCheckerTool()
        assert tool.name == "linter_checker"
        assert "Runs a code linter" in tool.description

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_linter_checker_with_defaults(self, mock_run_command):
        """Test linter check with default parameters."""
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

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_linter_checker_with_custom_path(self, mock_run_command):
        """Test linter check with custom path."""
        # Setup mock
        mock_run_command.return_value = "Linter output"
        
        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute(path="src")
        
        # Verify results
        mock_run_command.assert_called_once()
        args, kwargs = mock_run_command.call_args
        assert args[0] == ["ruff", "check", os.path.abspath("src")]

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_linter_checker_with_custom_command(self, mock_run_command):
        """Test linter check with custom linter command."""
        # Setup mock
        mock_run_command.return_value = "Linter output"
        
        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute(linter_command="flake8")
        
        # Verify results
        mock_run_command.assert_called_once()
        args, kwargs = mock_run_command.call_args
        assert args[0] == ["flake8", os.path.abspath(".")]

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_linter_checker_with_complex_command(self, mock_run_command):
        """Test linter check with complex command including arguments."""
        # Setup mock
        mock_run_command.return_value = "Linter output"
        
        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute(linter_command="flake8 --max-line-length=100")
        
        # Verify results
        mock_run_command.assert_called_once()
        args, kwargs = mock_run_command.call_args
        assert args[0] == ["flake8", "--max-line-length=100", os.path.abspath(".")]

    def test_linter_checker_with_parent_directory_traversal(self):
        """Test linter check with parent directory traversal."""
        tool = LinterCheckerTool()
        result = tool.execute(path="../dangerous")
        
        # Verify results
        assert "Error: Invalid path" in result
        assert "Cannot access parent directories" in result


class TestFormatterTool:
    """Tests for the FormatterTool class."""

    def test_init(self):
        """Test initialization of FormatterTool."""
        tool = FormatterTool()
        assert tool.name == "formatter"
        assert "Runs a code formatter" in tool.description

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_formatter_with_defaults(self, mock_run_command):
        """Test formatter with default parameters."""
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

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_formatter_with_custom_path(self, mock_run_command):
        """Test formatter with custom path."""
        # Setup mock
        mock_run_command.return_value = "Formatter output"
        
        # Execute tool
        tool = FormatterTool()
        result = tool.execute(path="src")
        
        # Verify results
        mock_run_command.assert_called_once()
        args, kwargs = mock_run_command.call_args
        assert args[0] == ["black", os.path.abspath("src")]

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_formatter_with_custom_command(self, mock_run_command):
        """Test formatter with custom formatter command."""
        # Setup mock
        mock_run_command.return_value = "Formatter output"
        
        # Execute tool
        tool = FormatterTool()
        result = tool.execute(formatter_command="prettier")
        
        # Verify results
        mock_run_command.assert_called_once()
        args, kwargs = mock_run_command.call_args
        assert args[0] == ["prettier", os.path.abspath(".")]

    @patch("cli_code.tools.quality_tools._run_quality_command")
    def test_formatter_with_complex_command(self, mock_run_command):
        """Test formatter with complex command including arguments."""
        # Setup mock
        mock_run_command.return_value = "Formatter output"
        
        # Execute tool
        tool = FormatterTool()
        result = tool.execute(formatter_command="black -l 100")
        
        # Verify results
        mock_run_command.assert_called_once()
        args, kwargs = mock_run_command.call_args
        assert args[0] == ["black", "-l", "100", os.path.abspath(".")]

    def test_formatter_with_parent_directory_traversal(self):
        """Test formatter with parent directory traversal."""
        tool = FormatterTool()
        result = tool.execute(path="../dangerous")
        
        # Verify results
        assert "Error: Invalid path" in result
        assert "Cannot access parent directories" in result 