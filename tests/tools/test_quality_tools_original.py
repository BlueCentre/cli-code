"""
Tests for code quality tools.
"""
import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, ANY

# Direct import for coverage tracking
import src.cli_code.tools.quality_tools
from src.cli_code.tools.quality_tools import _run_quality_command, LinterCheckerTool, FormatterTool


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

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_linter_checker_with_defaults(self, mock_subprocess_run):
        """Test linter check with default parameters."""
        # Setup mock for subprocess.run
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Mocked Linter output - Defaults"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute()

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            ["ruff", "check", os.path.abspath(".")], # Use absolute path
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Mocked Linter output - Defaults" in result, f"Expected output not in result: {result}"

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_linter_checker_with_custom_path(self, mock_subprocess_run):
        """Test linter check with custom path."""
        # Setup mock
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Linter output for src"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process
        custom_path = "src/my_module"

        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute(path=custom_path)

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            ["ruff", "check", os.path.abspath(custom_path)], # Use absolute path
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Linter output for src" in result

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_linter_checker_with_custom_command(self, mock_subprocess_run):
        """Test linter check with custom linter command."""
        # Setup mock
        custom_linter_command = "flake8"
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Linter output - Custom Command"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute(linter_command=custom_linter_command)

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            ["flake8", os.path.abspath(".")], # Use absolute path
             capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Linter output - Custom Command" in result

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_linter_checker_with_complex_command(self, mock_subprocess_run):
        """Test linter check with complex command including arguments."""
        # Setup mock
        complex_linter_command = "flake8 --max-line-length=100"
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Linter output - Complex Command"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute(linter_command=complex_linter_command)

        # Verify results
        expected_cmd_list = ["flake8", "--max-line-length=100", os.path.abspath(".")] # Use absolute path
        mock_subprocess_run.assert_called_once_with(
            expected_cmd_list,
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Linter output - Complex Command" in result

    @patch("src.cli_code.tools.quality_tools.subprocess.run", side_effect=FileNotFoundError)
    def test_linter_checker_command_not_found(self, mock_subprocess_run):
        """Test linter check when the linter command is not found."""
        # Execute tool
        tool = LinterCheckerTool()
        result = tool.execute()

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            ["ruff", "check", os.path.abspath(".")], # Use absolute path
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Error: Command 'ruff' not found." in result

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

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_formatter_with_defaults(self, mock_subprocess_run):
        """Test formatter with default parameters."""
        # Setup mock
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Formatted code output - Defaults"
        mock_process.stderr = "files were modified"
        mock_subprocess_run.return_value = mock_process

        # Execute tool
        tool = FormatterTool()
        result = tool.execute()

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            ["black", os.path.abspath(".")], # Use absolute path
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Formatted code output - Defaults" in result
        assert "files were modified" in result

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_formatter_with_custom_path(self, mock_subprocess_run):
        """Test formatter with custom path."""
        # Setup mock
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Formatted code output - Custom Path"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process
        custom_path = "src/my_module"

        # Execute tool
        tool = FormatterTool()
        result = tool.execute(path=custom_path)

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            ["black", os.path.abspath(custom_path)], # Use absolute path
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Formatted code output - Custom Path" in result

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_formatter_with_custom_command(self, mock_subprocess_run):
        """Test formatter with custom formatter command."""
        # Setup mock
        custom_formatter_command = "isort"
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Formatted code output - Custom Command"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        # Execute tool
        tool = FormatterTool()
        result = tool.execute(formatter_command=custom_formatter_command)

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            [custom_formatter_command, os.path.abspath(".")], # Use absolute path, command directly
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Formatted code output - Custom Command" in result

    @patch("src.cli_code.tools.quality_tools.subprocess.run")
    def test_formatter_with_complex_command(self, mock_subprocess_run):
        """Test formatter with complex command including arguments."""
        # Setup mock
        formatter_base_command = "black"
        complex_formatter_command = f"{formatter_base_command} --line-length 88"
        mock_process = MagicMock(spec=subprocess.CompletedProcess)
        mock_process.returncode = 0
        mock_process.stdout = "Formatted code output - Complex Command"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        # Execute tool
        tool = FormatterTool()
        result = tool.execute(formatter_command=complex_formatter_command)

        # Verify results
        expected_cmd_list = [formatter_base_command, "--line-length", "88", os.path.abspath(".")] # Use absolute path
        mock_subprocess_run.assert_called_once_with(
            expected_cmd_list,
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Formatted code output - Complex Command" in result

    @patch("src.cli_code.tools.quality_tools.subprocess.run", side_effect=FileNotFoundError)
    def test_formatter_command_not_found(self, mock_subprocess_run):
        """Test formatter when the formatter command is not found."""
        # Execute tool
        tool = FormatterTool()
        result = tool.execute()

        # Verify results
        mock_subprocess_run.assert_called_once_with(
            ["black", os.path.abspath(".")], # Use absolute path
            capture_output=True, text=True, check=False, timeout=ANY
        )
        assert "Error: Command 'black' not found." in result

    def test_formatter_with_parent_directory_traversal(self):
        """Test formatter with parent directory traversal."""
        tool = FormatterTool()
        result = tool.execute(path="../dangerous")
        
        # Verify results
        assert "Error: Invalid path" in result
        assert "Cannot access parent directories" in result 