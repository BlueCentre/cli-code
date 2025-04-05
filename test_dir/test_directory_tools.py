"""
Tests for directory tools.
"""
import os
import pytest
import subprocess
from unittest.mock import patch, MagicMock

from cli_code.tools.directory_tools import CreateDirectoryTool, LsTool


class TestCreateDirectoryTool:
    """Tests for the CreateDirectoryTool."""

    def test_init(self):
        """Test initialization of the CreateDirectoryTool."""
        tool = CreateDirectoryTool()
        assert tool.name == "create_directory"
        assert "Creates a new directory" in tool.description

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_create_directory_success(self, mock_makedirs, mock_exists):
        """Test successful directory creation."""
        # Setup mocks
        mock_exists.return_value = False
        
        # Execute the tool
        tool = CreateDirectoryTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Successfully created directory" in result
        mock_makedirs.assert_called_once()
        mock_exists.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_create_directory_already_exists(self, mock_isdir, mock_exists):
        """Test creating a directory that already exists."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isdir.return_value = True
        
        # Execute the tool
        tool = CreateDirectoryTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Directory already exists" in result
        mock_exists.assert_called_once()
        mock_isdir.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_create_directory_path_not_directory(self, mock_isdir, mock_exists):
        """Test creating a directory when a file with the same name exists."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isdir.return_value = False
        
        # Execute the tool
        tool = CreateDirectoryTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Error: Path exists but is not a directory" in result
        mock_exists.assert_called_once()
        mock_isdir.assert_called_once()

    def test_create_directory_parent_traversal(self):
        """Test attempt to create directory with parent directory traversal."""
        tool = CreateDirectoryTool()
        result = tool.execute("../dangerous_dir")
        
        # Verify results
        assert "Error: Invalid path" in result
        assert "Cannot access parent directories" in result

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_create_directory_os_error(self, mock_makedirs, mock_exists):
        """Test OS error during directory creation."""
        # Setup mocks
        mock_exists.return_value = False
        mock_makedirs.side_effect = OSError("Permission denied")
        
        # Execute the tool
        tool = CreateDirectoryTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Error creating directory" in result
        assert "Permission denied" in result
        mock_makedirs.assert_called_once()

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_create_directory_unexpected_error(self, mock_makedirs, mock_exists):
        """Test unexpected error during directory creation."""
        # Setup mocks
        mock_exists.return_value = False
        mock_makedirs.side_effect = Exception("Unexpected error")
        
        # Execute the tool
        tool = CreateDirectoryTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Error creating directory" in result
        assert "Unexpected error" in result
        mock_makedirs.assert_called_once()


class TestLsTool:
    """Tests for the LsTool."""

    def test_init(self):
        """Test initialization of the LsTool."""
        tool = LsTool()
        assert tool.name == "ls"
        assert "Lists the contents of a specified directory" in tool.description

    @patch("subprocess.run")
    def test_ls_success(self, mock_run):
        """Test successful ls command."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "file1\nfile2\nfile3"
        mock_run.return_value = mock_process
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert result == "file1\nfile2\nfile3"
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["ls", "-lA", "test_dir"]
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True

    @patch("subprocess.run")
    def test_ls_with_default_path(self, mock_run):
        """Test ls command with default path (no path specified)."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "file1\nfile2\nfile3"
        mock_run.return_value = mock_process
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute()
        
        # Verify results
        assert result == "file1\nfile2\nfile3"
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["ls", "-lA", "."]  # Should use current directory

    def test_ls_parent_traversal(self):
        """Test attempt to list directory with parent directory traversal."""
        tool = LsTool()
        result = tool.execute("../dangerous_dir")
        
        # Verify results
        assert "Error: Invalid path" in result
        assert "Cannot access parent directories" in result

    @patch("subprocess.run")
    def test_ls_directory_not_found(self, mock_run):
        """Test ls command when directory doesn't exist."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "ls: cannot access 'nonexistent_dir': No such file or directory"
        mock_run.return_value = mock_process
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("nonexistent_dir")
        
        # Verify results
        assert "Error: Directory not found" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_ls_command_generic_error(self, mock_run):
        """Test ls command with a generic error."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 2
        mock_process.stderr = "Some generic error"
        mock_run.return_value = mock_process
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Error executing ls command" in result
        assert "Some generic error" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_ls_command_no_stderr(self, mock_run):
        """Test ls command error with empty stderr."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Error executing ls command" in result
        assert "(No stderr)" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_ls_command_file_not_found(self, mock_run):
        """Test ls command when the 'ls' command itself isn't found."""
        # Setup mock
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'ls'")
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Error: 'ls' command not found" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_ls_command_timeout(self, mock_run):
        """Test ls command timeout."""
        # Setup mock
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ls", timeout=15)
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "Error: ls command timed out" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_ls_command_unexpected_error(self, mock_run):
        """Test unexpected error during ls command."""
        # Setup mock
        mock_run.side_effect = Exception("Unexpected error")
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "An unexpected error occurred while executing ls" in result
        assert "Unexpected error" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_ls_command_truncate_long_output(self, mock_run):
        """Test ls command with very long output that gets truncated."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        # Create an output with 101 lines (more than the 100 line limit)
        mock_process.stdout = "\n".join([f"line{i}" for i in range(101)])
        mock_run.return_value = mock_process
        
        # Execute the tool
        tool = LsTool()
        result = tool.execute("test_dir")
        
        # Verify results
        assert "... (output truncated)" in result
        # Result should have only 100 lines + truncation message
        assert len(result.splitlines()) == 101
        # The 100th line should be "line99"
        assert "line99" in result
        # The 101st line (which would be "line100") should NOT be in the result
        assert "line100" not in result
        mock_run.assert_called_once() 