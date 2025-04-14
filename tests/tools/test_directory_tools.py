"""
Tests for directory tools module.
"""
import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Direct import for coverage tracking
import src.cli_code.tools.directory_tools
from src.cli_code.tools.directory_tools import CreateDirectoryTool, LsTool


def test_create_directory_tool_init():
    """Test CreateDirectoryTool initialization."""
    tool = CreateDirectoryTool()
    assert tool.name == "create_directory"
    assert "Creates a new directory" in tool.description


@patch("os.path.exists")
@patch("os.path.isdir")
@patch("os.makedirs")
def test_create_directory_success(mock_makedirs, mock_isdir, mock_exists):
    """Test successful directory creation."""
    # Configure mocks
    mock_exists.return_value = False
    
    # Create tool and execute
    tool = CreateDirectoryTool()
    result = tool.execute("new_directory")
    
    # Verify
    assert "Successfully created directory" in result
    mock_makedirs.assert_called_once()


@patch("os.path.exists")
@patch("os.path.isdir")
def test_create_directory_already_exists(mock_isdir, mock_exists):
    """Test handling when directory already exists."""
    # Configure mocks
    mock_exists.return_value = True
    mock_isdir.return_value = True
    
    # Create tool and execute
    tool = CreateDirectoryTool()
    result = tool.execute("existing_directory")
    
    # Verify
    assert "Directory already exists" in result


@patch("os.path.exists")
@patch("os.path.isdir")
def test_create_directory_path_not_dir(mock_isdir, mock_exists):
    """Test handling when path exists but is not a directory."""
    # Configure mocks
    mock_exists.return_value = True
    mock_isdir.return_value = False
    
    # Create tool and execute
    tool = CreateDirectoryTool()
    result = tool.execute("not_a_directory")
    
    # Verify
    assert "Path exists but is not a directory" in result


def test_create_directory_parent_access():
    """Test blocking access to parent directories."""
    tool = CreateDirectoryTool()
    result = tool.execute("../outside_directory")
    
    # Verify
    assert "Invalid path" in result
    assert "Cannot access parent directories" in result


@patch("os.makedirs")
def test_create_directory_os_error(mock_makedirs):
    """Test handling of OSError during directory creation."""
    # Configure mock to raise OSError
    mock_makedirs.side_effect = OSError("Permission denied")
    
    # Create tool and execute
    tool = CreateDirectoryTool()
    result = tool.execute("protected_directory")
    
    # Verify
    assert "Error creating directory" in result
    assert "Permission denied" in result


@patch("os.makedirs")
def test_create_directory_unexpected_error(mock_makedirs):
    """Test handling of unexpected errors during directory creation."""
    # Configure mock to raise an unexpected error
    mock_makedirs.side_effect = ValueError("Unexpected error")
    
    # Create tool and execute
    tool = CreateDirectoryTool()
    result = tool.execute("problem_directory")
    
    # Verify
    assert "Error creating directory" in result


def test_ls_tool_init():
    """Test LsTool initialization."""
    tool = LsTool()
    assert tool.name == "ls"
    assert "Lists the contents of a specified directory" in tool.description
    assert isinstance(tool.args_schema, dict)
    assert "path" in tool.args_schema


@patch("subprocess.run")
def test_ls_success(mock_run):
    """Test successful directory listing."""
    # Configure mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "total 12\ndrwxr-xr-x 2 user group 4096 Jan 1 10:00 folder1\n-rw-r--r-- 1 user group 1234 Jan 1 10:00 file1.txt"
    mock_run.return_value = mock_process
    
    # Create tool and execute
    tool = LsTool()
    result = tool.execute("test_dir")
    
    # Verify
    assert "folder1" in result
    assert "file1.txt" in result
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == ["ls", "-lA", "test_dir"]


@patch("subprocess.run")
def test_ls_default_dir(mock_run):
    """Test ls with default directory."""
    # Configure mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "listing content"
    mock_run.return_value = mock_process
    
    # Create tool and execute with no path
    tool = LsTool()
    result = tool.execute()
    
    # Verify default directory used
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == ["ls", "-lA", "."]


def test_ls_invalid_path():
    """Test ls with path attempting to access parent directory."""
    tool = LsTool()
    result = tool.execute("../outside_dir")
    
    # Verify
    assert "Invalid path" in result
    assert "Cannot access parent directories" in result


@patch("subprocess.run")
def test_ls_directory_not_found(mock_run):
    """Test handling when directory is not found."""
    # Configure mock
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stderr = "ls: cannot access 'nonexistent_dir': No such file or directory"
    mock_run.return_value = mock_process
    
    # Create tool and execute
    tool = LsTool()
    result = tool.execute("nonexistent_dir")
    
    # Verify
    assert "Directory not found" in result


@patch("subprocess.run")
def test_ls_truncate_long_output(mock_run):
    """Test truncation of long directory listings."""
    # Create a long listing (more than 100 lines)
    long_listing = "\n".join([f"file{i}.txt" for i in range(150)])
    
    # Configure mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = long_listing
    mock_run.return_value = mock_process
    
    # Create tool and execute
    tool = LsTool()
    result = tool.execute("big_dir")
    
    # Verify truncation
    assert "output truncated" in result
    # Should only have 101 lines (100 files + truncation message)
    assert len(result.splitlines()) <= 101


@patch("subprocess.run")
def test_ls_generic_error(mock_run):
    """Test handling of generic errors."""
    # Configure mock
    mock_process = MagicMock()
    mock_process.returncode = 2
    mock_process.stderr = "ls: some generic error"
    mock_run.return_value = mock_process
    
    # Create tool and execute
    tool = LsTool()
    result = tool.execute("problem_dir")
    
    # Verify
    assert "Error executing ls command" in result
    assert "Code: 2" in result


@patch("subprocess.run")
def test_ls_command_not_found(mock_run):
    """Test handling when ls command is not found."""
    # Configure mock
    mock_run.side_effect = FileNotFoundError("No such file or directory: 'ls'")
    
    # Create tool and execute
    tool = LsTool()
    result = tool.execute()
    
    # Verify
    assert "'ls' command not found" in result


@patch("subprocess.run")
def test_ls_timeout(mock_run):
    """Test handling of ls command timeout."""
    # Configure mock
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ls", timeout=15)
    
    # Create tool and execute
    tool = LsTool()
    result = tool.execute()
    
    # Verify
    assert "ls command timed out" in result


@patch("subprocess.run")
def test_ls_unexpected_error(mock_run):
    """Test handling of unexpected errors during ls command."""
    # Configure mock
    mock_run.side_effect = Exception("Something unexpected happened")
    
    # Create tool and execute
    tool = LsTool()
    result = tool.execute()
    
    # Verify
    assert "An unexpected error occurred" in result
    assert "Something unexpected happened" in result 