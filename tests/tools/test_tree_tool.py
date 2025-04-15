"""
Tests for tree_tool module.
"""

import os
import pathlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Direct import for coverage tracking
import src.cli_code.tools.tree_tool
from src.cli_code.tools.tree_tool import DEFAULT_TREE_DEPTH, MAX_TREE_DEPTH, TreeTool


def test_tree_tool_init():
    """Test TreeTool initialization."""
    tool = TreeTool()
    assert tool.name == "tree"
    assert "directory structure" in tool.description
    assert f"depth of {DEFAULT_TREE_DEPTH}" in tool.description
    assert "args_schema" in dir(tool)
    assert "path" in tool.args_schema
    assert "depth" in tool.args_schema


@patch("subprocess.run")
def test_tree_success(mock_run):
    """Test successful tree command execution."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = ".\n├── file1.txt\n└── dir1/\n    ├── file2.txt\n    └── file3.txt"
    mock_run.return_value = mock_process

    # Execute tool
    tool = TreeTool()
    result = tool.execute()

    # Verify results
    assert "file1.txt" in result
    assert "dir1/" in result
    assert "file2.txt" in result
    mock_run.assert_called_once_with(
        ["tree", "-L", str(DEFAULT_TREE_DEPTH)], capture_output=True, text=True, check=False, timeout=15
    )


@patch("subprocess.run")
def test_tree_with_custom_path(mock_run):
    """Test tree with custom path."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = ".\n└── test_dir/\n    └── file.txt"
    mock_run.return_value = mock_process

    # Execute tool with custom path
    tool = TreeTool()
    result = tool.execute(path="test_dir")

    # Verify correct command
    mock_run.assert_called_once()
    assert "test_dir" in mock_run.call_args[0][0]


@patch("subprocess.run")
def test_tree_with_custom_depth_int(mock_run):
    """Test tree with custom depth as integer."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Directory tree"
    mock_run.return_value = mock_process

    # Execute tool with custom depth
    tool = TreeTool()
    result = tool.execute(depth=2)

    # Verify depth parameter used
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][2] == "2"


@patch("subprocess.run")
def test_tree_with_custom_depth_string(mock_run):
    """Test tree with custom depth as string."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Directory tree"
    mock_run.return_value = mock_process

    # Execute tool with custom depth as string
    tool = TreeTool()
    result = tool.execute(depth="4")

    # Verify string was converted to int
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][2] == "4"


@patch("subprocess.run")
def test_tree_with_invalid_depth(mock_run):
    """Test tree with invalid depth value."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Directory tree"
    mock_run.return_value = mock_process

    # Execute tool with invalid depth
    tool = TreeTool()
    result = tool.execute(depth="invalid")

    # Verify default was used instead
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][2] == str(DEFAULT_TREE_DEPTH)


@patch("subprocess.run")
def test_tree_with_depth_exceeding_max(mock_run):
    """Test tree with depth exceeding maximum allowed."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Directory tree"
    mock_run.return_value = mock_process

    # Execute tool with too large depth
    tool = TreeTool()
    result = tool.execute(depth=MAX_TREE_DEPTH + 5)

    # Verify depth was clamped to maximum
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][2] == str(MAX_TREE_DEPTH)


@patch("subprocess.run")
def test_tree_long_output_truncation(mock_run):
    """Test truncation of long tree output."""
    # Create a long tree output (> 200 lines)
    long_output = ".\n" + "\n".join([f"├── file{i}.txt" for i in range(250)])

    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = long_output
    mock_run.return_value = mock_process

    # Execute tool
    tool = TreeTool()
    result = tool.execute()

    # Verify truncation
    assert "... (output truncated)" in result
    assert len(result.splitlines()) <= 202  # 200 lines + truncation message + header


@patch("subprocess.run")
def test_tree_command_not_found(mock_run):
    """Test when tree command is not found (returncode 127)."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 127
    mock_process.stderr = "tree: command not found"
    mock_run.return_value = mock_process

    # Setup fallback mock
    with patch.object(TreeTool, "_fallback_tree_implementation", return_value="Fallback tree output"):
        # Execute tool
        tool = TreeTool()
        result = tool.execute()

        # Verify fallback was used
        assert result == "Fallback tree output"


@patch("subprocess.run")
def test_tree_command_other_error(mock_run):
    """Test when tree command fails with an error other than 'not found'."""
    # Setup mock
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stderr = "tree: some other error"
    mock_run.return_value = mock_process

    # Setup fallback mock
    with patch.object(TreeTool, "_fallback_tree_implementation", return_value="Fallback tree output"):
        # Execute tool
        tool = TreeTool()
        result = tool.execute()

        # Verify fallback was used
        assert result == "Fallback tree output"


@patch("subprocess.run")
def test_tree_file_not_found_error(mock_run):
    """Test handling of FileNotFoundError."""
    # Setup mock to raise FileNotFoundError
    mock_run.side_effect = FileNotFoundError("No such file or directory: 'tree'")

    # Setup fallback mock
    with patch.object(TreeTool, "_fallback_tree_implementation", return_value="Fallback tree output"):
        # Execute tool
        tool = TreeTool()
        result = tool.execute()

        # Verify fallback was used
        assert result == "Fallback tree output"


@patch("subprocess.run")
def test_tree_timeout(mock_run):
    """Test handling of command timeout."""
    # Setup mock to raise TimeoutExpired
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="tree", timeout=15)

    # Execute tool
    tool = TreeTool()
    result = tool.execute()

    # Verify timeout message
    assert "Error: Tree command timed out" in result
    assert "The directory might be too large or complex" in result


@patch("subprocess.run")
def test_tree_unexpected_error(mock_run):
    """Test handling of unexpected error with successful fallback."""
    # Setup mock to raise an unexpected error
    mock_run.side_effect = Exception("Unexpected error")

    # Setup fallback mock
    with patch.object(TreeTool, "_fallback_tree_implementation", return_value="Fallback tree output"):
        # Execute tool
        tool = TreeTool()
        result = tool.execute()

        # Verify fallback was used
        assert result == "Fallback tree output"


@patch("subprocess.run")
def test_tree_unexpected_error_with_fallback_failure(mock_run):
    """Test handling of unexpected error with fallback also failing."""
    # Setup mock to raise an unexpected error
    mock_run.side_effect = Exception("Unexpected error")

    # Setup fallback mock to also fail
    with patch.object(TreeTool, "_fallback_tree_implementation", side_effect=Exception("Fallback error")):
        # Execute tool
        tool = TreeTool()
        result = tool.execute()

        # Verify error message
        assert "An unexpected error occurred while displaying directory structure" in result


@patch("subprocess.run")
def test_fallback_tree_implementation(mock_run):
    """Test the fallback tree implementation when tree command fails."""
    # Setup mock to simulate tree command failure
    mock_process = MagicMock()
    mock_process.returncode = 127  # Command not found
    mock_process.stderr = "tree: command not found"
    mock_run.return_value = mock_process

    # Mock the fallback implementation to provide a custom output
    with patch.object(TreeTool, "_fallback_tree_implementation") as mock_fallback:
        mock_fallback.return_value = "Mocked fallback tree output\nfile1.txt\ndir1/\n└── file2.txt"

        # Execute tool
        tool = TreeTool()
        result = tool.execute(path="test_path")

        # Verify the fallback was called with correct parameters
        mock_fallback.assert_called_once_with("test_path", DEFAULT_TREE_DEPTH)

        # Verify result came from fallback
        assert result == "Mocked fallback tree output\nfile1.txt\ndir1/\n└── file2.txt"


def test_fallback_tree_nonexistent_path():
    """Test fallback tree with non-existent path."""
    with patch("pathlib.Path.resolve", return_value=Path("nonexistent")):
        with patch("pathlib.Path.exists", return_value=False):
            # Execute fallback implementation
            tool = TreeTool()
            result = tool._fallback_tree_implementation("nonexistent", 3)

            # Verify error message
            assert "Error: Path 'nonexistent' does not exist" in result


def test_fallback_tree_not_a_directory():
    """Test fallback tree with path that is not a directory."""
    with patch("pathlib.Path.resolve", return_value=Path("file.txt")):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                # Execute fallback implementation
                tool = TreeTool()
                result = tool._fallback_tree_implementation("file.txt", 3)

                # Verify error message
                assert "Error: Path 'file.txt' is not a directory" in result


def test_fallback_tree_with_exception():
    """Test fallback tree handling of unexpected exceptions."""
    with patch("os.walk", side_effect=Exception("Test error")):
        # Execute fallback implementation
        tool = TreeTool()
        result = tool._fallback_tree_implementation(".", 3)

        # Verify error message
        assert "Error generating directory tree" in result
        assert "Test error" in result
