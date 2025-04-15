"""
Tests for edge cases of the TreeTool.

To run just this test file:
python -m pytest tests/tools/test_tree_tool_edge_cases.py

To run a specific test function:
python -m pytest tests/tools/test_tree_tool_edge_cases.py::TestTreeToolEdgeCases::test_tree_empty_result

To run all tests related to tree tool:
    python -m pytest -k "tree_tool"
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from src.cli_code.tools.tree_tool import DEFAULT_TREE_DEPTH, MAX_TREE_DEPTH, TreeTool


class TestTreeToolEdgeCases:
    """Tests for edge cases of the TreeTool class."""

    @patch("subprocess.run")
    def test_tree_complex_path_handling(self, mock_run):
        """Test tree command with a complex path containing spaces and special characters."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "path with spaces\n└── file.txt"
        mock_run.return_value = mock_process

        # Execute tool with path containing spaces
        tool = TreeTool()
        complex_path = "path with spaces"
        result = tool.execute(path=complex_path)

        # Verify results
        assert "path with spaces" in result
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", str(DEFAULT_TREE_DEPTH), complex_path]

    @patch("subprocess.run")
    def test_tree_empty_result(self, mock_run):
        """Test tree command with an empty result."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ""  # Empty output
        mock_run.return_value = mock_process

        # Execute tool
        tool = TreeTool()
        result = tool.execute()

        # Verify results
        assert result == ""  # Should return the empty string as is

    @patch("subprocess.run")
    def test_tree_special_characters_in_output(self, mock_run):
        """Test tree command with special characters in the output."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n├── file-with-dashes.txt\n├── file_with_underscores.txt\n├── 特殊字符.txt"
        mock_run.return_value = mock_process

        # Execute tool
        tool = TreeTool()
        result = tool.execute()

        # Verify results
        assert "file-with-dashes.txt" in result
        assert "file_with_underscores.txt" in result
        assert "特殊字符.txt" in result

    @patch("subprocess.run")
    def test_tree_with_negative_depth(self, mock_run):
        """Test tree command with a negative depth value."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n└── file.txt"
        mock_run.return_value = mock_process

        # Execute tool with negative depth
        tool = TreeTool()
        result = tool.execute(depth=-5)

        # Verify results
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        # Should be clamped to minimum depth of 1
        assert args[0] == ["tree", "-L", "1"]

    @patch("subprocess.run")
    def test_tree_with_float_depth(self, mock_run):
        """Test tree command with a float depth value."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n└── file.txt"
        mock_run.return_value = mock_process

        # Execute tool with float depth
        tool = TreeTool()
        result = tool.execute(depth=2.7)

        # Verify results
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        # FloatingPointError: The TreeTool doesn't convert floats to int, it passes them as strings
        assert args[0] == ["tree", "-L", "2.7"]

    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("os.walk")
    def test_fallback_nested_directories(self, mock_walk, mock_is_dir, mock_exists, mock_resolve):
        """Test fallback tree implementation with nested directories."""
        # Setup mocks
        mock_resolve.return_value = Path("test_dir")
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        # Setup mock directory structure:
        # test_dir/
        # ├── dir1/
        # │   ├── subdir1/
        # │   │   └── file3.txt
        # │   └── file2.txt
        # └── file1.txt
        mock_walk.return_value = [
            ("test_dir", ["dir1"], ["file1.txt"]),
            ("test_dir/dir1", ["subdir1"], ["file2.txt"]),
            ("test_dir/dir1/subdir1", [], ["file3.txt"]),
        ]

        # Execute fallback tree implementation
        tool = TreeTool()
        result = tool._fallback_tree_implementation("test_dir", 3)

        # Verify results
        assert "." in result
        assert "file1.txt" in result
        assert "dir1/" in result
        assert "file2.txt" in result
        assert "subdir1/" in result
        assert "file3.txt" in result

    @patch("subprocess.run")
    def test_tree_command_os_error(self, mock_run):
        """Test tree command raising an OSError."""
        # Setup mock to raise OSError
        mock_run.side_effect = OSError("Simulated OS error")

        # Mock the fallback implementation
        with patch.object(TreeTool, "_fallback_tree_implementation") as mock_fallback:
            mock_fallback.return_value = "Fallback tree output"

            # Execute tool
            tool = TreeTool()
            result = tool.execute()

            # Verify results
            assert result == "Fallback tree output"
            mock_fallback.assert_called_once_with(".", DEFAULT_TREE_DEPTH)

    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("os.walk")
    def test_fallback_empty_directory(self, mock_walk, mock_is_dir, mock_exists, mock_resolve):
        """Test fallback tree implementation with an empty directory."""
        # Setup mocks
        mock_resolve.return_value = Path("empty_dir")
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        # Empty directory
        mock_walk.return_value = [
            ("empty_dir", [], []),
        ]

        # Execute fallback tree implementation
        tool = TreeTool()
        result = tool._fallback_tree_implementation("empty_dir", 3)

        # Verify results
        assert "." in result
        assert len(result.splitlines()) == 1  # Only the root directory line

    @patch("subprocess.run")
    def test_tree_command_with_long_path(self, mock_run):
        """Test tree command with a very long path."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "very/long/path\n└── file.txt"
        mock_run.return_value = mock_process

        # Very long path
        long_path = "/".join(["directory"] * 20)  # Creates a very long path

        # Execute tool
        tool = TreeTool()
        result = tool.execute(path=long_path)

        # Verify results
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", str(DEFAULT_TREE_DEPTH), long_path]

    @patch("subprocess.run")
    def test_tree_command_path_does_not_exist(self, mock_run):
        """Test tree command with a path that doesn't exist."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "tree: nonexistent_path: No such file or directory"
        mock_run.return_value = mock_process

        # Mock the fallback implementation
        with patch.object(TreeTool, "_fallback_tree_implementation") as mock_fallback:
            mock_fallback.return_value = "Error: Path 'nonexistent_path' does not exist."

            # Execute tool
            tool = TreeTool()
            result = tool.execute(path="nonexistent_path")

            # Verify results
            assert "does not exist" in result
            mock_fallback.assert_called_once_with("nonexistent_path", DEFAULT_TREE_DEPTH)
