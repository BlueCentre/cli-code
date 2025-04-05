"""
Tests for tree tool.
"""
import os
import subprocess
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, mock_open

from cli_code.tools.tree_tool import TreeTool, DEFAULT_TREE_DEPTH, MAX_TREE_DEPTH


class TestTreeTool:
    """Tests for the TreeTool class."""

    def test_init(self):
        """Test initialization of TreeTool."""
        tool = TreeTool()
        assert tool.name == "tree"
        assert "Displays the directory structure as a tree" in tool.description
        assert "depth" in tool.args_schema
        assert "path" in tool.args_schema
        assert len(tool.required_args) == 0  # All args are optional

    @patch("subprocess.run")
    def test_tree_command_success(self, mock_run):
        """Test successful execution of tree command."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n├── file1.txt\n└── dir1\n    └── file2.txt"
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute()
        
        # Verify results
        assert "file1.txt" in result
        assert "dir1" in result
        assert "file2.txt" in result
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", str(DEFAULT_TREE_DEPTH)]
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True

    @patch("subprocess.run")
    def test_tree_with_custom_path(self, mock_run):
        """Test tree command with custom path."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "test_dir\n├── file1.txt\n└── file2.txt"
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute(path="test_dir")
        
        # Verify results
        assert "test_dir" in result
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", str(DEFAULT_TREE_DEPTH), "test_dir"]

    @patch("subprocess.run")
    def test_tree_with_custom_depth(self, mock_run):
        """Test tree command with custom depth."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n├── file1.txt\n└── dir1"
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute(depth=2)
        
        # Verify results
        assert "file1.txt" in result
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", "2"]  # Depth should be converted to string

    @patch("subprocess.run")
    def test_tree_with_string_depth(self, mock_run):
        """Test tree command with depth as string."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n├── file1.txt\n└── dir1"
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute(depth="2")  # String instead of int
        
        # Verify results
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", "2"]  # Should be converted properly

    @patch("subprocess.run")
    def test_tree_with_invalid_depth_string(self, mock_run):
        """Test tree command with invalid depth string."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n├── file1.txt\n└── dir1"
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute(depth="invalid")  # Invalid depth string
        
        # Verify results
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", str(DEFAULT_TREE_DEPTH)]  # Should use default

    @patch("subprocess.run")
    def test_tree_with_too_large_depth(self, mock_run):
        """Test tree command with depth larger than maximum."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n├── file1.txt\n└── dir1"
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute(depth=MAX_TREE_DEPTH + 5)  # Too large
        
        # Verify results
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", str(MAX_TREE_DEPTH)]  # Should be clamped to max

    @patch("subprocess.run")
    def test_tree_with_too_small_depth(self, mock_run):
        """Test tree command with depth smaller than minimum."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ".\n├── file1.txt\n└── dir1"
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute(depth=0)  # Too small
        
        # Verify results
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["tree", "-L", "1"]  # Should be clamped to min (1)

    @patch("subprocess.run")
    def test_tree_truncate_long_output(self, mock_run):
        """Test tree command with very long output that gets truncated."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        # Create an output with 201 lines (more than the 200 line limit)
        mock_process.stdout = "\n".join([f"line{i}" for i in range(201)])
        mock_run.return_value = mock_process
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute()
        
        # Verify results
        assert "... (output truncated)" in result
        # Result should have only 200 lines + truncation message
        assert len(result.splitlines()) == 201
        # The 200th line should be "line199"
        assert "line199" in result
        # The 201st line (which would be "line200") should NOT be in the result
        assert "line200" not in result

    @patch("subprocess.run")
    def test_tree_command_not_found_fallback(self, mock_run):
        """Test fallback when tree command is not found."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 127  # Command not found
        mock_process.stderr = "tree: command not found"
        mock_run.return_value = mock_process
        
        # Mock the fallback implementation
        with patch.object(TreeTool, '_fallback_tree_implementation') as mock_fallback:
            mock_fallback.return_value = "Fallback tree output"
            
            # Execute tool
            tool = TreeTool()
            result = tool.execute()
            
            # Verify results
            assert result == "Fallback tree output"
            mock_fallback.assert_called_once_with(".", DEFAULT_TREE_DEPTH)

    @patch("subprocess.run")
    def test_tree_command_error_fallback(self, mock_run):
        """Test fallback when tree command returns an error."""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 1  # Error
        mock_process.stderr = "Some error"
        mock_run.return_value = mock_process
        
        # Mock the fallback implementation
        with patch.object(TreeTool, '_fallback_tree_implementation') as mock_fallback:
            mock_fallback.return_value = "Fallback tree output"
            
            # Execute tool
            tool = TreeTool()
            result = tool.execute()
            
            # Verify results
            assert result == "Fallback tree output"
            mock_fallback.assert_called_once_with(".", DEFAULT_TREE_DEPTH)

    @patch("subprocess.run")
    def test_tree_command_file_not_found(self, mock_run):
        """Test when the 'tree' command itself isn't found."""
        # Setup mock
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'tree'")
        
        # Mock the fallback implementation
        with patch.object(TreeTool, '_fallback_tree_implementation') as mock_fallback:
            mock_fallback.return_value = "Fallback tree output"
            
            # Execute tool
            tool = TreeTool()
            result = tool.execute()
            
            # Verify results
            assert result == "Fallback tree output"
            mock_fallback.assert_called_once_with(".", DEFAULT_TREE_DEPTH)

    @patch("subprocess.run")
    def test_tree_command_timeout(self, mock_run):
        """Test tree command timeout."""
        # Setup mock
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tree", timeout=15)
        
        # Execute tool
        tool = TreeTool()
        result = tool.execute()
        
        # Verify results
        assert "Error: Tree command timed out" in result
        assert "too large or complex" in result

    @patch("subprocess.run")
    def test_tree_command_unexpected_error_with_fallback_success(self, mock_run):
        """Test unexpected error with successful fallback."""
        # Setup mock
        mock_run.side_effect = Exception("Unexpected error")
        
        # Mock the fallback implementation
        with patch.object(TreeTool, '_fallback_tree_implementation') as mock_fallback:
            mock_fallback.return_value = "Fallback tree output"
            
            # Execute tool
            tool = TreeTool()
            result = tool.execute()
            
            # Verify results
            assert result == "Fallback tree output"
            mock_fallback.assert_called_once_with(".", DEFAULT_TREE_DEPTH)

    @patch("subprocess.run")
    def test_tree_command_unexpected_error_with_fallback_failure(self, mock_run):
        """Test unexpected error with failed fallback."""
        # Setup mock
        mock_run.side_effect = Exception("Unexpected error")
        
        # Mock the fallback implementation
        with patch.object(TreeTool, '_fallback_tree_implementation') as mock_fallback:
            mock_fallback.side_effect = Exception("Fallback error")
            
            # Execute tool
            tool = TreeTool()
            result = tool.execute()
            
            # Verify results
            assert "An unexpected error occurred" in result
            assert "Unexpected error" in result
            mock_fallback.assert_called_once_with(".", DEFAULT_TREE_DEPTH)

    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("os.walk")
    def test_fallback_tree_implementation(self, mock_walk, mock_is_dir, mock_exists, mock_resolve):
        """Test the fallback tree implementation."""
        # Setup mocks
        mock_resolve.return_value = Path("test_dir")
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_walk.return_value = [
            ("test_dir", ["dir1", "dir2"], ["file1.txt"]),
            ("test_dir/dir1", [], ["file2.txt"]),
            ("test_dir/dir2", [], ["file3.txt"])
        ]
        
        # Execute fallback
        tool = TreeTool()
        result = tool._fallback_tree_implementation("test_dir")
        
        # Verify results
        assert "." in result  # Root directory
        assert "dir1" in result  # Subdirectories
        assert "dir2" in result
        assert "file1.txt" in result  # Files
        assert "file2.txt" in result
        assert "file3.txt" in result

    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists")
    def test_fallback_tree_nonexistent_path(self, mock_exists, mock_resolve):
        """Test fallback tree with nonexistent path."""
        # Setup mocks
        mock_resolve.return_value = Path("nonexistent")
        mock_exists.return_value = False
        
        # Execute fallback
        tool = TreeTool()
        result = tool._fallback_tree_implementation("nonexistent")
        
        # Verify results
        assert "Error: Path 'nonexistent' does not exist" in result

    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    def test_fallback_tree_not_a_directory(self, mock_is_dir, mock_exists, mock_resolve):
        """Test fallback tree with a file path."""
        # Setup mocks
        mock_resolve.return_value = Path("file.txt")
        mock_exists.return_value = True
        mock_is_dir.return_value = False
        
        # Execute fallback
        tool = TreeTool()
        result = tool._fallback_tree_implementation("file.txt")
        
        # Verify results
        assert "Error: Path 'file.txt' is not a directory" in result

    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("os.walk")
    def test_fallback_tree_truncate_long_output(self, mock_walk, mock_is_dir, mock_exists, mock_resolve):
        """Test fallback tree with very long output that gets truncated."""
        # Setup mocks
        mock_resolve.return_value = Path("test_dir")
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        
        # Create a directory structure with more than 200 files
        dirs = [("test_dir", [], [f"file{i}.txt" for i in range(201)])]
        mock_walk.return_value = dirs
        
        # Execute fallback
        tool = TreeTool()
        result = tool._fallback_tree_implementation("test_dir")
        
        # Verify results
        assert "... (output truncated)" in result
        assert len(result.splitlines()) <= 201  # 200 lines + truncation message

    @patch("pathlib.Path.resolve")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("os.walk")
    def test_fallback_tree_error(self, mock_walk, mock_is_dir, mock_exists, mock_resolve):
        """Test error in fallback tree implementation."""
        # Setup mocks
        mock_resolve.return_value = Path("test_dir")
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_walk.side_effect = Exception("Unexpected error")
        
        # Execute fallback
        tool = TreeTool()
        result = tool._fallback_tree_implementation("test_dir")
        
        # Verify results
        assert "Error generating directory tree" in result
        assert "Unexpected error" in result 