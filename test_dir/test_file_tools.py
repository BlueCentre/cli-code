"""
Tests for file operation tools.
"""
import os
import re
import glob
import pytest
from unittest.mock import patch, mock_open, MagicMock, call

from cli_code.tools.file_tools import ViewTool, EditTool, GrepTool, GlobTool, MAX_CHARS_FOR_FULL_CONTENT


class TestViewTool:
    """Tests for the ViewTool."""

    def test_init(self):
        """Test initialization of ViewTool."""
        tool = ViewTool()
        assert tool.name == "view"
        assert "View specific sections" in tool.description

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\nline3\n")
    def test_view_file_with_offset_and_limit(self, mock_file, mock_getsize, mock_isfile, mock_exists):
        """Test viewing file with offset and limit parameters."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = 100  # Small file
        
        # Execute tool
        tool = ViewTool()
        result = tool.execute("test.txt", offset=2, limit=1)
        
        # Verify results
        assert "Content of test.txt (Lines 2-2)" in result
        assert "     2 line2" in result
        mock_exists.assert_called_once()
        mock_isfile.assert_called_once()
        mock_file.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\nline3\n")
    def test_view_entire_small_file(self, mock_file, mock_getsize, mock_isfile, mock_exists):
        """Test viewing an entire small file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = 100  # Small file
        
        # Execute tool
        tool = ViewTool()
        result = tool.execute("test.txt")
        
        # Verify results
        assert "Full Content of test.txt" in result
        assert "     1 line1" in result
        assert "     2 line2" in result
        assert "     3 line3" in result
        mock_getsize.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    def test_view_large_file_without_offset_limit(self, mock_getsize, mock_isfile, mock_exists):
        """Test viewing a large file without offset/limit."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = MAX_CHARS_FOR_FULL_CONTENT + 1  # Large file
        
        # Execute tool
        tool = ViewTool()
        result = tool.execute("large.txt")
        
        # Verify results
        assert "Error: File 'large.txt' is large" in result
        assert "summarize_code" in result
        mock_getsize.assert_called_once()

    def test_view_file_with_parent_directory_traversal(self):
        """Test viewing a file with parent directory traversal."""
        tool = ViewTool()
        result = tool.execute("../dangerous.txt")
        
        # Verify results
        assert "Error: Invalid file path" in result
        assert "Cannot access parent directories" in result

    @patch("os.path.exists")
    def test_view_nonexistent_file(self, mock_exists):
        """Test viewing a file that doesn't exist."""
        # Setup mock
        mock_exists.return_value = False
        
        # Execute tool
        tool = ViewTool()
        result = tool.execute("nonexistent.txt")
        
        # Verify results
        assert "Error: File not found" in result
        mock_exists.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_view_directory(self, mock_isfile, mock_exists):
        """Test attempting to view a directory."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = False
        
        # Execute tool
        tool = ViewTool()
        result = tool.execute("dir/")
        
        # Verify results
        assert "Error: Cannot view a directory" in result
        mock_exists.assert_called_once()
        mock_isfile.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_view_file_with_read_error(self, mock_file, mock_getsize, mock_isfile, mock_exists):
        """Test viewing a file with read error."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = 100  # Small file
        mock_file.side_effect = Exception("Read error")
        
        # Execute tool
        tool = ViewTool()
        result = tool.execute("test.txt")
        
        # Verify results
        assert "Error viewing file" in result
        assert "Read error" in result
        mock_exists.assert_called_once()
        mock_isfile.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_view_empty_file(self, mock_file, mock_getsize, mock_isfile, mock_exists):
        """Test viewing an empty file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = 0  # Empty file
        
        # Execute tool
        tool = ViewTool()
        result = tool.execute("empty.txt")
        
        # Verify results
        assert "Full Content of empty.txt" in result
        assert "(File is empty or slice resulted in no lines)" in result
        mock_exists.assert_called_once()
        mock_isfile.assert_called_once()


class TestEditTool:
    """Tests for the EditTool."""

    def test_init(self):
        """Test initialization of EditTool."""
        tool = EditTool()
        assert tool.name == "edit"
        assert "Edit or create a file" in tool.description

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_edit_create_new_file_with_content(self, mock_file, mock_makedirs, mock_exists):
        """Test creating a new file with content."""
        # Setup mocks
        mock_exists.return_value = True  # Directory exists
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("test.txt", content="New content")
        
        # Verify results
        assert "Successfully wrote content to test.txt" in result
        mock_file.assert_called_once_with(os.path.abspath("test.txt"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("New content")

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_directory_if_not_exists(self, mock_file, mock_makedirs, mock_exists):
        """Test creating a directory if it doesn't exist."""
        # Setup mocks
        mock_exists.side_effect = [False, True]  # Directory doesn't exist, then file exists
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("new_dir/test.txt", content="New content")
        
        # Verify results
        assert "Successfully wrote content to new_dir/test.txt" in result
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once()

    def test_edit_file_with_parent_directory_traversal(self):
        """Test editing a file with parent directory traversal."""
        tool = EditTool()
        result = tool.execute("../dangerous.txt", content="Unsafe content")
        
        # Verify results
        assert "Error: Invalid file path" in result

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_edit_create_empty_file(self, mock_file, mock_makedirs, mock_exists):
        """Test creating an empty file."""
        # Setup mocks
        mock_exists.return_value = True  # Directory exists
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("empty.txt")
        
        # Verify results
        assert "Successfully created/emptied file empty.txt" in result
        mock_file.assert_called_once_with(os.path.abspath("empty.txt"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("")

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="Original content here")
    def test_edit_replace_string(self, mock_file, mock_exists):
        """Test replacing a string in a file."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("test.txt", old_string="Original", new_string="Modified")
        
        # Verify results
        assert "Successfully replaced first occurrence in test.txt" in result
        # Check that we first read the file then write to it
        assert mock_file.call_args_list == [
            call(os.path.abspath("test.txt"), "r", encoding="utf-8"),
            call(os.path.abspath("test.txt"), "w", encoding="utf-8")
        ]
        # The new content should be the original with the replacement
        mock_file().write.assert_called_once_with("Modified content here")

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="Original content")
    def test_edit_delete_string(self, mock_file, mock_exists):
        """Test deleting a string in a file."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("test.txt", old_string="Original ", new_string="")
        
        # Verify results
        assert "Successfully deleted first occurrence in test.txt" in result
        mock_file().write.assert_called_once_with("content")

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="Different content")
    def test_edit_string_not_found(self, mock_file, mock_exists):
        """Test when the string to replace isn't found."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("test.txt", old_string="Not present", new_string="Replacement")
        
        # Verify results
        assert "Error: `old_string` not found in test.txt" in result

    @patch("os.path.exists")
    def test_edit_missing_file_for_replacement(self, mock_exists):
        """Test replacing string in non-existent file."""
        # Setup mock
        mock_exists.return_value = False
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("missing.txt", old_string="Original", new_string="Modified")
        
        # Verify results
        assert "Error: File not found for replacement" in result

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_edit_file_with_read_error(self, mock_file, mock_exists):
        """Test replacing string with file read error."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.side_effect = Exception("Read error")
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("test.txt", old_string="Original", new_string="Modified")
        
        # Verify results
        assert "Error reading file for replacement" in result

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open")
    def test_edit_file_with_generic_error(self, mock_file, mock_makedirs, mock_exists):
        """Test editing file with a generic error."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.side_effect = Exception("Generic error")
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("test.txt", content="New content")
        
        # Verify results
        assert "Error editing file" in result
        assert "Generic error" in result

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open")
    def test_edit_directory(self, mock_file, mock_makedirs, mock_exists):
        """Test trying to edit a directory."""
        # Setup mocks
        mock_exists.return_value = True
        mock_file.side_effect = IsADirectoryError("Is a directory")
        
        # Execute tool
        tool = EditTool()
        result = tool.execute("dir/", content="New content")
        
        # Verify results
        assert "Error: Cannot edit a directory" in result

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_edit_invalid_arguments(self, mock_file, mock_makedirs, mock_exists):
        """Test providing invalid combination of arguments."""
        # Execute tool
        tool = EditTool()
        result = tool.execute("test.txt", old_string="Original")  # Missing new_string
        
        # Verify results
        assert "Error: Invalid arguments" in result


class TestGrepTool:
    """Tests for the GrepTool."""

    def test_init(self):
        """Test initialization of GrepTool."""
        tool = GrepTool()
        assert tool.name == "grep"
        assert "Search for a pattern" in tool.description

    @patch("os.path.isdir")
    @patch("os.walk")
    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=mock_open, read_data="line with pattern\nno match\nanother pattern line\n")
    def test_grep_basic_search(self, mock_file, mock_isfile, mock_walk, mock_isdir):
        """Test basic grep search in all files."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        
        # Create absolute paths for the files
        mock_walk.return_value = [
            (os.path.abspath("."), [], ["file1.txt", "file2.txt"])
        ]
        
        # Execute tool
        tool = GrepTool()
        result = tool.execute("pattern")
        
        # Verify results (without ./ prefix since we're checking partial strings)
        assert "file1.txt:1: line with pattern" in result
        assert "file1.txt:3: another pattern line" in result
        mock_isdir.assert_called_once()
        mock_walk.assert_called_once()
        assert mock_file.call_count >= 1  # Called at least once for a file

    def test_grep_with_parent_directory_traversal(self):
        """Test grep with parent directory traversal."""
        tool = GrepTool()
        result = tool.execute("pattern", path="../dangerous")
        
        # Verify results
        assert "Error: Invalid path" in result

    @patch("os.path.isdir")
    def test_grep_invalid_directory(self, mock_isdir):
        """Test grep with invalid directory."""
        # Setup mock
        mock_isdir.return_value = False
        
        # Execute tool
        tool = GrepTool()
        result = tool.execute("pattern", path="nonexistent")
        
        # Verify results
        assert "Error: Path is not a directory" in result

    def test_grep_invalid_regex(self):
        """Test grep with invalid regex pattern."""
        tool = GrepTool()
        result = tool.execute("[invalid regex")
        
        # Verify results
        assert "Error: Invalid regex pattern" in result

    @patch("os.path.isdir")
    @patch("glob.glob")
    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=mock_open, read_data="line with pattern\nno match\n")
    def test_grep_with_include_pattern(self, mock_file, mock_isfile, mock_glob, mock_isdir):
        """Test grep with include pattern."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_glob.return_value = [os.path.abspath("./matched_file.py")]
        
        # Execute tool
        tool = GrepTool()
        result = tool.execute("pattern", path=".", include="*.py")
        
        # Verify results
        assert "matched_file.py:1: line with pattern" in result
        mock_isdir.assert_called_once()
        mock_glob.assert_called_once()

    @patch("os.path.isdir")
    @patch("glob.glob")
    def test_grep_glob_error(self, mock_glob, mock_isdir):
        """Test grep with glob error."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_glob.side_effect = Exception("Glob error")
        
        # Execute tool
        tool = GrepTool()
        result = tool.execute("pattern", path=".", include="*.py")
        
        # Verify results
        assert "Error finding files with include pattern" in result

    @patch("os.path.isdir")
    @patch("os.walk")
    @patch("os.path.isfile")
    @patch("builtins.open")
    def test_grep_file_open_error(self, mock_file, mock_isfile, mock_walk, mock_isdir):
        """Test grep with file open error."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_walk.return_value = [(".", [], ["file.txt"])]
        mock_isfile.return_value = True
        mock_file.side_effect = OSError("Cannot open file")
        
        # Execute tool
        tool = GrepTool()
        result = tool.execute("pattern")
        
        # Verify results
        assert "No matches found for pattern" in result  # Errors are logged but not returned

    @patch("os.path.isdir")
    @patch("os.walk")
    def test_grep_no_matches(self, mock_walk, mock_isdir):
        """Test grep with no matches."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_walk.return_value = []  # No files
        
        # Execute tool
        tool = GrepTool()
        result = tool.execute("pattern")
        
        # Verify results
        assert "No matches found for pattern" in result

    @patch("os.path.isdir")
    def test_grep_unexpected_error(self, mock_isdir):
        """Test grep with unexpected error."""
        # Setup mock
        mock_isdir.side_effect = Exception("Unexpected error")
        
        # Execute tool
        tool = GrepTool()
        result = tool.execute("pattern")
        
        # Verify results
        assert "Error searching files" in result
        assert "Unexpected error" in result


class TestGlobTool:
    """Tests for the GlobTool."""

    def test_init(self):
        """Test initialization of GlobTool."""
        tool = GlobTool()
        assert tool.name == "glob"
        assert "Find files/directories matching specific glob patterns" in tool.description

    @patch("os.path.isdir")
    @patch("glob.glob")
    def test_glob_basic_search(self, mock_glob, mock_isdir):
        """Test basic glob search."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_glob.return_value = [
            os.path.abspath("./file1.txt"),
            os.path.abspath("./subdir/file2.txt")
        ]
        
        # Execute tool
        tool = GlobTool()
        result = tool.execute("*.txt")
        
        # Verify results
        assert "./file1.txt" in result
        assert "subdir/file2.txt" in result
        mock_isdir.assert_called_once()
        mock_glob.assert_called_once()

    def test_glob_with_parent_directory_traversal(self):
        """Test glob with parent directory traversal."""
        tool = GlobTool()
        result = tool.execute("*.txt", path="../dangerous")
        
        # Verify results
        assert "Error: Invalid path" in result

    @patch("os.path.isdir")
    def test_glob_invalid_directory(self, mock_isdir):
        """Test glob with invalid directory."""
        # Setup mock
        mock_isdir.return_value = False
        
        # Execute tool
        tool = GlobTool()
        result = tool.execute("*.txt", path="nonexistent")
        
        # Verify results
        assert "Error: Path is not a directory" in result

    @patch("os.path.isdir")
    @patch("glob.glob")
    def test_glob_no_matches(self, mock_glob, mock_isdir):
        """Test glob with no matches."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_glob.return_value = []  # No matches
        
        # Execute tool
        tool = GlobTool()
        result = tool.execute("*.nonexistent")
        
        # Verify results
        assert "No files or directories found matching pattern" in result

    @patch("os.path.isdir")
    @patch("glob.glob")
    def test_glob_unexpected_error(self, mock_glob, mock_isdir):
        """Test glob with unexpected error."""
        # Setup mocks
        mock_isdir.return_value = True
        mock_glob.side_effect = Exception("Glob error")
        
        # Execute tool
        tool = GlobTool()
        result = tool.execute("*.txt")
        
        # Verify results
        assert "Error finding files" in result
        assert "Glob error" in result 