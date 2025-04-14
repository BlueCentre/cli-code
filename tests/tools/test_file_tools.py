"""
Tests for the file operation tools.
"""

import os
import pytest
import builtins
from unittest.mock import patch, MagicMock, mock_open

# Import tools from the correct path
from src.cli_code.tools.file_tools import ViewTool, EditTool, GrepTool, GlobTool

# --- Test Fixtures ---

@pytest.fixture
def view_tool():
    """Provides an instance of ViewTool."""
    return ViewTool()

@pytest.fixture
def edit_tool():
    """Provides an instance of EditTool."""
    return EditTool()

@pytest.fixture
def grep_tool():
    """Provides an instance of GrepTool."""
    return GrepTool()

@pytest.fixture
def glob_tool():
    """Provides an instance of GlobTool."""
    return GlobTool()

@pytest.fixture
def test_fs(tmp_path):
    """Creates a temporary file structure for view/edit testing."""
    small_file = tmp_path / "small.txt"
    small_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5", encoding="utf-8")

    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    large_file_content = "L" * (60 * 1024) # Assuming MAX_CHARS is around 50k
    large_file = tmp_path / "large.txt"
    large_file.write_text(large_file_content, encoding="utf-8")

    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    return tmp_path

@pytest.fixture
def grep_fs(tmp_path):
    """Creates a temporary file structure for grep testing."""
    # Root files
    (tmp_path / "file1.txt").write_text("Hello world\nSearch pattern here\nAnother line")
    (tmp_path / "file2.log").write_text("Log entry 1\nAnother search hit")
    (tmp_path / ".hiddenfile").write_text("Should be ignored")

    # Subdirectory
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file3.txt").write_text("Subdir file\nContains pattern match")
    (sub_dir / "file4.dat").write_text("Data file, no match")

    # Nested subdirectory
    nested_dir = sub_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "file5.txt").write_text("Deeply nested pattern hit")

    # __pycache__ directory
    pycache_dir = tmp_path / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "cache.pyc").write_text("ignore me pattern")

    return tmp_path

# --- ViewTool Tests ---

def test_view_small_file_entirely(view_tool, test_fs):
    file_path = str(test_fs / "small.txt")
    result = view_tool.execute(file_path=file_path)
    expected_prefix = f"--- Full Content of {file_path} ---"
    assert expected_prefix in result
    assert "1 Line 1" in result
    assert "5 Line 5" in result
    assert len(result.strip().split('\n')) == 6 # Prefix + 5 lines

def test_view_with_offset(view_tool, test_fs):
    file_path = str(test_fs / "small.txt")
    result = view_tool.execute(file_path=file_path, offset=3)
    expected_prefix = f"--- Content of {file_path} (Lines 3-5) ---"
    assert expected_prefix in result
    assert "1 Line 1" not in result
    assert "2 Line 2" not in result
    assert "3 Line 3" in result
    assert "5 Line 5" in result
    assert len(result.strip().split('\n')) == 4 # Prefix + 3 lines

def test_view_with_limit(view_tool, test_fs):
    file_path = str(test_fs / "small.txt")
    result = view_tool.execute(file_path=file_path, limit=2)
    expected_prefix = f"--- Content of {file_path} (Lines 1-2) ---"
    assert expected_prefix in result
    assert "1 Line 1" in result
    assert "2 Line 2" in result
    assert "3 Line 3" not in result
    assert len(result.strip().split('\n')) == 3 # Prefix + 2 lines

def test_view_with_offset_and_limit(view_tool, test_fs):
    file_path = str(test_fs / "small.txt")
    result = view_tool.execute(file_path=file_path, offset=2, limit=2)
    expected_prefix = f"--- Content of {file_path} (Lines 2-3) ---"
    assert expected_prefix in result
    assert "1 Line 1" not in result
    assert "2 Line 2" in result
    assert "3 Line 3" in result
    assert "4 Line 4" not in result
    assert len(result.strip().split('\n')) == 3 # Prefix + 2 lines

def test_view_empty_file(view_tool, test_fs):
    file_path = str(test_fs / "empty.txt")
    result = view_tool.execute(file_path=file_path)
    expected_prefix = f"--- Full Content of {file_path} ---"
    assert expected_prefix in result
    assert "(File is empty or slice resulted in no lines)" in result

def test_view_non_existent_file(view_tool, test_fs):
    file_path = str(test_fs / "nonexistent.txt")
    result = view_tool.execute(file_path=file_path)
    assert f"Error: File not found: {file_path}" in result

def test_view_directory(view_tool, test_fs):
    dir_path = str(test_fs / "test_dir")
    result = view_tool.execute(file_path=dir_path)
    assert f"Error: Cannot view a directory: {dir_path}" in result

def test_view_invalid_path_parent_access(view_tool, test_fs):
    # Note: tmp_path makes it hard to truly test ../ escaping sandbox
    # We check if the tool's internal logic catches it anyway.
    file_path = "../some_file.txt"
    result = view_tool.execute(file_path=file_path)
    assert f"Error: Invalid file path '{file_path}'. Cannot access parent directories." in result

# Patch MAX_CHARS_FOR_FULL_CONTENT for this specific test
@patch('src.cli_code.tools.file_tools.MAX_CHARS_FOR_FULL_CONTENT', 1024)
def test_view_large_file_without_offset_limit(view_tool, test_fs):
    file_path = str(test_fs / "large.txt")
    result = view_tool.execute(file_path=file_path)
    assert f"Error: File '{file_path}' is large. Use the 'summarize_code' tool" in result

def test_view_offset_beyond_file_length(view_tool, test_fs):
    file_path = str(test_fs / "small.txt")
    result = view_tool.execute(file_path=file_path, offset=10)
    expected_prefix = f"--- Content of {file_path} (Lines 10-9) ---" # End index reflects slice start + len
    assert expected_prefix in result
    assert "(File is empty or slice resulted in no lines)" in result

def test_view_limit_zero(view_tool, test_fs):
    file_path = str(test_fs / "small.txt")
    result = view_tool.execute(file_path=file_path, limit=0)
    expected_prefix = f"--- Content of {file_path} (Lines 1-0) ---" # End index calculation
    assert expected_prefix in result
    assert "(File is empty or slice resulted in no lines)" in result

@patch('builtins.open', new_callable=mock_open)
def test_view_general_exception(mock_open_func, view_tool, test_fs):
    mock_open_func.side_effect = Exception("Unexpected error")
    file_path = str(test_fs / "small.txt") # Need a path for the tool to attempt
    result = view_tool.execute(file_path=file_path)
    assert "Error viewing file: Unexpected error" in result

# --- EditTool Tests ---

def test_edit_create_new_file_with_content(edit_tool, test_fs):
    file_path = test_fs / "new_file.txt"
    content = "Hello World!"
    result = edit_tool.execute(file_path=str(file_path), content=content)
    assert "Successfully wrote content" in result
    assert file_path.read_text() == content

def test_edit_overwrite_existing_file(edit_tool, test_fs):
    file_path_obj = test_fs / "small.txt"
    original_content = file_path_obj.read_text()
    new_content = "Overwritten!"
    result = edit_tool.execute(file_path=str(file_path_obj), content=new_content)
    assert "Successfully wrote content" in result
    assert file_path_obj.read_text() == new_content
    assert file_path_obj.read_text() != original_content

def test_edit_replace_string(edit_tool, test_fs):
    file_path = test_fs / "small.txt" # Content: "Line 1\nLine 2..."
    result = edit_tool.execute(file_path=str(file_path), old_string="Line 2", new_string="Replaced Line")
    assert "Successfully replaced first occurrence" in result
    content = file_path.read_text()
    assert "Line 1" in content
    assert "Line 2" not in content
    assert "Replaced Line" in content
    assert "Line 3" in content

def test_edit_delete_string(edit_tool, test_fs):
    file_path = test_fs / "small.txt"
    result = edit_tool.execute(file_path=str(file_path), old_string="Line 3\n", new_string="") # Include newline for exact match
    assert "Successfully deleted first occurrence" in result
    content = file_path.read_text()
    assert "Line 2" in content
    assert "Line 3" not in content
    assert "Line 4" in content # Should follow Line 2

def test_edit_replace_string_not_found(edit_tool, test_fs):
    file_path_obj = test_fs / "small.txt"
    original_content = file_path_obj.read_text()
    result = edit_tool.execute(file_path=str(file_path_obj), old_string="NonExistent", new_string="Replaced")
    assert "Error: `old_string` not found" in result
    assert file_path_obj.read_text() == original_content # File unchanged

def test_edit_replace_in_non_existent_file(edit_tool, test_fs):
    file_path = str(test_fs / "nonexistent.txt")
    result = edit_tool.execute(file_path=file_path, old_string="a", new_string="b")
    assert "Error: File not found for replacement" in result

def test_edit_create_empty_file(edit_tool, test_fs):
    file_path = test_fs / "new_empty.txt"
    result = edit_tool.execute(file_path=str(file_path))
    assert "Successfully created/emptied file" in result
    assert file_path.exists()
    assert file_path.read_text() == ""

def test_edit_create_file_with_dirs(edit_tool, test_fs):
    file_path = test_fs / "new_dir" / "nested_file.txt"
    content = "Nested content."
    result = edit_tool.execute(file_path=str(file_path), content=content)
    assert "Successfully wrote content" in result
    assert file_path.exists()
    assert file_path.read_text() == content
    assert file_path.parent.is_dir()

def test_edit_content_priority_warning(edit_tool, test_fs):
    file_path = test_fs / "priority_test.txt"
    content = "Content wins."
    # Patch logging to check for warning
    with patch('src.cli_code.tools.file_tools.log') as mock_log:
        result = edit_tool.execute(file_path=str(file_path), content=content, old_string="a", new_string="b")
        assert "Successfully wrote content" in result
        assert file_path.read_text() == content
        mock_log.warning.assert_called_once_with("Prioritizing 'content' over 'old/new_string'.")

def test_edit_invalid_path_parent_access(edit_tool):
    file_path = "../some_other_file.txt"
    result = edit_tool.execute(file_path=file_path, content="test")
    assert f"Error: Invalid file path '{file_path}'." in result

def test_edit_directory(edit_tool, test_fs):
    dir_path = str(test_fs / "test_dir")
    # Test writing content to a directory
    result_content = edit_tool.execute(file_path=dir_path, content="test")
    assert f"Error: Cannot edit a directory: {dir_path}" in result_content
    # Test replacing in a directory
    result_replace = edit_tool.execute(file_path=dir_path, old_string="a", new_string="b")
    assert f"Error reading file for replacement: [Errno 21] Is a directory: '{dir_path}'" in result_replace

def test_edit_invalid_arguments(edit_tool):
    file_path = "test.txt"
    result = edit_tool.execute(file_path=file_path, old_string="a") # Missing new_string
    assert "Error: Invalid arguments" in result
    result = edit_tool.execute(file_path=file_path, new_string="b") # Missing old_string
    assert "Error: Invalid arguments" in result

@patch('builtins.open', new_callable=mock_open)
def test_edit_general_exception(mock_open_func, edit_tool):
    mock_open_func.side_effect = IOError("Disk full")
    file_path = "some_file.txt"
    result = edit_tool.execute(file_path=file_path, content="test")
    assert "Error editing file: Disk full" in result

@patch('builtins.open', new_callable=mock_open)
def test_edit_read_exception_during_replace(mock_open_func, edit_tool):
    # Mock setup: successful exists check, then fail on read
    m = mock_open_func.return_value
    m.read.side_effect = IOError("Read error")

    with patch('os.path.exists', return_value=True):
        result = edit_tool.execute(file_path="existing.txt", old_string="a", new_string="b")
        assert "Error reading file for replacement: Read error" in result

# --- GrepTool Tests ---

def test_grep_basic(grep_tool, grep_fs):
    # Run from root of grep_fs
    os.chdir(grep_fs)
    result = grep_tool.execute(pattern="pattern")
    # Should find in file1.txt, file3.txt, file5.txt
    # Should NOT find in file2.log, file4.dat, .hiddenfile, __pycache__
    assert "./file1.txt:2: Search pattern here" in result
    assert "subdir/file3.txt:2: Contains pattern match" in result
    assert "subdir/nested/file5.txt:1: Deeply nested pattern hit" in result
    assert "file2.log" not in result
    assert "file4.dat" not in result
    assert ".hiddenfile" not in result
    assert "__pycache__" not in result
    assert len(result.strip().split('\n')) == 3

def test_grep_in_subdir(grep_tool, grep_fs):
    # Run from root, but specify subdir path
    os.chdir(grep_fs)
    result = grep_tool.execute(pattern="pattern", path="subdir")
    assert "./file3.txt:2: Contains pattern match" in result
    assert "nested/file5.txt:1: Deeply nested pattern hit" in result
    assert "file1.txt" not in result
    assert "file4.dat" not in result
    assert len(result.strip().split('\n')) == 2

def test_grep_include_txt(grep_tool, grep_fs):
    os.chdir(grep_fs)
    # Include only .txt files in the root dir
    result = grep_tool.execute(pattern="pattern", include="*.txt")
    assert "./file1.txt:2: Search pattern here" in result
    assert "subdir" not in result # Non-recursive by default
    assert "file2.log" not in result
    assert len(result.strip().split('\n')) == 1

def test_grep_include_recursive(grep_tool, grep_fs):
    os.chdir(grep_fs)
    # Include all .txt files recursively
    result = grep_tool.execute(pattern="pattern", include="**/*.txt")
    assert "./file1.txt:2: Search pattern here" in result
    assert "subdir/file3.txt:2: Contains pattern match" in result
    assert "subdir/nested/file5.txt:1: Deeply nested pattern hit" in result
    assert "file2.log" not in result
    assert len(result.strip().split('\n')) == 3

def test_grep_no_matches(grep_tool, grep_fs):
    os.chdir(grep_fs)
    pattern = "NonExistentPattern"
    result = grep_tool.execute(pattern=pattern)
    assert f"No matches found for pattern: {pattern}" in result

def test_grep_include_no_matches(grep_tool, grep_fs):
    os.chdir(grep_fs)
    result = grep_tool.execute(pattern="pattern", include="*.nonexistent")
    # The execute method returns based on regex matches, not file finding.
    # If no files are found by glob, the loop won't run, results empty.
    assert f"No matches found for pattern: pattern" in result

def test_grep_invalid_regex(grep_tool, grep_fs):
    os.chdir(grep_fs)
    invalid_pattern = "["
    result = grep_tool.execute(pattern=invalid_pattern)
    assert f"Error: Invalid regex pattern: {invalid_pattern}" in result

def test_grep_invalid_path_parent(grep_tool):
    result = grep_tool.execute(pattern="test", path="../somewhere")
    assert "Error: Invalid path '../somewhere'." in result

def test_grep_path_is_file(grep_tool, grep_fs):
    os.chdir(grep_fs)
    file_path = "file1.txt"
    result = grep_tool.execute(pattern="test", path=file_path)
    assert f"Error: Path is not a directory: {file_path}" in result

@patch('builtins.open', new_callable=mock_open)
def test_grep_read_oserror(mock_open_method, grep_tool, grep_fs):
    os.chdir(grep_fs)
    # Make open raise OSError for a specific file
    original_open = builtins.open
    def patched_open(*args, **kwargs):
        # Need to handle the file path correctly within the test
        abs_file1_path = str(grep_fs / 'file1.txt')
        abs_file2_path = str(grep_fs / 'file2.log')
        if args[0] == abs_file1_path:
            raise OSError("Permission denied")
        # Allow reading file2.log
        elif args[0] == abs_file2_path:
            # If mocking open completely, need to provide mock file object
            return mock_open(read_data="Log entry 1\nAnother search hit")(*args, **kwargs)
        else:
            # Fallback for other potential opens, or raise error
            raise FileNotFoundError(f"Unexpected open call in test: {args[0]}")
    mock_open_method.side_effect = patched_open

    # Patch glob to ensure file1.txt is considered
    with patch('glob.glob', return_value=[str(grep_fs / 'file1.txt'), str(grep_fs / 'file2.log')]):
        result = grep_tool.execute(pattern="search", include="*.*")
        # Should only find the match in file2.log, skipping file1.txt due to OSError
        assert "file1.txt" not in result
        assert "./file2.log:2: Another search hit" in result
        assert len(result.strip().split('\n')) == 1

@patch('glob.glob')
def test_grep_glob_exception(mock_glob, grep_tool, grep_fs):
    os.chdir(grep_fs)
    mock_glob.side_effect = Exception("Glob error")
    result = grep_tool.execute(pattern="test", include="*.txt")
    assert "Error finding files with include pattern: Glob error" in result

@patch('os.walk')
def test_grep_general_exception(mock_walk, grep_tool):
    # Need to change directory for os.walk patching to be effective if tool uses relative paths
    # However, the tool converts path to absolute, so patching os.walk directly should work
    mock_walk.side_effect = Exception("Walk error")
    result = grep_tool.execute(pattern="test", path=".") # Execute in current dir
    assert "Error searching files: Walk error" in result

# --- GlobTool Tests ---

def test_glob_basic(glob_tool, grep_fs): # Reusing grep_fs structure
    os.chdir(grep_fs)
    result = glob_tool.execute(pattern="*.txt")
    results_list = sorted(result.strip().split('\n'))
    assert "./file1.txt" in results_list
    assert "./subdir/file3.txt" not in results_list # Not recursive
    assert len(results_list) == 1

def test_glob_in_subdir(glob_tool, grep_fs):
    os.chdir(grep_fs)
    result = glob_tool.execute(pattern="*.txt", path="subdir")
    results_list = sorted(result.strip().split('\n'))
    assert "./file3.txt" in results_list
    assert "./nested/file5.txt" not in results_list # Not recursive within subdir
    assert len(results_list) == 1

def test_glob_recursive(glob_tool, grep_fs):
    os.chdir(grep_fs)
    result = glob_tool.execute(pattern="**/*.txt")
    results_list = sorted(result.strip().split('\n'))
    assert "./file1.txt" in results_list
    assert "subdir/file3.txt" in results_list
    assert "subdir/nested/file5.txt" in results_list
    assert len(results_list) == 3

def test_glob_no_matches(glob_tool, grep_fs):
    os.chdir(grep_fs)
    result = glob_tool.execute(pattern="*.nonexistent")
    assert "No files or directories found matching pattern: *.nonexistent" in result

def test_glob_invalid_path_parent(glob_tool):
    result = glob_tool.execute(pattern="*.txt", path="../somewhere")
    assert "Error: Invalid path '../somewhere'." in result

def test_glob_path_is_file(glob_tool, grep_fs):
    os.chdir(grep_fs)
    file_path = "file1.txt"
    result = glob_tool.execute(pattern="*.txt", path=file_path)
    assert f"Error: Path is not a directory: {file_path}" in result

@patch('glob.glob')
def test_glob_general_exception(mock_glob, glob_tool):
    mock_glob.side_effect = Exception("Globbing failed")
    result = glob_tool.execute(pattern="*.txt")
    assert "Error finding files: Globbing failed" in result 