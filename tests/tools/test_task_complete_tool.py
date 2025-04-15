"""
Tests for the TaskCompleteTool.
"""

from unittest.mock import patch

import pytest

from cli_code.tools.task_complete_tool import TaskCompleteTool


def test_task_complete_tool_init():
    """Test TaskCompleteTool initialization."""
    tool = TaskCompleteTool()
    assert tool.name == "task_complete"
    assert "Signals task completion" in tool.description


def test_execute_with_valid_summary():
    """Test execution with a valid summary."""
    tool = TaskCompleteTool()
    summary = "This is a valid summary of task completion."
    result = tool.execute(summary)

    assert result == summary


def test_execute_with_short_summary():
    """Test execution with a summary that's too short."""
    tool = TaskCompleteTool()
    summary = "Shrt"  # Less than 5 characters
    result = tool.execute(summary)

    assert "insufficient" in result
    assert result != summary


def test_execute_with_empty_summary():
    """Test execution with an empty summary."""
    tool = TaskCompleteTool()
    summary = ""
    result = tool.execute(summary)

    assert "insufficient" in result
    assert result != summary


def test_execute_with_none_summary():
    """Test execution with None as summary."""
    tool = TaskCompleteTool()
    summary = None

    with patch("cli_code.tools.task_complete_tool.log") as mock_log:
        result = tool.execute(summary)

        # Verify logging behavior - should be called at least once
        assert mock_log.warning.call_count >= 1
        # Check that one of the warnings is about non-string type
        assert any("non-string summary type" in str(args[0]) for args, _ in mock_log.warning.call_args_list)
        # Check that one of the warnings is about short summary
        assert any("missing or very short" in str(args[0]) for args, _ in mock_log.warning.call_args_list)

        assert "Task marked as complete" in result


def test_execute_with_non_string_summary():
    """Test execution with a non-string summary."""
    tool = TaskCompleteTool()
    summary = 12345  # Integer, not a string

    with patch("cli_code.tools.task_complete_tool.log") as mock_log:
        result = tool.execute(summary)

        # Verify logging behavior
        assert mock_log.warning.call_count >= 1
        assert any("non-string summary type" in str(args[0]) for args, _ in mock_log.warning.call_args_list)

        # The integer should be converted to a string
        assert result == "12345"


def test_execute_with_quoted_summary():
    """Test execution with a summary that has quotes and spaces to be cleaned."""
    tool = TaskCompleteTool()
    summary = '  "This summary has quotes and spaces"  '
    result = tool.execute(summary)

    # The quotes and spaces should be removed
    assert result == "This summary has quotes and spaces"


def test_execute_with_complex_cleaning():
    """Test execution with a summary that requires complex cleaning."""
    tool = TaskCompleteTool()
    summary = "\n\t \"'  Nested quotes and whitespace  '\" \t\n"
    result = tool.execute(summary)

    # All the nested quotes and whitespace should be removed
    assert result == "Nested quotes and whitespace"
