import pytest
from unittest import mock
import logging

from src.cli_code.tools.task_complete_tool import TaskCompleteTool


@pytest.fixture
def task_complete_tool():
    """Provides an instance of TaskCompleteTool."""
    return TaskCompleteTool()

# Test cases for various summary inputs
@pytest.mark.parametrize(
    "input_summary, expected_output",
    [
        ("Task completed successfully.", "Task completed successfully."), # Normal case
        ("  \n\'Finished the process.\' \t", "Finished the process."), # Needs cleaning
        ("  Done.  ", "Done."), # Needs cleaning (less complex)
        (" \" \" ", ""), # Only quotes and spaces -> empty
        ("Okay", "Okay"), # Short but acceptable
    ],
)
def test_execute_normal_and_cleaning(task_complete_tool, input_summary, expected_output):
    """Test execute method with summaries needing cleaning and normal ones."""
    result = task_complete_tool.execute(summary=input_summary)
    assert result == expected_output

@pytest.mark.parametrize(
    "input_summary",
    [
        (""), # Empty string
        ("   "), # Only whitespace
        (" \n\t "), # Only whitespace chars
        ("ok"), # Too short
        ("    a   "), # Too short after stripping
        (" \" b \" "), # Too short after stripping
    ],
)
def test_execute_insufficient_summary(task_complete_tool, input_summary):
    """Test execute method with empty or very short summaries."""
    expected_output = "Task marked as complete, but the provided summary was insufficient."
    # Capture log messages
    with mock.patch("src.cli_code.tools.task_complete_tool.log") as mock_log:
        result = task_complete_tool.execute(summary=input_summary)
        assert result == expected_output
        mock_log.warning.assert_called_once_with(
            "TaskCompleteTool called with missing or very short summary."
        )

def test_execute_non_string_summary(task_complete_tool):
    """Test execute method with non-string input."""
    input_summary = 12345
    expected_output = str(input_summary)
    # Capture log messages
    with mock.patch("src.cli_code.tools.task_complete_tool.log") as mock_log:
        result = task_complete_tool.execute(summary=input_summary)
        assert result == expected_output
        mock_log.warning.assert_called_once_with(
            f"TaskCompleteTool received non-string summary type: {type(input_summary)}"
        )

def test_execute_stripping_loop(task_complete_tool):
    """Test that repeated stripping works correctly."""
    input_summary = " \" \'   Actual Summary   \' \" "
    expected_output = "Actual Summary"
    result = task_complete_tool.execute(summary=input_summary)
    assert result == expected_output 