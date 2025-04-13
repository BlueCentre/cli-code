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
        (" \" \" ", "Task marked as complete, but the provided summary was insufficient."), # Only quotes and spaces -> empty, too short
        ("Okay", "Task marked as complete, but the provided summary was insufficient."), # Too short after checking length
        ("This is a much longer and more detailed summary.", "This is a much longer and more detailed summary."), # Long enough
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

def test_execute_loop_break_condition(task_complete_tool):
    """Test that the loop break condition works when a string doesn't change after stripping."""
    # Create a special test class that will help us test the loop break condition
    class SpecialString(str):
        """String subclass that helps test the loop break condition."""
        def startswith(self, *args, **kwargs):
            return True  # Always start with a strippable char

        def endswith(self, *args, **kwargs):
            return True  # Always end with a strippable char
            
        def strip(self, chars=None):
            # Return the same string, which should trigger the loop break condition
            return self
    
    # Create our special string and run the test
    input_summary = SpecialString("Text that never changes when stripped")
    
    # We need to patch the logging to avoid actual logging
    with mock.patch("src.cli_code.tools.task_complete_tool.log") as mock_log:
        result = task_complete_tool.execute(summary=input_summary)
        # The string is long enough so it should pass through without being marked insufficient
        assert result == input_summary 