"""
Tests for summarizer_tool module.
"""

import os
from unittest.mock import MagicMock, mock_open, patch

import google.generativeai as genai
import pytest

# Direct import for coverage tracking
import src.cli_code.tools.summarizer_tool
from src.cli_code.tools.summarizer_tool import (
    MAX_CHARS_FOR_FULL_CONTENT,
    MAX_LINES_FOR_FULL_CONTENT,
    SUMMARIZATION_SYSTEM_PROMPT,
    SummarizeCodeTool,
)


# Mock classes for google.generativeai response structure
class MockPart:
    def __init__(self, text):
        self.text = text


class MockContent:
    def __init__(self, parts):
        self.parts = parts


class MockFinishReason:
    def __init__(self, name):
        self.name = name


class MockCandidate:
    def __init__(self, content, finish_reason):
        self.content = content
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, candidates=None):
        self.candidates = candidates if candidates is not None else []


def test_summarize_code_tool_init():
    """Test SummarizeCodeTool initialization."""
    # Create a mock model
    mock_model = MagicMock()

    # Initialize tool with model
    tool = SummarizeCodeTool(model_instance=mock_model)

    # Verify initialization
    assert tool.name == "summarize_code"
    assert "summary" in tool.description
    assert tool.model == mock_model


def test_summarize_code_tool_init_without_model():
    """Test SummarizeCodeTool initialization without a model."""
    # Initialize tool without model
    tool = SummarizeCodeTool()

    # Verify initialization with None model
    assert tool.model is None


def test_execute_without_model():
    """Test executing the tool without providing a model."""
    # Initialize tool without model
    tool = SummarizeCodeTool()

    # Execute tool
    result = tool.execute(file_path="test.py")

    # Verify error message
    assert "Error: Summarization tool not properly configured" in result


def test_execute_with_parent_directory_traversal():
    """Test executing the tool with a file path containing parent directory traversal."""
    # Initialize tool with mock model
    tool = SummarizeCodeTool(model_instance=MagicMock())

    # Execute tool with parent directory traversal
    result = tool.execute(file_path="../dangerous.py")

    # Verify error message
    assert "Error: Invalid file path" in result


@patch("os.path.exists")
def test_execute_file_not_found(mock_exists):
    """Test executing the tool with a non-existent file."""
    # Setup mock
    mock_exists.return_value = False

    # Initialize tool with mock model
    tool = SummarizeCodeTool(model_instance=MagicMock())

    # Execute tool with non-existent file
    result = tool.execute(file_path="nonexistent.py")

    # Verify error message
    assert "Error: File not found" in result


@patch("os.path.exists")
@patch("os.path.isfile")
def test_execute_not_a_file(mock_isfile, mock_exists):
    """Test executing the tool with a path that is not a file."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = False

    # Initialize tool with mock model
    tool = SummarizeCodeTool(model_instance=MagicMock())

    # Execute tool with directory path
    result = tool.execute(file_path="directory/")

    # Verify error message
    assert "Error: Path is not a file" in result


@patch("os.path.exists")
@patch("os.path.isfile")
@patch("os.path.getsize")
@patch("builtins.open", new_callable=mock_open, read_data="Small file content")
def test_execute_small_file(mock_file, mock_getsize, mock_isfile, mock_exists):
    """Test executing the tool with a small file."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = True
    mock_getsize.return_value = 100  # Small file size

    # Create mock for line counting - small file
    mock_file_handle = mock_file()
    mock_file_handle.__iter__.return_value = ["Line 1", "Line 2", "Line 3"]

    # Initialize tool with mock model
    mock_model = MagicMock()
    tool = SummarizeCodeTool(model_instance=mock_model)

    # Execute tool with small file
    result = tool.execute(file_path="small_file.py")

    # Verify full content returned and model not called
    assert "Full Content of small_file.py" in result
    assert "Small file content" in result
    mock_model.generate_content.assert_not_called()


@patch("os.path.exists")
@patch("os.path.isfile")
@patch("os.path.getsize")
@patch("builtins.open")
def test_execute_large_file(mock_file, mock_getsize, mock_isfile, mock_exists):
    """Test executing the tool with a large file."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = True
    mock_getsize.return_value = MAX_CHARS_FOR_FULL_CONTENT + 1000  # Large file

    # Create mock file handle for line counting - large file
    file_handle = MagicMock()
    file_handle.__iter__.return_value = ["Line " + str(i) for i in range(MAX_LINES_FOR_FULL_CONTENT + 100)]
    # Create mock file handle for content reading
    file_handle_read = MagicMock()
    file_handle_read.read.return_value = "Large file content " * 1000

    # Set up different return values for different calls to open()
    mock_file.side_effect = [file_handle, file_handle_read]

    # Create mock model response
    mock_model = MagicMock()
    mock_parts = [MockPart("This is a summary of the large file.")]
    mock_content = MockContent(mock_parts)
    mock_finish_reason = MockFinishReason("STOP")
    mock_candidate = MockCandidate(mock_content, mock_finish_reason)
    mock_response = MockResponse([mock_candidate])
    mock_model.generate_content.return_value = mock_response

    # Initialize tool with mock model
    tool = SummarizeCodeTool(model_instance=mock_model)

    # Execute tool with large file
    result = tool.execute(file_path="large_file.py")

    # Verify summary returned and model called
    assert "Summary of large_file.py" in result
    assert "This is a summary of the large file." in result
    mock_model.generate_content.assert_called_once()

    # Verify prompt content
    call_args = mock_model.generate_content.call_args[1]
    assert "contents" in call_args

    # Verify system prompt
    contents = call_args["contents"][0]
    assert "role" in contents
    assert "parts" in contents
    assert SUMMARIZATION_SYSTEM_PROMPT in contents["parts"]


@patch("os.path.exists")
@patch("os.path.isfile")
@patch("os.path.getsize")
@patch("builtins.open")
def test_execute_with_empty_large_file(mock_file, mock_getsize, mock_isfile, mock_exists):
    """Test executing the tool with a large but empty file."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = True
    mock_getsize.return_value = MAX_CHARS_FOR_FULL_CONTENT + 1000  # Large file

    # Create mock file handle for line counting - large file
    file_handle = MagicMock()
    file_handle.__iter__.return_value = ["Line " + str(i) for i in range(MAX_LINES_FOR_FULL_CONTENT + 100)]
    # Create mock file handle for content reading - truly empty content (not just whitespace)
    file_handle_read = MagicMock()
    file_handle_read.read.return_value = ""  # Truly empty, not whitespace

    # Set up different return values for different calls to open()
    mock_file.side_effect = [file_handle, file_handle_read]

    # Initialize tool with mock model
    mock_model = MagicMock()
    # Setup mock response from model
    mock_parts = [MockPart("This is a summary of an empty file.")]
    mock_content = MockContent(mock_parts)
    mock_finish_reason = MockFinishReason("STOP")
    mock_candidate = MockCandidate(mock_content, mock_finish_reason)
    mock_response = MockResponse([mock_candidate])
    mock_model.generate_content.return_value = mock_response

    # Execute tool with large but empty file
    tool = SummarizeCodeTool(model_instance=mock_model)
    result = tool.execute(file_path="empty_large_file.py")

    # Verify that the model was called with appropriate parameters
    mock_model.generate_content.assert_called_once()

    # Verify the result contains a summary
    assert "Summary of empty_large_file.py" in result
    assert "This is a summary of an empty file." in result


@patch("os.path.exists")
@patch("os.path.isfile")
@patch("os.path.getsize")
@patch("builtins.open")
def test_execute_with_file_read_error(mock_file, mock_getsize, mock_isfile, mock_exists):
    """Test executing the tool with a file that has a read error."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = True
    mock_getsize.return_value = 100  # Small file

    # Create mock for file read error
    mock_file.side_effect = IOError("Read error")

    # Initialize tool with mock model
    mock_model = MagicMock()
    tool = SummarizeCodeTool(model_instance=mock_model)

    # Execute tool with file that has read error
    result = tool.execute(file_path="error_file.py")

    # Verify error message and model not called
    assert "Error" in result
    assert "Read error" in result
    mock_model.generate_content.assert_not_called()


@patch("os.path.exists")
@patch("os.path.isfile")
@patch("os.path.getsize")
@patch("builtins.open")
def test_execute_with_summarization_error(mock_file, mock_getsize, mock_isfile, mock_exists):
    """Test executing the tool when summarization fails."""
    # Setup mocks
    mock_exists.return_value = True
    mock_isfile.return_value = True
    mock_getsize.return_value = MAX_CHARS_FOR_FULL_CONTENT + 1000  # Large file

    # Create mock file handle for line counting - large file
    file_handle = MagicMock()
    file_handle.__iter__.return_value = ["Line " + str(i) for i in range(MAX_LINES_FOR_FULL_CONTENT + 100)]
    # Create mock file handle for content reading
    file_handle_read = MagicMock()
    file_handle_read.read.return_value = "Large file content " * 1000

    # Set up different return values for different calls to open()
    mock_file.side_effect = [file_handle, file_handle_read]

    # Create mock model with error
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("Summarization error")

    # Initialize tool with mock model
    tool = SummarizeCodeTool(model_instance=mock_model)

    # Execute tool when summarization fails
    result = tool.execute(file_path="error_summarize.py")

    # Verify error message
    assert "Error generating summary" in result
    assert "Summarization error" in result
    mock_model.generate_content.assert_called_once()


def test_extract_text_success():
    """Test extracting text from a successful response."""
    # Create mock response with successful candidate
    mock_parts = [MockPart("Part 1 text."), MockPart("Part 2 text.")]
    mock_content = MockContent(mock_parts)
    mock_finish_reason = MockFinishReason("STOP")
    mock_candidate = MockCandidate(mock_content, mock_finish_reason)
    mock_response = MockResponse([mock_candidate])

    # Initialize tool and extract text
    tool = SummarizeCodeTool(model_instance=MagicMock())
    result = tool._extract_text_from_summary_response(mock_response)

    # Verify text extraction
    assert result == "Part 1 text.Part 2 text."


def test_extract_text_with_failed_finish_reason():
    """Test extracting text when finish reason indicates failure."""
    # Create mock response with error finish reason
    mock_parts = [MockPart("Partial text")]
    mock_content = MockContent(mock_parts)
    mock_finish_reason = MockFinishReason("ERROR")
    mock_candidate = MockCandidate(mock_content, mock_finish_reason)
    mock_response = MockResponse([mock_candidate])

    # Initialize tool and extract text
    tool = SummarizeCodeTool(model_instance=MagicMock())
    result = tool._extract_text_from_summary_response(mock_response)

    # Verify failure message with reason
    assert result == "(Summarization failed: ERROR)"


def test_extract_text_with_no_candidates():
    """Test extracting text when response has no candidates."""
    # Create mock response with no candidates
    mock_response = MockResponse([])

    # Initialize tool and extract text
    tool = SummarizeCodeTool(model_instance=MagicMock())
    result = tool._extract_text_from_summary_response(mock_response)

    # Verify failure message for no candidates
    assert result == "(Summarization failed: No candidates)"


def test_extract_text_with_exception():
    """Test extracting text when an exception occurs."""

    # Create mock response that will cause exception
    class ExceptionResponse:
        @property
        def candidates(self):
            raise Exception("Extraction error")

    # Initialize tool and extract text
    tool = SummarizeCodeTool(model_instance=MagicMock())
    result = tool._extract_text_from_summary_response(ExceptionResponse())

    # Verify exception message
    assert result == "(Error extracting summary text)"


@patch("os.path.exists")
@patch("os.path.isfile")
@patch("os.path.getsize")
@patch("builtins.open")
def test_execute_general_exception(mock_file, mock_getsize, mock_isfile, mock_exists):
    """Test executing the tool when a general exception occurs."""
    # Setup mocks to raise exception outside the normal flow
    mock_exists.side_effect = Exception("Unexpected general error")

    # Initialize tool with mock model
    mock_model = MagicMock()
    tool = SummarizeCodeTool(model_instance=mock_model)

    # Execute tool with unexpected error
    result = tool.execute(file_path="file.py")

    # Verify error message
    assert "Error processing file for summary/view" in result
    assert "Unexpected general error" in result
    mock_model.generate_content.assert_not_called()
