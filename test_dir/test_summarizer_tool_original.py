"""
Tests for the summarizer tool module.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

# Direct import for coverage tracking
import src.cli_code.tools.summarizer_tool
from src.cli_code.tools.summarizer_tool import SummarizeCodeTool, MAX_LINES_FOR_FULL_CONTENT, MAX_CHARS_FOR_FULL_CONTENT

# Mock classes for google.generativeai
class MockCandidate:
    def __init__(self, text, finish_reason="STOP"):
        self.content = MagicMock()
        self.content.parts = [MagicMock(text=text)]
        self.finish_reason = MagicMock()
        self.finish_reason.name = finish_reason

class MockResponse:
    def __init__(self, text=None, finish_reason="STOP"):
        self.candidates = [MockCandidate(text, finish_reason)] if text is not None else []

class TestSummarizeCodeTool(unittest.TestCase):
    """Tests for the SummarizeCodeTool class."""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model
        self.mock_model = MagicMock()
        self.tool = SummarizeCodeTool(model_instance=self.mock_model)

    def test_init(self):
        """Test initialization of SummarizeCodeTool."""
        self.assertEqual(self.tool.name, "summarize_code")
        self.assertTrue("summary" in self.tool.description.lower())
        self.assertEqual(self.tool.model, self.mock_model)

    def test_init_without_model(self):
        """Test initialization without model."""
        tool = SummarizeCodeTool()
        self.assertIsNone(tool.model)

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open, read_data="Small file content")
    def test_execute_small_file(self, mock_file, mock_getsize, mock_isfile, mock_exists):
        """Test execution with a small file that returns full content."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = 100  # Small file
        
        # Execute with a test file path
        result = self.tool.execute(file_path="test_file.py")
        
        # Verify results
        self.assertIn("Full Content of test_file.py", result)
        self.assertIn("Small file content", result)
        # Ensure the model was not called for small files
        self.mock_model.generate_content.assert_not_called()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_execute_large_file(self, mock_open, mock_getsize, mock_isfile, mock_exists):
        """Test execution with a large file that generates a summary."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = MAX_CHARS_FOR_FULL_CONTENT + 1000  # Large file
        
        # Mock the file reading
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "Large file content" * 1000
        mock_open.return_value = mock_file
        
        # Mock the model response
        mock_response = MockResponse(text="This is a summary of the file")
        self.mock_model.generate_content.return_value = mock_response
        
        # Execute with a test file path
        result = self.tool.execute(file_path="large_file.py")
        
        # Verify results
        self.assertIn("Summary of large_file.py", result)
        self.assertIn("This is a summary of the file", result)
        self.mock_model.generate_content.assert_called_once()

    @patch("os.path.exists")
    def test_file_not_found(self, mock_exists):
        """Test handling of a non-existent file."""
        mock_exists.return_value = False
        
        # Execute with a non-existent file
        result = self.tool.execute(file_path="nonexistent.py")
        
        # Verify results
        self.assertIn("Error: File not found", result)
        self.mock_model.generate_content.assert_not_called()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_not_a_file(self, mock_isfile, mock_exists):
        """Test handling of a path that is not a file."""
        mock_exists.return_value = True
        mock_isfile.return_value = False
        
        # Execute with a directory path
        result = self.tool.execute(file_path="directory/")
        
        # Verify results
        self.assertIn("Error: Path is not a file", result)
        self.mock_model.generate_content.assert_not_called()

    def test_parent_directory_traversal(self):
        """Test protection against parent directory traversal."""
        # Execute with a path containing parent directory traversal
        result = self.tool.execute(file_path="../dangerous.py")
        
        # Verify results
        self.assertIn("Error: Invalid file path", result)
        self.mock_model.generate_content.assert_not_called()

    def test_missing_model(self):
        """Test execution when model is not provided."""
        # Create a tool without a model
        tool = SummarizeCodeTool()
        
        # Execute without a model
        result = tool.execute(file_path="test.py")
        
        # Verify results
        self.assertIn("Error: Summarization tool not properly configured", result)

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_empty_file(self, mock_open, mock_getsize, mock_isfile, mock_exists):
        """Test handling of an empty file for summarization."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = MAX_CHARS_FOR_FULL_CONTENT + 1000  # Large but empty file
        
        # Mock the file reading to return empty content
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = ""
        mock_open.return_value = mock_file
        
        # Execute with a test file path
        result = self.tool.execute(file_path="empty_file.py")
        
        # Verify results
        self.assertIn("Summary of empty_file.py", result)
        self.assertIn("(File is empty)", result)
        # Model should not be called for empty files
        self.mock_model.generate_content.assert_not_called()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_file_read_error(self, mock_open, mock_getsize, mock_isfile, mock_exists):
        """Test handling of errors when reading a file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = 100  # Small file
        mock_open.side_effect = IOError("Error reading file")
        
        # Execute with a test file path
        result = self.tool.execute(file_path="error_file.py")
        
        # Verify results
        self.assertIn("Error reading file", result)
        self.mock_model.generate_content.assert_not_called()

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_summarization_error(self, mock_open, mock_getsize, mock_isfile, mock_exists):
        """Test handling of errors during summarization."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_getsize.return_value = MAX_CHARS_FOR_FULL_CONTENT + 1000  # Large file
        
        # Mock the file reading
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "Large file content" * 1000
        mock_open.return_value = mock_file
        
        # Mock the model to raise an exception
        self.mock_model.generate_content.side_effect = Exception("Summarization error")
        
        # Execute with a test file path
        result = self.tool.execute(file_path="error_summarize.py")
        
        # Verify results
        self.assertIn("Error generating summary", result)
        self.mock_model.generate_content.assert_called_once()

    def test_extract_text_success(self):
        """Test successful text extraction from summary response."""
        # Create a mock response with text
        mock_response = MockResponse(text="Extracted summary text")
        
        # Extract text
        result = self.tool._extract_text_from_summary_response(mock_response)
        
        # Verify results
        self.assertEqual(result, "Extracted summary text")

    def test_extract_text_no_candidates(self):
        """Test text extraction when no candidates are available."""
        # Create a mock response without candidates
        mock_response = MockResponse()
        mock_response.candidates = []
        
        # Extract text
        result = self.tool._extract_text_from_summary_response(mock_response)
        
        # Verify results
        self.assertEqual(result, "(Summarization failed: No candidates)")

    def test_extract_text_failed_finish_reason(self):
        """Test text extraction when finish reason is not STOP."""
        # Create a mock response with a failed finish reason
        mock_response = MockResponse(text="Partial text", finish_reason="ERROR")
        
        # Extract text
        result = self.tool._extract_text_from_summary_response(mock_response)
        
        # Verify results
        self.assertEqual(result, "(Summarization failed: ERROR)")

    def test_extract_text_exception(self):
        """Test handling of exceptions during text extraction."""
        # Create a test response with a structure that will cause an exception
        # when accessing candidates
        
        # Create a response object that raises an exception when candidates is accessed
        class ExceptionRaisingResponse:
            @property
            def candidates(self):
                raise Exception("Extraction error")
        
        # Call the method directly
        result = self.tool._extract_text_from_summary_response(ExceptionRaisingResponse())
        
        # Verify results
        self.assertEqual(result, "(Error extracting summary text)")


if __name__ == "__main__":
    unittest.main() 