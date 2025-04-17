"""
Additional tests for the GeminiModel class to improve code coverage.

These tests specifically target methods and code paths that were
not covered by existing tests.
"""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from cli_code.models.gemini import MAX_AGENT_ITERATIONS, GeminiModel


class TestGeminiModelAdditionalCoverage(unittest.TestCase):
    """Additional test cases specifically designed to improve code coverage."""

    @pytest.fixture(autouse=True)
    def setup_gemini_model(self, monkeypatch):
        """Set up a GeminiModel instance with mocked dependencies."""
        self.mock_console = MagicMock(spec=Console)
        self.mock_genai = MagicMock()
        monkeypatch.setattr("cli_code.models.gemini.genai", self.mock_genai)

        # Setup model instance
        self.model = GeminiModel(api_key="fake_api_key", console=self.mock_console)
        self.model.genai_client = MagicMock()
        self.model.model_instance = MagicMock()
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "model", "parts": [{"text": "Model acknowledgment"}]},
        ]

    def test_validate_prompt_and_model_with_empty_prompt(self):
        """Test validation with empty prompt."""
        result = self.model._validate_prompt_and_model("")
        assert result is False

    def test_validate_prompt_and_model_without_model_instance(self):
        """Test validation without model instance."""
        self.model.model_instance = None
        result = self.model._validate_prompt_and_model("Test prompt")
        assert result is True

    def test_handle_empty_response(self):
        """Test handling of empty responses."""
        mock_response = MagicMock()
        mock_response.candidates = []
        # Configure the mock to have a block_reason when accessed
        mock_response.prompt_feedback.block_reason.name = "BLOCKED_REASON"
        result = self.model._handle_empty_response(mock_response)
        assert isinstance(result, str)
        assert "Error: Prompt was blocked by API" in result

    def test_handle_null_content_max_tokens(self):
        """Test handling null content with MAX_TOKENS reason."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "MAX_TOKENS"
        mock_candidate.content.parts = []
        result = self.model._handle_null_content(mock_candidate)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "error"
        assert "MAX_TOKENS" in result[1]

    def test_handle_null_content_recitation(self):
        """Test handling null content with RECITATION reason."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "RECITATION"
        mock_candidate.content.parts = []
        result = self.model._handle_null_content(mock_candidate)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "error"
        assert "RECITATION" in result[1]

    def test_handle_null_content_other_reason(self):
        """Test handling null content with OTHER finish reason."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "OTHER"
        mock_candidate.content.parts = []
        result = self.model._handle_null_content(mock_candidate)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "error"
        assert "OTHER" in result[1]

    def test_handle_no_actionable_content(self):
        """Test handling response with no actionable content."""
        mock_candidate = MagicMock()
        result = self.model._handle_no_actionable_content(mock_candidate)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "error"
        assert "finish reason" in result[1].lower()

    def test_store_tool_result_with_dict(self):
        """Test storing tool result when result is a dictionary."""
        # Setup a mock history before the test
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "user", "parts": [{"text": "Test request"}]},
            {"role": "model", "parts": [{"function_call": {"name": "test_tool"}}]},
        ]

        # Call the method with a dictionary result
        result_dict = {"key1": "value1", "key2": "value2"}
        self.model._store_tool_result("test_tool", {"arg1": "value1"}, result_dict)

        # Verify the result was stored correctly
        assert len(self.model.history) == 4
        assert self.model.history[-1]["role"] == "tool"
        # The result is stored in parts[0]["text"] rather than "content"
        assert "key1" in self.model.history[-1]["parts"][0]["text"]
        assert "value1" in self.model.history[-1]["parts"][0]["text"]

    def test_store_tool_result_with_string(self):
        """Test storing tool result when result is a string."""
        # Setup a mock history before the test
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "user", "parts": [{"text": "Test request"}]},
            {"role": "model", "parts": [{"function_call": {"name": "test_tool"}}]},
        ]

        # Call the method with a string result
        result_str = "This is a string result"
        self.model._store_tool_result("test_tool", {"arg1": "value1"}, result_str)

        # Verify the result was stored correctly
        assert len(self.model.history) == 4
        assert self.model.history[-1]["role"] == "tool"
        # The result is stored in parts[0]["text"] rather than "content"
        assert result_str in self.model.history[-1]["parts"][0]["text"]

    def test_handle_loop_completion_with_max_iterations(self):
        """Test handling loop completion when max iterations reached."""
        result = self.model._handle_loop_completion(False, None, MAX_AGENT_ITERATIONS)
        assert f"Task exceeded max iterations ({MAX_AGENT_ITERATIONS})" in result

    def test_handle_loop_completion_with_task_completed(self):
        """Test handling loop completion when task is completed."""
        final_summary = "Task completed successfully"
        result = self.model._handle_loop_completion(True, final_summary, 5)
        assert result == final_summary

    @patch("cli_code.models.gemini.questionary")
    def test_request_tool_confirmation_rejected(self, mock_questionary):
        """Test tool confirmation that gets rejected."""
        # Mock questionary to simulate user rejecting
        mock_questionary.confirm.return_value.ask.return_value = False

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.get_function_declaration.return_value.name = "edit_file"

        # Call the method
        result = self.model._request_tool_confirmation(
            mock_tool, "edit_file", {"file_path": "test.py", "content": "print('hello')"}
        )

        # Verify rejection was handled correctly
        assert "rejected" in result.lower()
        mock_questionary.confirm.assert_called_once()

    async def test_request_tool_confirmation_async(self):
        """Test the async version of tool confirmation."""
        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.get_function_declaration.return_value.name = "edit_file"

        # Use AsyncMock for questionary
        with patch("cli_code.models.gemini.questionary") as mock_questionary:
            mock_confirm = MagicMock()
            mock_confirm.ask.return_value = True
            mock_questionary.confirm.return_value = mock_confirm

            # Call the async method
            result = await self.model._request_tool_confirmation_async(
                mock_tool, "edit_file", {"file_path": "test.py", "content": "print('hello')"}
            )

            # Verify confirmation was handled correctly
            assert result == "confirmed"

    def test_handle_task_complete(self):
        """Test handling task_complete tool."""
        tool_args = {"summary": "Task has been completed successfully"}
        result = self.model._handle_task_complete(tool_args)
        assert result == ("task_completed", True)

        # Test with no summary provided
        result = self.model._handle_task_complete({})
        assert result == ("task_completed", True)

    def test_find_last_model_text_with_text(self):
        """Test finding the last model text when it exists."""
        # Setup history with model text - use proper dictionary structure
        # The implementation is looking for objects with a 'text' attribute, but in tests we need to use dictionaries
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "user", "parts": [{"text": "User message 1"}]},
            {"role": "model", "parts": ["Model response 1"]},  # Use simple string in parts array
            {"role": "user", "parts": [{"text": "User message 2"}]},
            {"role": "model", "parts": ["Model response 2"]},
        ]

        result = self.model._find_last_model_text(self.model.history)
        assert result == "Model response 2"

    def test_find_last_model_text_no_text(self):
        """Test finding the last model text when it doesn't exist."""
        # Setup history without model text
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "user", "parts": [{"text": "User message 1"}]},
            {"role": "model", "parts": [{"function_call": {"name": "tool"}}]},
        ]

        result = self.model._find_last_model_text(self.model.history)
        assert result is None


if __name__ == "__main__":
    unittest.main()
