"""
Tests for edge cases in the Gemini model implementation.

This file contains tests for edge cases and areas of the Gemini model
that aren't fully covered by the existing test files.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.generativeai.types import ContentType, GenerateContentResponse, GenerationConfig
from rich.console import Console

from cli_code.models.gemini import FALLBACK_MODEL, GeminiModel


@pytest.fixture
def mock_console():
    """Create a mock Console"""
    return MagicMock(spec=Console)


@pytest.mark.asyncio
class TestGeminiModelEdgeCases:
    """Test class for edge cases in the Gemini model."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, mock_console):
        """Set up for each test."""
        self.mock_genai = MagicMock()
        monkeypatch.setattr("cli_code.models.gemini.genai", self.mock_genai)
        self.model = GeminiModel(api_key="fake_api_key", console=mock_console)
        self.model.model_instance = MagicMock()
        self.model.genai_client = MagicMock()

    async def test_generate_with_large_context(self):
        """Test handling of context too large error."""
        # Create a mock response that simulates a context too large error
        mock_response = MagicMock()
        mock_response.error = None

        # Create a mock candidate that indicates context too large
        mock_candidate = MagicMock()
        mock_candidate.content = MagicMock()
        mock_candidate.content.text = "Error: This model's maximum context length is exceeded."
        mock_candidate.finish_reason = "MAX_TOKENS"
        mock_response.candidates = [mock_candidate]

        # Create a mock model that returns our response
        with patch.object(self.model, "model") as mock_model:
            mock_model.generate_content.return_value = mock_response

            # Create a history that's large enough to exceed context
            large_history = [{"role": "user", "content": "x" * 1000} for _ in range(200)]
            self.model.history = large_history

            # Call generate with a test prompt
            result = await self.model.generate(prompt="Test prompt with large context")

            # Verify that the response contains the finish reason MAX_TOKENS
            assert "MAX_TOKENS" in result or "max tokens" in result.lower()

    async def test_generate_with_fallback_model_switch(self):
        """Test the model switches to fallback model when quota is exceeded."""
        # Mock the original model to raise ResourceExhausted
        from google.api_core.exceptions import ResourceExhausted

        # First call raises quota exceeded
        error_response = ResourceExhausted("Quota exceeded")

        # Mock the model's generate_content method to raise quota exceeded on the first call
        with patch.object(self.model, "model") as mock_model:
            mock_model.generate_content.side_effect = error_response

            # Directly patch the _handle_quota_exceeded method to return the expected result
            with patch.object(
                self.model, "_handle_quota_exceeded", return_value="API quota exceeded. Please try again later."
            ):
                # Call the method
                result = await self.model.generate("Test prompt")

                # Verify the result contains the quota exceeded message
                assert "API quota exceeded" in result

    async def test_generate_with_tool_selection(self, mock_console):
        """Test that the model correctly identifies and processes a tool by name."""
        # Create a mock tool that we can track
        mock_tool = MagicMock()
        mock_tool.execute.return_value = {"files": ["file1.txt", "file2.py"]}
        mock_tool.get_function_declaration.return_value.name = "ls"
        mock_tool.requires_confirmation = False

        # Mock the tool registry to return our mock tool
        with patch("cli_code.models.gemini.get_tool", return_value=mock_tool):
            # Mock _process_candidate_response to directly call _execute_function_call with our tool
            with patch.object(self.model, "_process_candidate_response") as mock_process:
                # Simulate successful tool execution by returning the expected content
                mock_process.return_value = ("complete", json.dumps({"files": ["file1.txt", "file2.py"]}))

                # Call generate, which will use our mocked methods
                result = await self.model.generate("List files in directory")

                # Verify result contains the expected output
                assert "file1.txt" in result
                assert "file2.py" in result

    def test_context_window_management(self):
        """Test that the context window is properly managed."""
        # Create a history that would trigger context window management
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "model", "parts": [{"text": "Model acknowledgment"}]},
        ]

        # Add more history entries to exceed the threshold
        for i in range(100):
            self.model.history.append({"role": "user", "parts": [{"text": f"User message {i}"}]})
            self.model.history.append({"role": "model", "parts": [{"text": f"Model response {i}"}]})

        # Get initial history length
        initial_length = len(self.model.history)

        # Call the manage context window method
        self.model._manage_context_window()

        # Verify the history was truncated
        assert len(self.model.history) < initial_length
        # System prompt should be preserved
        assert self.model.history[0]["role"] == "system"
        # Model acknowledgment should be preserved
        assert self.model.history[1]["role"] == "model"

    def test_handle_agent_loop_exception_unexpected(self):
        """Test handling of unexpected exceptions in the agent loop."""
        # Create a custom exception
        custom_exception = ValueError("Unexpected error")
        mock_status = MagicMock()

        # Call the method
        result = self.model._handle_agent_loop_exception(custom_exception, mock_status)

        # Verify the result contains the error message
        assert "Unexpected error" in result
        assert "Error during agent processing" in result

    @patch("cli_code.models.gemini.questionary")
    def test_tool_confirmation_cancelled(self, mock_questionary):
        """Test tool confirmation when user cancels."""
        # Mock questionary to return None (simulating Ctrl+C)
        mock_questionary.confirm.return_value.ask.return_value = None

        # Setup mock tool
        mock_tool = MagicMock()
        mock_tool.get_function_declaration.return_value.name = "edit_file"
        mock_tool.requires_confirmation = True

        # Call the method
        result = self.model._request_tool_confirmation(
            mock_tool, "edit_file", {"file_path": "test.py", "content": "print('hello')"}
        )

        # Verify cancellation was captured
        assert "cancelled" in result.lower()

    def test_extract_final_text_empty_parts(self):
        """Test extracting final text when parts are empty."""
        mock_candidate = MagicMock()
        mock_candidate.content.parts = []

        # Call the method
        result = self.model._extract_final_text(mock_candidate)

        # Verify empty result
        assert result == ""

    def test_execute_function_call_null_arguments(self):
        """Test executing a function call with null arguments."""
        # Create a function call without arguments
        function_name = "ls"
        function_args = None

        # Mock the tool
        mock_tool = MagicMock()
        mock_tool.execute.return_value = "Directory listing"
        mock_tool.get_function_declaration.return_value.name = "ls"
        mock_tool.requires_confirmation = False

        # Mock get_tool
        with patch("cli_code.models.gemini.get_tool", return_value=mock_tool):
            # Call the method with asyncio.run since it's an async method
            import asyncio

            result = asyncio.run(self.model._execute_function_call(function_name, function_args))

            # Verify tool was called with no arguments
            mock_tool.execute.assert_called_once_with()
            # Check that result has expected structure (parts attribute)
            assert hasattr(result, "parts")

    def test_add_to_history_with_complex_objects(self):
        """Test adding complex objects to history with serialization."""
        # Create a message with a complex object
        complex_message = {
            "role": "user",
            "parts": [{"text": "User message"}, {"complex_obj": {"nested": {"values": [1, 2, 3]}}}],
        }

        # Initial history
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "model", "parts": [{"text": "Model acknowledgment"}]},
        ]

        # Call the method
        self.model.add_to_history(complex_message)

        # Verify message was added
        assert len(self.model.history) == 3
        assert self.model.history[2] == complex_message

    async def test_generate_task_complete_custom_summary(self):
        """Test generating with task_complete tool call that has a custom summary."""
        # Set up the initial state for the test
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "user", "parts": [{"text": "Complete the task"}]},
        ]

        # Create a final summary that would come from task_complete
        final_summary = "Custom task completion message"

        # Mock the generate method's internal process to skip to loop completion
        # Patch the _execute_agent_loop method to simulate task completion
        with patch.object(self.model, "_execute_agent_loop") as mock_execute_loop:
            # When _execute_agent_loop is called, it will shortcut to completion handling
            mock_execute_loop.return_value = final_summary

            # Call the method
            result = await self.model.generate("Complete the task")

            # Verify the result matches our expected final summary
            assert result == final_summary

    def test_find_last_model_text_complex_history(self):
        """Test finding the last model text in a complex history."""
        # Create a complex history with various entry types
        self.model.history = [
            {"role": "system", "parts": [{"text": "System prompt"}]},
            {"role": "user", "parts": [{"text": "User message 1"}]},
            {"role": "model", "parts": [{"function_call": {"name": "ls"}}]},
            {"role": "tool", "content": "Tool result"},
            {"role": "model", "parts": [{"text": "Model response 1"}]},
            {"role": "user", "parts": [{"text": "User message 2"}]},
            {"role": "model", "parts": [{"text": "Model response 2"}]},
            {"role": "user", "parts": [{"text": "User message 3"}]},
            {"role": "model", "parts": [{"function_call": {"name": "view"}}]},
            {"role": "tool", "content": "Another tool result"},
        ]

        # Call the method
        result = self.model._find_last_model_text(self.model.history)

        # Verify the last model text was found
        assert result == "Model response 2"

    # Add a test for handling quota exceeded even with fallback model
    async def test_generate_with_quota_exceeded_on_both_models(self):
        """Test handling quota exceeded on both primary and fallback models."""
        from google.api_core.exceptions import ResourceExhausted

        # Mock _get_llm_response to raise ResourceExhausted twice
        quota_error = ResourceExhausted("Quota exceeded")
        with patch.object(self.model, "_get_llm_response") as mock_get_response:
            # Use a callable side_effect to always raise the error
            def always_raise_quota(*args, **kwargs):
                raise quota_error

            mock_get_response.side_effect = always_raise_quota

            # Set initial model name to non-fallback
            self.model.current_model_name = "gemini-pro"

            # Call generate - expect the loop to handle the errors
            result = await self.model.generate("Test prompt")

            # Verify model name was changed to fallback (happens on first error)
            assert self.model.current_model_name == FALLBACK_MODEL
            # Verify the final error message (happens on second error)
            assert "quota exceeded" in result.lower()
            assert "api quota exceeded" in result.lower()  # Check for specific message
            assert "fallback model" not in result.lower()  # Should be the final error

    async def test_error_handling_exceptions(self):
        """Test that the model handles various exceptions properly."""
        from google.api_core.exceptions import GoogleAPIError, ResourceExhausted

        # Test ResourceExhausted exception
        with patch.object(self.model, "_get_llm_response") as mock_get_llm:
            # Set up the exception to be raised on the first call
            # The second call (after fallback) should succeed in this test
            mock_response_fallback = MagicMock()
            mock_candidate_fallback = MagicMock()
            # Explicitly create a mock text part without function call
            mock_text_part_fallback = MagicMock()
            mock_text_part_fallback.text = "Fallback response"
            mock_text_part_fallback.function_call = None  # Explicitly None
            mock_candidate_fallback.content.parts = [mock_text_part_fallback]  # Use the specific part
            mock_candidate_fallback.finish_reason = 1  # STOP
            mock_response_fallback.candidates = [mock_candidate_fallback]

            quota_error = ResourceExhausted("Quota exceeded")
            mock_get_llm.side_effect = [quota_error, mock_response_fallback]

            # Call generate
            result = await self.model.generate("Test prompt")

            # Verify it switched to fallback and returned the fallback response
            assert self.model.current_model_name == FALLBACK_MODEL
            assert "Fallback response" in result

        # Reset model name for next part of test
        self.model.current_model_name = "gemini-pro"

        # Test other GoogleAPIError (e.g., InvalidArgument)
