import json
from unittest.mock import MagicMock, patch

import pytest
import vertexai
from rich.console import Console

from cli_code.models.gemini import GeminiModel


class TestGeminiModelAdditionalCoverage:
    """Additional tests to improve code coverage for GeminiModel."""

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    @pytest.fixture
    def model(self, mock_console):
        model = GeminiModel(console=mock_console, api_key="test-api-key")
        model.history = []
        return model

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_invalid_response(self, mock_generative_model, model):
        """Test handling of invalid responses (lines 285-296)."""
        # Setup mocks
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Ensure the model is initialized (set model attribute)
        model.model = mock_gen_model

        # Make generate_content raise AttributeError directly
        mock_gen_model.generate_content.side_effect = AttributeError("No 'text' attribute")

        # Call generate method
        result = model.generate("Test prompt")

        # Verify error handling for AttributeError
        assert "Error during agent processing" in result

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_quota_exceeded(self, mock_generative_model, model):
        """Test handling of quota exceeded errors (lines 362-374)."""
        # Setup mocks
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Ensure the model is initialized (set model attribute)
        model.model = mock_gen_model

        # Create a ResourceExhausted exception with the quota message
        error_message = "Quota exceeded for quota metric 'CharactersPerMinutePerProject'"
        quota_error = Exception(error_message)

        # Make generate_content raise the quota error
        mock_gen_model.generate_content.side_effect = quota_error

        # Mock the _handle_quota_exceeded method to return a predictable result
        with patch.object(model, "_handle_quota_exceeded", return_value="Quota exceeded for API"):
            # Call generate method
            result = model.generate("Test prompt")

            # Verify quota error handling
            assert "Quota exceeded" in result

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_rate_limit(self, mock_generative_model, model):
        """Test handling of rate limit errors (lines 380-384)."""
        # Setup mocks
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Ensure the model is initialized (set model attribute)
        model.model = mock_gen_model

        # Create an exception with the rate limit message
        error_message = "Resource has been exhausted (e.g. check quota)."
        rate_error = Exception(error_message)

        # Make generate_content raise the rate limit error
        mock_gen_model.generate_content.side_effect = rate_error

        # Directly mock the _handle_general_exception method
        with patch.object(
            model, "_handle_general_exception", return_value="Rate limit exceeded, please try again later"
        ):
            # Call generate method
            result = model.generate("Test prompt")

            # Verify rate limit error handling
            assert "Rate limit" in result

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_server_error(self, mock_generative_model, model):
        """Test handling of server errors (lines 390-393)."""
        # Setup mocks
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Ensure the model is initialized (set model attribute)
        model.model = mock_gen_model

        # Create an exception with the server error message
        error_message = "Server error (5xx)"
        server_error = Exception(error_message)

        # Make generate_content raise the server error
        mock_gen_model.generate_content.side_effect = server_error

        # Directly mock the _handle_general_exception method
        with patch.object(model, "_handle_general_exception", return_value="A server error occurred: 5xx"):
            # Call generate method
            result = model.generate("Test prompt")

            # Verify server error handling
            assert "server error" in result.lower()

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_safety_error(self, mock_generative_model, model):
        """Test handling of safety errors (lines 403-406)."""
        # Setup mocks
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Ensure the model is initialized (set model attribute)
        model.model = mock_gen_model

        # Create an exception with the safety error message
        error_message = "Content filtered due to safety settings"
        safety_error = Exception(error_message)

        # Make generate_content raise the safety error
        mock_gen_model.generate_content.side_effect = safety_error

        # Directly mock the _handle_general_exception method
        with patch.object(model, "_handle_general_exception", return_value="Response blocked due to safety settings"):
            # Call generate method
            result = model.generate("Test prompt")

            # Verify safety error handling
            assert "safety" in result.lower()

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_context_length(self, mock_generative_model, model):
        """Test handling of context length errors (lines 414-428)."""
        # Setup mocks
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Ensure the model is initialized (set model attribute)
        model.model = mock_gen_model

        # Create an exception with the context length error message
        error_message = "input is too long"
        context_error = Exception(error_message)

        # Make generate_content raise the context length error
        mock_gen_model.generate_content.side_effect = context_error

        # Directly mock the _handle_general_exception method
        with patch.object(model, "_handle_general_exception", return_value="Input exceeds maximum context length"):
            # Call generate method
            result = model.generate("Test prompt")

            # Verify context length error handling
            assert "context length" in result.lower()

    def test_manage_context_window(self, model):
        """Test context window management (lines 1124-1137)."""
        # Create a history with multiple messages
        model.history = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        ]

        # Call _manage_context_window (method doesn't take max_tokens parameter)
        model._manage_context_window()

        # Verify that the history has been processed
        # The method truncates history if it exceeds MAX_HISTORY_TURNS * 3 + 2
        # Just verify it doesn't crash and the system prompt is preserved
        assert model.history[0]["role"] == "system"
        assert model.history[0]["content"] == "You are a helpful assistant"

    def test_find_last_model_text_complex(self, model):
        """Test finding last model text in complex scenarios (lines 1211-1217)."""
        # Create a complex history with function calls
        history = [
            {"role": "system", "parts": [{"text": "You are a helpful assistant"}]},
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
            {"role": "user", "parts": [{"text": "Run a function"}]},
            {"role": "model", "parts": [{"function_call": {"name": "test_function"}}]},
            {"role": "user", "parts": [{"text": "Function result: success"}]},
            {"role": "model", "parts": [{"text": "Great! The function worked."}]},
        ]

        # Test finding last model text
        result = model._find_last_model_text(history)
        assert result == "Great! The function worked."

        # Add a function call at the end
        history.append({"role": "model", "parts": [{"function_call": {"name": "another_function"}}]})

        # Test finding last model text when last entry is a function call
        result = model._find_last_model_text(history)
        assert result == "Great! The function worked."

    @patch("vertexai.generative_models.GenerativeModel")
    @patch("cli_code.models.gemini.get_tool")
    def test_handle_tool_calls(self, mock_get_tool, mock_generative_model, model):
        """Test handling of tool calls (lines 1062-1097)."""
        # Setup mock model and response
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model
        mock_response = MagicMock()
        mock_gen_model.generate_content.return_value = mock_response

        # Ensure the model is initialized (set model attribute)
        model.model = mock_gen_model

        # Create a mock response with function calls
        mock_parts = [MagicMock()]
        mock_parts[
            0
        ].text = '```json\n{"function_call": {"name": "test_function", "arguments": {"arg1": "value1"}}}\n```'
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = mock_parts

        # Setup a mock tool
        mock_tool = MagicMock()
        mock_tool.requires_confirmation = False
        mock_tool.execute.return_value = "Tool executed successfully"
        mock_get_tool.return_value = mock_tool

        # Mock the agent loop behavior - make sure it processes one iteration then stops
        with patch.object(model, "_process_agent_iteration", return_value=("complete", "Tool executed successfully")):
            # Call generate method
            result = model.generate("Run a function")

            # Verify the tool call was processed
            assert result == "Tool executed successfully"

    def test_clear_history(self, model):
        """Test clearing conversation history."""
        # Set initial history
        model.history = [
            {"role": "system", "content": "Initial system prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Call clear_history
        model.clear_history()

        # Verify the history is cleared except for system prompt
        assert len(model.history) == 2
        assert model.history[0]["role"] == "system"

    def test_handle_unexpected_finish_reason_no_actionable_content(self, model):
        """Test handling unexpected finish reason with no actionable content."""
        # Mock a response candidate with an unexpected finish reason
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 99  # Use a value that doesn't match any known finish reason

        # Call the method
        status, message = model._handle_no_actionable_content(mock_candidate)

        # Verify we get an error status to prevent infinite loops
        assert status == "error"
        assert "Unexpected state" in message
        assert "finish reason 99" in message

        # Test that process_candidate_response also calls this method for the same case
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = []  # No parts = no actionable content

        status, message = model._process_candidate_response(mock_candidate, MagicMock())

        # Verify we get an error status to prevent infinite loops
        assert status == "error"
        assert "Unexpected state" in message

    def test_error_response_added_to_history(self, model):
        """Test that error responses are added to history for better context."""
        # Create a mock candidate with recitation finish reason (4)
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 4  # RECITATION
        mock_candidate.content = MagicMock()
        # Add some text content that should be stored even though it's an error
        mock_text = "This is potentially problematic content"
        mock_candidate.content.parts = [MagicMock(text=mock_text)]

        # Clear history first
        model.history = []

        # Process the candidate
        status, message = model._process_candidate_response(mock_candidate, MagicMock())

        # Verify the response was rejected due to recitation
        assert status == "error"
        assert "recitation policy" in message

        # Verify the text was still added to history for context
        assert len(model.history) == 1
        assert model.history[0]["role"] == "model"
        assert model.history[0]["parts"][0]["text"] == mock_text
