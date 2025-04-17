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
        # Setup mock
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Make generate_content raise AttributeError
        mock_gen_model.generate_content.side_effect = AttributeError("No 'text' attribute")

        # Call generate method
        result = model.generate("Test prompt")

        # Verify error handling
        assert "An error occurred" in result
        model.console.print.assert_called()

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_quota_exceeded(self, mock_generative_model, model):
        """Test handling of quota exceeded errors (lines 362-374)."""
        # Setup mock
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Make generate_content raise an exception with quota exceeded message
        error_message = "Quota exceeded for quota metric 'CharactersPerMinutePerProject'"
        mock_gen_model.generate_content.side_effect = Exception(error_message)

        # Call generate method
        result = model.generate("Test prompt")

        # Verify quota error handling
        assert "Quota exceeded" in result
        model.console.print.assert_called()

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_rate_limit(self, mock_generative_model, model):
        """Test handling of rate limit errors (lines 380-384)."""
        # Setup mock
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Make generate_content raise an exception with rate limit message
        error_message = "Resource has been exhausted (e.g. check quota)."
        mock_gen_model.generate_content.side_effect = Exception(error_message)

        # Call generate method
        result = model.generate("Test prompt")

        # Verify rate limit error handling
        assert "Rate limit" in result
        model.console.print.assert_called()

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_server_error(self, mock_generative_model, model):
        """Test handling of server errors (lines 390-393)."""
        # Setup mock
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Make generate_content raise an exception with server error message
        error_message = "Server error (5"  # Partial match for 5xx errors
        mock_gen_model.generate_content.side_effect = Exception(error_message)

        # Call generate method
        result = model.generate("Test prompt")

        # Verify server error handling
        assert "server error" in result.lower()
        model.console.print.assert_called()

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_safety_error(self, mock_generative_model, model):
        """Test handling of safety errors (lines 403-406)."""
        # Setup mock
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Make generate_content raise an exception with safety error message
        error_message = "Content filtered due to safety settings"
        mock_gen_model.generate_content.side_effect = Exception(error_message)

        # Call generate method
        result = model.generate("Test prompt")

        # Verify safety error handling
        assert "safety" in result.lower()
        model.console.print.assert_called()

    @patch("vertexai.generative_models.GenerativeModel")
    def test_error_handling_context_length(self, mock_generative_model, model):
        """Test handling of context length errors (lines 414-428)."""
        # Setup mock
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model

        # Make generate_content raise an exception with context length error
        error_message = "input is too long"
        mock_gen_model.generate_content.side_effect = Exception(error_message)

        # Call generate method
        result = model.generate("Test prompt")

        # Verify context length error handling
        assert "context length" in result.lower()
        model.console.print.assert_called()

    def test_manage_context_window(self, model):
        """Test context window management (lines 1124-1137)."""
        # Create a history with multiple messages
        model.history = [
            {"role": "system", "content": "You are a helpful assistant", "tokens": 10},
            {"role": "user", "content": "Hello", "tokens": 1},
            {"role": "assistant", "content": "Hi there!", "tokens": 2},
            {"role": "user", "content": "How are you?", "tokens": 3},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!", "tokens": 9},
        ]

        # Call _manage_context_window with a limit that should remove some entries
        model._manage_context_window(max_tokens=15)

        # Verify that the history has been trimmed but system message is preserved
        assert len(model.history) < 5
        assert model.history[0]["role"] == "system"
        assert model.history[0]["content"] == "You are a helpful assistant"

        # Test with a limit that should clear all but system message
        model.history = [
            {"role": "system", "content": "You are a helpful assistant", "tokens": 10},
            {"role": "user", "content": "Hello", "tokens": 1},
        ]
        model._manage_context_window(max_tokens=5)
        assert len(model.history) == 1
        assert model.history[0]["role"] == "system"

    def test_find_last_model_text_complex(self, model):
        """Test finding last model text in complex scenarios (lines 1211-1217)."""
        # Create a complex history with function calls
        model.history = [
            {"role": "system", "content": "You are a helpful assistant", "tokens": 10},
            {"role": "user", "content": "Hello", "tokens": 1},
            {"role": "assistant", "content": "Hi there!", "tokens": 2},
            {"role": "user", "content": "Run a function", "tokens": 3},
            {
                "role": "assistant",
                "content": '```json\n{"function_call": {"name": "test_function"}}\n```',
                "tokens": 10,
            },
            {"role": "user", "content": "Function result: success", "tokens": 4},
            {"role": "assistant", "content": "Great! The function worked.", "tokens": 5},
        ]

        # Test finding last model text
        result = model._find_last_model_text()
        assert result == "Great! The function worked."

        # Add a function call at the end
        model.history.append(
            {
                "role": "assistant",
                "content": '```json\n{"function_call": {"name": "another_function"}}\n```',
                "tokens": 10,
            }
        )

        # Test finding last model text when last entry is a function call
        result = model._find_last_model_text()
        assert result == "Great! The function worked."

    @patch("vertexai.generative_models.GenerativeModel")
    def test_handle_tool_calls(self, mock_generative_model, model):
        """Test handling of tool calls (lines 1062-1097)."""
        # Setup mock
        mock_gen_model = MagicMock()
        mock_generative_model.return_value = mock_gen_model
        mock_response = MagicMock()
        mock_gen_model.generate_content.return_value = mock_response

        # Create a mock response with function calls
        function_json = {"function_call": {"name": "test_function", "arguments": {"arg1": "value1", "arg2": "value2"}}}
        mock_response.text = f"```json\n{json.dumps(function_json)}\n```"

        # Setup a mock tool registry
        model.tool_registry = MagicMock()
        model.tool_registry.get_tool.return_value = MagicMock()
        model.tool_registry.get_tool.return_value.execute.return_value = "Tool executed successfully"

        # Mock _request_tool_confirmation to always return True
        with patch.object(model, "_request_tool_confirmation", return_value=True):
            # Call generate method
            result = model.generate("Run a function")

            # Verify tool execution flow
            model.tool_registry.get_tool.assert_called_with("test_function")
            assert "Tool executed successfully" in str(model.history)

    def test_update_system_prompt(self, model):
        """Test updating system prompt (lines 1257-1258)."""
        # Set initial system prompt
        model.add_to_history("system", "Initial system prompt")
        assert model.history[0]["role"] == "system"
        assert model.history[0]["content"] == "Initial system prompt"

        # Update system prompt
        model.update_system_prompt("Updated system prompt")

        # Verify system prompt was updated
        assert len(model.history) == 1
        assert model.history[0]["role"] == "system"
        assert model.history[0]["content"] == "Updated system prompt"

        # Add a user message then update system prompt again
        model.add_to_history("user", "Hello")
        model.update_system_prompt("Second update to system prompt")

        # Verify system prompt was updated and user message preserved
        assert len(model.history) == 2
        assert model.history[0]["role"] == "system"
        assert model.history[0]["content"] == "Second update to system prompt"
        assert model.history[1]["role"] == "user"
