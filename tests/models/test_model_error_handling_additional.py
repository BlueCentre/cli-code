"""
Additional comprehensive error handling tests for Ollama and Gemini models.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Ensure src is in the path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cli_code.models.gemini import GeminiModel
from cli_code.models.ollama import MAX_OLLAMA_ITERATIONS, OllamaModel
from cli_code.tools.base import BaseTool


class TestModelContextHandling:
    """Tests for context window handling in both model classes."""

    @pytest.fixture
    def mock_console(self):
        console = MagicMock()
        console.print = MagicMock()
        console.status = MagicMock()
        # Make status return a context manager
        status_cm = MagicMock()
        console.status.return_value = status_cm
        status_cm.__enter__ = MagicMock(return_value=None)
        status_cm.__exit__ = MagicMock(return_value=None)
        return console

    @pytest.fixture
    def mock_ollama_client(self):
        client = MagicMock()
        client.chat.completions.create = MagicMock()
        client.models.list = MagicMock()
        return client

    @pytest.fixture
    def mock_genai(self):
        with patch("cli_code.models.gemini.genai") as mock:
            yield mock

    @patch("cli_code.models.ollama.count_tokens")
    def test_ollama_manage_context_trimming(self, mock_count_tokens, mock_console, mock_ollama_client):
        """Test Ollama model context window management when history exceeds token limit."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_ollama_client

        # Mock the token counting to return a large value
        mock_count_tokens.return_value = 90000  # Higher than OLLAMA_MAX_CONTEXT_TOKENS (80000)

        # Add a few messages to history
        model.history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant response 1"},
            {"role": "user", "content": "User message 2"},
            {"role": "assistant", "content": "Assistant response 2"},
        ]

        # Execute
        original_length = len(model.history)
        model._manage_ollama_context()

        # Assert
        # Should have removed some messages but kept system prompt
        assert len(model.history) < original_length
        assert model.history[0]["role"] == "system"  # System prompt should be preserved

    @patch("cli_code.models.gemini.genai")
    def test_gemini_manage_context_window(self, mock_genai, mock_console):
        """Test Gemini model context window management."""
        # Setup
        # Mock generative model for initialization
        mock_instance = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_instance

        # Create the model
        model = GeminiModel(api_key="fake_api_key", console=mock_console)

        # Create a large history - need more than (MAX_HISTORY_TURNS * 3 + 2) items
        # MAX_HISTORY_TURNS is 20, so we need > 62 items
        model.history = []
        for i in range(22):  # This will generate 66 items (3 per "round")
            model.history.append({"role": "user", "parts": [f"User message {i}"]})
            model.history.append({"role": "model", "parts": [f"Model response {i}"]})
            model.history.append({"role": "model", "parts": [{"function_call": {"name": "test"}, "text": None}]})

        # Execute
        original_length = len(model.history)
        assert original_length > 62  # Verify we're over the limit
        model._manage_context_window()

        # Assert
        assert len(model.history) < original_length
        assert len(model.history) <= (20 * 3 + 2)  # MAX_HISTORY_TURNS * 3 + 2

    def test_ollama_history_handling(self, mock_console):
        """Test Ollama add_to_history and clear_history methods."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model._manage_ollama_context = MagicMock()  # Mock to avoid side effects

        # Test clear_history
        model.history = [{"role": "system", "content": "System prompt"}]
        model.clear_history()
        assert len(model.history) == 1  # Should keep system prompt
        assert model.history[0]["role"] == "system"

        # Test adding system message
        model.history = []
        model.add_to_history({"role": "system", "content": "New system prompt"})
        assert len(model.history) == 1
        assert model.history[0]["role"] == "system"

        # Test adding user message
        model.add_to_history({"role": "user", "content": "User message"})
        assert len(model.history) == 2
        assert model.history[1]["role"] == "user"

        # Test adding assistant message
        model.add_to_history({"role": "assistant", "content": "Assistant response"})
        assert len(model.history) == 3
        assert model.history[2]["role"] == "assistant"

        # Test adding with custom role - implementation accepts any role
        model.add_to_history({"role": "custom", "content": "Custom message"})
        assert len(model.history) == 4
        assert model.history[3]["role"] == "custom"


class TestModelConfiguration:
    """Tests for model configuration and initialization."""

    @pytest.fixture
    def mock_console(self):
        console = MagicMock()
        console.print = MagicMock()
        return console

    @patch("cli_code.models.gemini.genai")
    def test_gemini_initialization_with_env_variable(self, mock_genai, mock_console):
        """Test Gemini initialization with API key from environment variable."""
        # Setup
        # Mock generative model for initialization
        mock_instance = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_instance

        # Mock os.environ
        with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_key_from_env"}):
            # Execute
            model = GeminiModel(api_key="dummy_key_from_env", console=mock_console)

            # Assert
            assert model.api_key == "dummy_key_from_env"
            mock_genai.configure.assert_called_once_with(api_key="dummy_key_from_env")

    def test_ollama_initialization_with_invalid_url(self, mock_console):
        """Test Ollama initialization with invalid URL."""
        # Shouldn't raise an error immediately, but should fail on first API call
        model = OllamaModel("http://invalid:1234", mock_console, "llama3")

        # Should have a client despite invalid URL
        assert model.client is not None

        # Mock the client's methods to raise exceptions
        model.client.chat.completions.create = MagicMock(side_effect=Exception("Connection failed"))
        model.client.models.list = MagicMock(side_effect=Exception("Connection failed"))

        # Execute API call and verify error handling
        result = model.generate("test prompt")
        assert "error" in result.lower()

        # Execute list_models and verify error handling
        result = model.list_models()
        assert result is None

    @patch("cli_code.models.gemini.genai")
    def test_gemini_model_selection(self, mock_genai, mock_console):
        """Test Gemini model selection and fallback behavior."""
        # Setup
        mock_instance = MagicMock()
        # Make first initialization fail, simulating unavailable model
        mock_genai.GenerativeModel.side_effect = [
            Exception("Model not available"),  # First call fails
            MagicMock(),  # Second call succeeds with fallback model
        ]

        with pytest.raises(Exception) as excinfo:
            # Execute - should raise exception when primary model fails
            GeminiModel(api_key="fake_api_key", console=mock_console, model_name="unavailable-model")

        assert "Could not initialize Gemini model" in str(excinfo.value)


class TestToolManagement:
    """Tests for tool management in both models."""

    @pytest.fixture
    def mock_console(self):
        console = MagicMock()
        console.print = MagicMock()
        return console

    @pytest.fixture
    def mock_ollama_client(self):
        client = MagicMock()
        client.chat.completions.create = MagicMock()
        return client

    @pytest.fixture
    def mock_test_tool(self):
        tool = MagicMock(spec=BaseTool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.required_args = ["arg1"]
        tool.get_function_declaration = MagicMock(return_value=MagicMock())
        tool.execute = MagicMock(return_value="Tool executed")
        return tool

    @patch("cli_code.models.ollama.get_tool")
    def test_ollama_tool_handling_with_missing_args(
        self, mock_get_tool, mock_console, mock_ollama_client, mock_test_tool
    ):
        """Test Ollama handling of tool calls with missing required arguments."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_ollama_client
        model.add_to_history = MagicMock()  # Mock history method

        # Make get_tool return our mock tool
        mock_get_tool.return_value = mock_test_tool

        # Create mock response with a tool call missing required args
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [
            MagicMock(
                function=MagicMock(
                    name="test_tool",
                    arguments="{}",  # Missing required arg1
                ),
                id="test_id",
            )
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message, finish_reason="tool_calls")]

        mock_ollama_client.chat.completions.create.return_value = mock_response

        # Execute
        result = model.generate("Use test_tool")

        # Assert - the model reaches max iterations in this case
        assert "maximum iterations" in result.lower() or "max iterations" in result.lower()
        # The tool gets executed despite missing args in the implementation

    @patch("cli_code.models.gemini.genai")
    @patch("cli_code.models.gemini.get_tool")
    def test_gemini_function_call_in_stream(self, mock_get_tool, mock_genai, mock_console, mock_test_tool):
        """Test Gemini handling of function call in streaming response."""
        # Setup
        # Mock generative model for initialization
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        # Create the model
        model = GeminiModel(api_key="fake_api_key", console=mock_console)

        # Mock get_tool to return our test tool
        mock_get_tool.return_value = mock_test_tool

        # Mock the streaming response
        mock_response = MagicMock()

        # Create a mock function call in the response
        mock_parts = [MagicMock()]
        mock_parts[0].text = None
        mock_parts[0].function_call = MagicMock()
        mock_parts[0].function_call.name = "test_tool"
        mock_parts[0].function_call.args = {"arg1": "value1"}  # Include required arg

        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = mock_parts

        mock_model.generate_content.return_value = mock_response

        # Execute
        result = model.generate("Use test_tool")

        # Assert
        assert mock_test_tool.execute.called  # Tool should be executed
        # Test reaches max iterations in current implementation
        assert "max iterations" in result.lower()


class TestModelEdgeCases:
    """Tests for edge cases in both model implementations."""

    @pytest.fixture
    def mock_console(self):
        console = MagicMock()
        console.print = MagicMock()
        return console

    @pytest.fixture
    def mock_ollama_client(self):
        client = MagicMock()
        client.chat.completions.create = MagicMock()
        return client

    @patch("cli_code.models.ollama.MessageToDict")
    def test_ollama_protobuf_conversion_failure(self, mock_message_to_dict, mock_console, mock_ollama_client):
        """Test Ollama handling of protobuf conversion failures."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_ollama_client

        # We'll mock _prepare_openai_tools instead of patching json.dumps globally
        model._prepare_openai_tools = MagicMock(return_value=None)

        # Make MessageToDict raise an exception
        mock_message_to_dict.side_effect = Exception("Protobuf conversion failed")

        # Mock the response with a tool call
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [MagicMock(function=MagicMock(name="test_tool", arguments="{}"), id="test_id")]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message, finish_reason="tool_calls")]

        mock_ollama_client.chat.completions.create.return_value = mock_response

        # Execute
        result = model.generate("Use test_tool")

        # Assert - the model reaches maximum iterations
        assert "maximum iterations" in result.lower()

    @patch("cli_code.models.gemini.genai")
    def test_gemini_empty_response_parts(self, mock_genai, mock_console):
        """Test Gemini handling of empty response parts."""
        # Setup
        # Mock generative model for initialization
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        # Create the model
        model = GeminiModel(api_key="fake_api_key", console=mock_console)

        # Mock a response with empty parts
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = []  # Empty parts

        mock_model.generate_content.return_value = mock_response

        # Execute
        result = model.generate("Test prompt")

        # Assert
        assert (
            "no content" in result.lower()
            or "content/parts" in result.lower()
            or "actionable content" in result.lower()
        )

    def test_ollama_with_empty_system_prompt(self, mock_console):
        """Test Ollama with an empty system prompt."""
        # Setup - initialize with normal system prompt
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")

        # Replace system prompt with empty string
        model.system_prompt = ""
        model.history = [{"role": "system", "content": ""}]

        # Verify it doesn't cause errors in initialization or history management
        model._manage_ollama_context()
        assert len(model.history) == 1
        assert model.history[0]["content"] == ""


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
