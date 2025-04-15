"""
Tests for the Gemini Model error handling scenarios.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Import the actual exception class
from google.api_core.exceptions import InvalidArgument, ResourceExhausted

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

# Ensure FALLBACK_MODEL is imported
from src.cli_code.models.gemini import FALLBACK_MODEL, GeminiModel
from src.cli_code.tools import AVAILABLE_TOOLS
from src.cli_code.tools.base import BaseTool


class TestGeminiModelErrorHandling:
    """Tests for error handling in GeminiModel."""

    @pytest.fixture
    def mock_generative_model(self):
        """Mock the Gemini generative model."""
        with patch("src.cli_code.models.gemini.genai.GenerativeModel") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def gemini_model(self, mock_generative_model):
        """Create a GeminiModel instance with mocked dependencies."""
        console = Console()
        with patch("src.cli_code.models.gemini.genai") as mock_gm:
            # Configure the mock
            mock_gm.GenerativeModel = MagicMock()
            mock_gm.GenerativeModel.return_value = mock_generative_model

            # Create the model
            model = GeminiModel(api_key="fake_api_key", console=console, model_name="gemini-pro")
            yield model

    @patch("src.cli_code.models.gemini.genai")
    def test_initialization_error(self, mock_gm):
        """Test error handling during initialization."""
        # Make the GenerativeModel constructor raise an exception
        mock_gm.GenerativeModel.side_effect = Exception("API initialization error")

        # Create a console for the model
        console = Console()

        # Attempt to create the model - should raise an error
        with pytest.raises(Exception) as excinfo:
            GeminiModel(api_key="fake_api_key", console=console, model_name="gemini-pro")

        # Verify the error message
        assert "API initialization error" in str(excinfo.value)

    def test_empty_prompt_error(self, gemini_model, mock_generative_model):
        """Test error handling when an empty prompt is provided."""
        # Call generate with an empty prompt
        result = gemini_model.generate("")

        # Verify error message is returned
        assert result is not None
        assert result == "Error: Cannot process empty prompt. Please provide a valid input."

        # Verify that no API call was made
        mock_generative_model.generate_content.assert_not_called()

    def test_api_error_handling(self, gemini_model, mock_generative_model):
        """Test handling of API errors during generation."""
        # Make the API call raise an exception
        mock_generative_model.generate_content.side_effect = Exception("API error")

        # Call generate
        result = gemini_model.generate("Test prompt")

        # Verify error message is returned
        assert result is not None
        assert "error" in result.lower()
        assert "api error" in result.lower()

    def test_rate_limit_error_handling(self, gemini_model, mock_generative_model):
        """Test handling of rate limit errors."""
        # Create a rate limit error
        rate_limit_error = Exception("Rate limit exceeded")
        mock_generative_model.generate_content.side_effect = rate_limit_error

        # Call generate
        result = gemini_model.generate("Test prompt")

        # Verify rate limit error message is returned
        assert result is not None
        assert "rate limit" in result.lower() or "quota" in result.lower()

    def test_invalid_api_key_error(self, gemini_model, mock_generative_model):
        """Test handling of invalid API key errors."""
        # Create an authentication error
        auth_error = Exception("Invalid API key")
        mock_generative_model.generate_content.side_effect = auth_error

        # Call generate
        result = gemini_model.generate("Test prompt")

        # Verify authentication error message is returned
        assert result is not None
        assert "api key" in result.lower() or "authentication" in result.lower()

    def test_model_not_found_error(self, mock_generative_model):
        """Test handling of model not found errors."""
        # Create a console for the model
        console = Console()

        # Create the model with an invalid model name
        with patch("src.cli_code.models.gemini.genai") as mock_gm:
            mock_gm.GenerativeModel.side_effect = Exception("Model not found: nonexistent-model")

            # Attempt to create the model
            with pytest.raises(Exception) as excinfo:
                GeminiModel(api_key="fake_api_key", console=console, model_name="nonexistent-model")

            # Verify the error message
            assert "model not found" in str(excinfo.value).lower()

    @patch("src.cli_code.models.gemini.get_tool")
    def test_tool_execution_error(self, mock_get_tool, gemini_model, mock_generative_model):
        """Test handling of errors during tool execution."""
        # Configure the mock to return a response with a function call
        mock_response = MagicMock()
        mock_parts = [MagicMock()]
        mock_parts[0].text = None  # No text
        mock_parts[0].function_call = MagicMock()
        mock_parts[0].function_call.name = "test_tool"
        mock_parts[0].function_call.args = {"arg1": "value1"}

        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = mock_parts

        mock_generative_model.generate_content.return_value = mock_response

        # Make the tool execution raise an error
        mock_tool = MagicMock()
        mock_tool.execute.side_effect = Exception("Tool execution error")
        mock_get_tool.return_value = mock_tool

        # Call generate
        result = gemini_model.generate("Use the test_tool")

        # Verify tool error is handled and included in the response
        assert result is not None
        assert result == "Error: Tool execution error with test_tool: Tool execution error"

    def test_invalid_function_call_format(self, gemini_model, mock_generative_model):
        """Test handling of invalid function call format."""
        # Configure the mock to return a response with an invalid function call
        mock_response = MagicMock()
        mock_parts = [MagicMock()]
        mock_parts[0].text = None  # No text
        mock_parts[0].function_call = MagicMock()
        mock_parts[0].function_call.name = "nonexistent_tool"  # Tool doesn't exist
        mock_parts[0].function_call.args = {"arg1": "value1"}

        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = mock_parts

        mock_generative_model.generate_content.return_value = mock_response

        # Call generate
        result = gemini_model.generate("Use a tool")

        # Verify invalid tool error is handled
        assert result is not None
        assert "tool not found" in result.lower() or "nonexistent_tool" in result.lower()

    def test_missing_required_args(self, gemini_model, mock_generative_model):
        """Test handling of function calls with missing required arguments."""
        # Create a mock test tool with required arguments
        test_tool = MagicMock()
        test_tool.name = "test_tool"
        test_tool.execute = MagicMock(side_effect=ValueError("Missing required argument 'required_param'"))

        # Configure the mock to return a response with a function call missing required args
        mock_response = MagicMock()
        mock_parts = [MagicMock()]
        mock_parts[0].text = None  # No text
        mock_parts[0].function_call = MagicMock()
        mock_parts[0].function_call.name = "test_tool"
        mock_parts[0].function_call.args = {}  # Empty args, missing required ones

        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = mock_parts

        mock_generative_model.generate_content.return_value = mock_response

        # Patch the get_tool function to return our test tool
        with patch("src.cli_code.models.gemini.get_tool") as mock_get_tool:
            mock_get_tool.return_value = test_tool

            # Call generate
            result = gemini_model.generate("Use a tool")

            # Verify missing args error is handled
            assert result is not None
            assert "missing" in result.lower() or "required" in result.lower() or "argument" in result.lower()

    def test_handling_empty_response(self, gemini_model, mock_generative_model):
        """Test handling of empty response from the API."""
        # Configure the mock to return an empty response
        mock_response = MagicMock()
        mock_response.candidates = []  # No candidates

        mock_generative_model.generate_content.return_value = mock_response

        # Call generate
        result = gemini_model.generate("Test prompt")

        # Verify empty response is handled
        assert result is not None
        assert "empty response" in result.lower() or "no response" in result.lower()

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
    def mock_genai(self):
        genai = MagicMock()
        genai.GenerativeModel = MagicMock()
        return genai

    def test_init_without_api_key(self, mock_console):
        """Test initialization when API key is not provided."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            # Execute and expect the ValueError
            with pytest.raises(ValueError, match="Gemini API key is required"):
                model = GeminiModel(None, mock_console)

    def test_init_with_invalid_api_key(self, mock_console):
        """Test initialization with an invalid API key."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            with patch("src.cli_code.models.gemini.genai") as mock_genai:
                mock_genai.configure.side_effect = ImportError("No module named 'google.generativeai'")

                # Should raise ConnectionError
                with pytest.raises(ConnectionError):
                    model = GeminiModel("invalid_key", mock_console)

    @patch("src.cli_code.models.gemini.genai")
    def test_generate_without_client(self, mock_genai, mock_console):
        """Test generate method when the client is not initialized."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            # Create model that will have model=None
            model = GeminiModel("valid_key", mock_console)
            # Manually set model to None to simulate uninitialized client
            model.model = None

            # Execute
            result = model.generate("test prompt")

            # Assert
            assert "Error" in result and "not initialized" in result

    @patch("src.cli_code.models.gemini.genai")
    def test_generate_with_api_error(self, mock_genai, mock_console):
        """Test generate method when the API call fails."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            # Create a model with a mock
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock to raise an exception
            mock_model = MagicMock()
            model.model = mock_model
            mock_model.generate_content.side_effect = Exception("API Error")

            # Execute
            result = model.generate("test prompt")

            # Assert error during agent processing appears
            assert "Error during agent processing" in result

    @patch("src.cli_code.models.gemini.genai")
    def test_generate_with_safety_block(self, mock_genai, mock_console):
        """Test generate method when content is blocked by safety filters."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Mock the model
            mock_model = MagicMock()
            model.model = mock_model

            # Configure the mock to return a blocked response
            mock_response = MagicMock()
            mock_response.prompt_feedback = MagicMock()
            mock_response.prompt_feedback.block_reason = "SAFETY"
            mock_response.candidates = []
            mock_model.generate_content.return_value = mock_response

            # Execute
            result = model.generate("test prompt")

            # Assert
            assert "Empty response" in result or "no candidates" in result.lower()

    @patch("src.cli_code.models.gemini.genai")
    @patch("src.cli_code.models.gemini.get_tool")
    @patch("src.cli_code.models.gemini.json.loads")
    def test_generate_with_invalid_tool_call(self, mock_json_loads, mock_get_tool, mock_genai, mock_console):
        """Test generate method with invalid JSON in tool arguments."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Create a mock response with tool calls
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "test_tool"
            mock_part.function_call.args = "invalid_json"
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response

            # Make JSON decoding fail
            mock_json_loads.side_effect = json.JSONDecodeError("Expecting value", "", 0)

            # Execute
            result = model.generate("test prompt")

            # Assert
            assert "Error" in result

    @patch("src.cli_code.models.gemini.genai")
    @patch("src.cli_code.models.gemini.get_tool")
    def test_generate_with_missing_required_tool_args(self, mock_get_tool, mock_genai, mock_console):
        """Test generate method when required tool arguments are missing."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Create a mock response with tool calls
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "test_tool"
            mock_part.function_call.args = {}  # Empty args dict
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response

            # Mock the tool to have required params
            tool_mock = MagicMock()
            tool_declaration = MagicMock()
            tool_declaration.parameters = {"required": ["required_param"]}
            tool_mock.get_function_declaration.return_value = tool_declaration
            mock_get_tool.return_value = tool_mock

            # Execute
            result = model.generate("test prompt")

            # We should get to the max iterations with the tool response
            assert "max iterations" in result.lower()

    @patch("src.cli_code.models.gemini.genai")
    def test_generate_with_tool_not_found(self, mock_genai, mock_console):
        """Test generate method when a requested tool is not found."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Create a mock response with tool calls
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "nonexistent_tool"
            mock_part.function_call.args = {}
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response

            # Mock get_tool to return None for nonexistent tool
            with patch("src.cli_code.models.gemini.get_tool", return_value=None):
                # Execute
                result = model.generate("test prompt")

            # We should mention the tool not found
            assert "not found" in result.lower() or "not available" in result.lower()

    @patch("src.cli_code.models.gemini.genai")
    @patch("src.cli_code.models.gemini.get_tool")
    def test_generate_with_tool_execution_error(self, mock_get_tool, mock_genai, mock_console):
        """Test generate method when a tool execution raises an error."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Create a mock response with tool calls
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "test_tool"
            mock_part.function_call.args = {}
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response

            # Mock the tool to raise an exception
            tool_mock = MagicMock()
            tool_mock.execute.side_effect = Exception("Tool execution error")
            mock_get_tool.return_value = tool_mock

            # Execute
            result = model.generate("test prompt")

            # Assert
            assert "error" in result.lower() and "tool" in result.lower()

    @patch("src.cli_code.models.gemini.genai")
    def test_list_models_error(self, mock_genai, mock_console):
        """Test list_models method when an error occurs."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock to raise an exception
            mock_genai.list_models.side_effect = Exception("List models error")

            # Execute
            result = model.list_models()

            # Assert
            assert result == []
            mock_console.print.assert_called()

    @patch("src.cli_code.models.gemini.genai")
    def test_generate_with_empty_response(self, mock_genai, mock_console):
        """Test generate method when the API returns an empty response."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Create a response with no candidates
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = []  # Empty candidates
            mock_model.generate_content.return_value = mock_response

            # Execute
            result = model.generate("test prompt")

            # Assert
            assert "no candidates" in result.lower()

    @patch("src.cli_code.models.gemini.genai")
    def test_generate_with_malformed_response(self, mock_genai, mock_console):
        """Test generate method when the API returns a malformed response."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console)

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Create a malformed response
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content = None  # Missing content
            mock_model.generate_content.return_value = mock_response

            # Execute
            result = model.generate("test prompt")

            # Assert
            assert "no content" in result.lower() or "no parts" in result.lower()

    @patch("src.cli_code.models.gemini.genai")
    @patch("src.cli_code.models.gemini.get_tool")
    @patch("src.cli_code.models.gemini.questionary")
    def test_generate_with_tool_confirmation_rejected(self, mock_questionary, mock_get_tool, mock_genai, mock_console):
        """Test generate method when user rejects sensitive tool confirmation."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console, "gemini-pro")  # Use the fixture?

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Mock the tool instance
            mock_tool = MagicMock()
            mock_get_tool.return_value = mock_tool

            # Mock the confirmation to return False (rejected)
            confirm_mock = MagicMock()
            confirm_mock.ask.return_value = False
            mock_questionary.confirm.return_value = confirm_mock

            # Create a mock response with a sensitive tool call (e.g., edit)
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "edit"  # Sensitive tool
            mock_part.function_call.args = {"file_path": "test.py", "content": "new content"}
            mock_response.candidates[0].content.parts = [mock_part]

            # First call returns the function call
            mock_model.generate_content.return_value = mock_response

            # Execute
            result = model.generate("Edit the file test.py")

            # Assertions
            mock_questionary.confirm.assert_called_once()  # Check confirm was called
            mock_tool.execute.assert_not_called()  # Tool should NOT be executed
            # The agent loop might continue or timeout, check for rejection message in history/result
            # Depending on loop continuation logic, it might hit max iterations or return the rejection text
            assert "rejected" in result.lower() or "maximum iterations" in result.lower()

    @patch("src.cli_code.models.gemini.genai")
    @patch("src.cli_code.models.gemini.get_tool")
    @patch("src.cli_code.models.gemini.questionary")
    def test_generate_with_tool_confirmation_cancelled(self, mock_questionary, mock_get_tool, mock_genai, mock_console):
        """Test generate method when user cancels sensitive tool confirmation."""
        # Setup
        with patch("src.cli_code.models.gemini.log"):
            model = GeminiModel("valid_key", mock_console, "gemini-pro")

            # Configure the mock model
            mock_model = MagicMock()
            model.model = mock_model

            # Mock the tool instance
            mock_tool = MagicMock()
            mock_get_tool.return_value = mock_tool

            # Mock the confirmation to return None (cancelled)
            confirm_mock = MagicMock()
            confirm_mock.ask.return_value = None
            mock_questionary.confirm.return_value = confirm_mock

            # Create a mock response with a sensitive tool call (e.g., edit)
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "edit"  # Sensitive tool
            mock_part.function_call.args = {"file_path": "test.py", "content": "new content"}
            mock_response.candidates[0].content.parts = [mock_part]

            mock_model.generate_content.return_value = mock_response

            # Execute
            result = model.generate("Edit the file test.py")

            # Assertions
            mock_questionary.confirm.assert_called_once()  # Check confirm was called
            mock_tool.execute.assert_not_called()  # Tool should NOT be executed
            assert "cancelled confirmation" in result.lower()
            assert "edit on test.py" in result.lower()


# --- Standalone Test for Quota Fallback ---
@pytest.mark.skip(reason="This test needs to be rewritten with proper mocking of the Gemini API integration path")
def test_generate_with_quota_error_and_fallback_returns_success():
    """Test that GeminiModel falls back to the fallback model on quota error and returns success."""
    with (
        patch("src.cli_code.models.gemini.Console") as mock_console_cls,
        patch("src.cli_code.models.gemini.genai") as mock_genai,
        patch("src.cli_code.models.gemini.GeminiModel._initialize_model_instance") as mock_init_model,
        patch("src.cli_code.models.gemini.AVAILABLE_TOOLS", {}) as mock_available_tools,
        patch("src.cli_code.models.gemini.log") as mock_log,
    ):
        # Arrange
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        # Mocks for the primary and fallback model behaviors
        mock_primary_model_instance = MagicMock(name="PrimaryModelInstance")
        mock_fallback_model_instance = MagicMock(name="FallbackModelInstance")

        # Configure Mock genai module with ResourceExhausted exception
        mock_genai.GenerativeModel.return_value = mock_primary_model_instance
        mock_genai.api_core.exceptions.ResourceExhausted = ResourceExhausted

        # Configure the generate_content behavior for the primary mock to raise the ResourceExhausted exception
        mock_primary_model_instance.generate_content.side_effect = ResourceExhausted("Quota exhausted")

        # Configure the generate_content behavior for the fallback mock
        mock_fallback_response = MagicMock()
        mock_fallback_candidate = MagicMock()
        mock_fallback_part = MagicMock()
        mock_fallback_part.text = "Fallback successful"
        mock_fallback_candidate.content = MagicMock()
        mock_fallback_candidate.content.parts = [mock_fallback_part]
        mock_fallback_response.candidates = [mock_fallback_candidate]
        mock_fallback_model_instance.generate_content.return_value = mock_fallback_response

        # Define the side effect for the _initialize_model_instance method
        def init_side_effect(*args, **kwargs):
            # After the quota error, replace the model with the fallback model
            if mock_init_model.call_count > 1:
                # Replace the model that will be returned by GenerativeModel
                mock_genai.GenerativeModel.return_value = mock_fallback_model_instance
                return None
            return None

        mock_init_model.side_effect = init_side_effect

        # Setup the GeminiModel instance
        gemini_model = GeminiModel(api_key="fake_key", model_name="gemini-1.5-pro-latest", console=mock_console)

        # Create an empty history to allow test to run properly
        gemini_model.history = [{"role": "user", "parts": [{"text": "test prompt"}]}]

        # Act
        response = gemini_model.generate("test prompt")

        # Assert
        # Check that warning and info logs were called
        mock_log.warning.assert_any_call("Quota exceeded for model 'gemini-1.5-pro-latest': 429 Quota exhausted")
        mock_log.info.assert_any_call("Switching to fallback model: gemini-1.0-pro")

        # Check initialization was called twice
        assert mock_init_model.call_count >= 2

        # Check that generate_content was called
        assert mock_primary_model_instance.generate_content.call_count >= 1
        assert mock_fallback_model_instance.generate_content.call_count >= 1

        # Check final response
        assert response == "Fallback successful"


# ... (End of file or other tests) ...
