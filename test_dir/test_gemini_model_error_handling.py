"""
Tests for the Gemini Model error handling scenarios.
"""
import json
from unittest.mock import patch, MagicMock

import pytest
from rich.console import Console

from cli_code.models.gemini import GeminiModel
from cli_code.tools import AVAILABLE_TOOLS


class TestGeminiModelErrorHandling:
    """Tests for error handling in GeminiModel."""

    @pytest.fixture
    def mock_generative_model(self):
        """Mock the Gemini generative model."""
        with patch("cli_code.models.gemini.generative_models.GenerativeModel") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def gemini_model(self, mock_generative_model):
        """Create a GeminiModel instance with mocked dependencies."""
        console = Console()
        with patch("cli_code.models.gemini.generative_models") as mock_gm:
            # Configure the mock
            mock_gm.GenerativeModel = MagicMock()
            mock_gm.GenerativeModel.return_value = mock_generative_model
            
            # Create the model
            model = GeminiModel(api_key="fake_api_key", console=console, model_name="gemini-pro")
            yield model

    @patch("cli_code.models.gemini.generative_models")
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
        assert "empty prompt" in result.lower()
        
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
        with patch("cli_code.models.gemini.generative_models") as mock_gm:
            mock_gm.GenerativeModel.side_effect = Exception("Model not found: nonexistent-model")
            
            # Attempt to create the model
            with pytest.raises(Exception) as excinfo:
                GeminiModel(api_key="fake_api_key", console=console, model_name="nonexistent-model")
            
            # Verify the error message
            assert "model not found" in str(excinfo.value).lower()

    @patch("cli_code.models.gemini.get_tool")
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
        assert "error" in result.lower()
        assert "tool execution error" in result.lower()

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
        # First mock getting a real tool from AVAILABLE_TOOLS
        test_tool = None
        for tool in AVAILABLE_TOOLS:
            if tool.required_args:  # Find a tool with required args
                test_tool = tool
                break
        
        if not test_tool:
            pytest.skip("No tools with required arguments found for testing")
        
        # Configure the mock to return a response with a function call missing required args
        mock_response = MagicMock()
        mock_parts = [MagicMock()]
        mock_parts[0].text = None  # No text
        mock_parts[0].function_call = MagicMock()
        mock_parts[0].function_call.name = test_tool.name
        mock_parts[0].function_call.args = {}  # Empty args, missing required ones
        
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = mock_parts
        
        mock_generative_model.generate_content.return_value = mock_response
        
        # Patch the get_tool function to return our test tool
        with patch("cli_code.models.gemini.get_tool") as mock_get_tool:
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