"""
Tests for the Gemini Model error handling scenarios.
"""
import pytest
import json
from unittest.mock import MagicMock, patch, call
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from cli_code.models.gemini import GeminiModel
from cli_code.tools.base import BaseTool
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
        with patch('cli_code.models.gemini.log'):
            # Execute
            model = GeminiModel(None, mock_console)
            
            # Assert
            assert model.api_key is None
            assert model.genai_client is None
    
    def test_init_with_invalid_api_key(self, mock_console):
        """Test initialization with an invalid API key."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            with patch('cli_code.models.gemini.genai', side_effect=ImportError("No module named 'google.generativeai'")):
                # Execute
                model = GeminiModel("invalid_key", mock_console)
                
                # Assert
                assert model.api_key == "invalid_key"
                assert model.genai_client is None
    
    def test_generate_without_client(self, mock_console):
        """Test generate method when the client is not initialized."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel(None, mock_console)
            
            # Execute
            result = model.generate("test prompt")
            
            # Assert
            assert "Error: Gemini client not initialized." in result
            mock_console.print.assert_called_once()
    
    @patch('cli_code.models.gemini.genai')
    def test_generate_with_api_error(self, mock_genai, mock_console):
        """Test generate method when the API call fails."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to raise an exception
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_model.generate_content.side_effect = Exception("API Error")
            
            # Execute
            result = model.generate("test prompt")
            
            # Assert
            assert "Error communicating with Gemini API" in result
            assert "API Error" in result
    
    @patch('cli_code.models.gemini.genai')
    def test_generate_with_safety_block(self, mock_genai, mock_console):
        """Test generate method when content is blocked by safety filters."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to return a blocked response
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.prompt_feedback.block_reason = "SAFETY"
            mock_response.parts = []
            mock_model.generate_content.return_value = mock_response
            
            # Execute
            result = model.generate("test prompt")
            
            # Assert
            assert "Content blocked by safety filters" in result
    
    @patch('cli_code.models.gemini.genai')
    @patch('cli_code.models.gemini.get_tool')
    def test_generate_with_invalid_tool_call(self, mock_get_tool, mock_genai, mock_console):
        """Test generate method with invalid JSON in tool arguments."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to return a response with tool calls
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "test_tool"
            mock_part.function_call.args = "invalid_json"
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response
            
            # Execute
            with patch('cli_code.models.gemini.json.loads', side_effect=json.JSONDecodeError("Expecting value", "", 0)):
                result = model.generate("test prompt")
            
            # Assert
            assert "Error formatting Gemini response" in result
    
    @patch('cli_code.models.gemini.genai')
    @patch('cli_code.models.gemini.get_tool')
    def test_generate_with_missing_required_tool_args(self, mock_get_tool, mock_genai, mock_console):
        """Test generate method when required tool arguments are missing."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to return a response with tool calls
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "test_tool"
            mock_part.function_call.args = '{"optional_param": "value"}'
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response
            
            # Mock the tool to have required params
            tool_mock = MagicMock()
            mock_get_tool.return_value = tool_mock
            tool_declaration = MagicMock()
            tool_declaration.parameters = {"required": ["required_param"]}
            tool_mock.get_function_declaration.return_value = tool_declaration
            
            # Execute
            result = model.generate("test prompt")
            
            # Assert
            assert "Missing required tool parameters" in result
    
    @patch('cli_code.models.gemini.genai')
    def test_generate_with_tool_not_found(self, mock_genai, mock_console):
        """Test generate method when a requested tool is not found."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to return a response with tool calls
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "nonexistent_tool"
            mock_part.function_call.args = '{}'
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response
            
            # Mock get_tool to return None for nonexistent tool
            with patch('cli_code.models.gemini.get_tool', return_value=None):
                # Execute
                result = model.generate("test prompt")
            
            # Assert
            assert "Requested tool not found" in result
    
    @patch('cli_code.models.gemini.genai')
    @patch('cli_code.models.gemini.get_tool')
    def test_generate_with_tool_execution_error(self, mock_get_tool, mock_genai, mock_console):
        """Test generate method when a tool execution raises an error."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to return a response with tool calls
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_part = MagicMock()
            mock_part.function_call = MagicMock()
            mock_part.function_call.name = "test_tool"
            mock_part.function_call.args = '{"param": "value"}'
            mock_response.candidates[0].content.parts = [mock_part]
            mock_model.generate_content.return_value = mock_response
            
            # Mock the tool to raise an exception
            tool_mock = MagicMock()
            tool_mock.execute.side_effect = Exception("Tool execution error")
            mock_get_tool.return_value = tool_mock
            
            # Execute
            result = model.generate("test prompt")
            
            # Assert
            assert "Error executing tool" in result
    
    @patch('cli_code.models.gemini.genai')
    def test_list_models_error(self, mock_genai, mock_console):
        """Test list_models method when an error occurs."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to raise an exception
            mock_genai.list_models.side_effect = Exception("List models error")
            
            # Execute
            result = model.list_models()
            
            # Assert
            assert result == []
            mock_console.print.assert_called()
            assert any("Error listing Gemini models" in str(call_args) for call_args in mock_console.print.call_args_list)
    
    @patch('cli_code.models.gemini.genai')
    def test_generate_with_empty_response(self, mock_genai, mock_console):
        """Test generate method when the API returns an empty response."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to return an empty response
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = []  # Empty candidates
            mock_model.generate_content.return_value = mock_response
            
            # Execute
            result = model.generate("test prompt")
            
            # Assert
            assert "No valid response received" in result
    
    @patch('cli_code.models.gemini.genai')
    def test_generate_with_malformed_response(self, mock_genai, mock_console):
        """Test generate method when the API returns a malformed response."""
        # Setup
        with patch('cli_code.models.gemini.log'):
            model = GeminiModel("valid_key", mock_console)
            model.genai_client = mock_genai
            
            # Configure the mock to return a malformed response
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content = None  # Missing content
            mock_model.generate_content.return_value = mock_response
            
            # Execute
            result = model.generate("test prompt")
            
            # Assert
            assert "Error parsing Gemini response" in result 