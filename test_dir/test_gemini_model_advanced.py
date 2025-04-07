"""
Tests specifically for the GeminiModel class targeting advanced scenarios and edge cases
to improve code coverage on complex methods like generate().
"""

import os
import json
import sys
from unittest.mock import patch, MagicMock, mock_open, call, ANY
import pytest

# Check if running in CI
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Handle imports
try:
    from cli_code.models.gemini import GeminiModel, MAX_AGENT_ITERATIONS
    from rich.console import Console
    import google.generativeai as genai
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    GeminiModel = MagicMock
    Console = MagicMock
    genai = MagicMock
    MAX_AGENT_ITERATIONS = 10

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestGeminiModelAdvanced:
    """Test suite for GeminiModel class focusing on complex methods and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock genai module
        self.genai_configure_patch = patch('google.generativeai.configure')
        self.mock_genai_configure = self.genai_configure_patch.start()
        
        self.genai_model_patch = patch('google.generativeai.GenerativeModel')
        self.mock_genai_model_class = self.genai_model_patch.start()
        self.mock_model_instance = MagicMock()
        self.mock_genai_model_class.return_value = self.mock_model_instance
        
        # Mock console
        self.mock_console = MagicMock(spec=Console)
        
        # Mock tool-related components
        self.get_tool_patch = patch('cli_code.models.gemini.get_tool')
        self.mock_get_tool = self.get_tool_patch.start()
        
        # Default tool mock
        self.mock_tool = MagicMock()
        self.mock_tool.execute.return_value = "Tool execution result"
        self.mock_get_tool.return_value = self.mock_tool
        
        # Mock initial context method to avoid complexity
        self.get_initial_context_patch = patch.object(
            GeminiModel, '_get_initial_context', return_value="Initial context")
        self.mock_get_initial_context = self.get_initial_context_patch.start()
        
        # Create model instance
        self.model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
    def teardown_method(self):
        """Tear down test fixtures."""
        self.genai_configure_patch.stop()
        self.genai_model_patch.stop()
        self.get_tool_patch.stop()
        self.get_initial_context_patch.stop()
    
    def test_generate_command_handling(self):
        """Test command handling in generate method."""
        # Test /exit command
        result = self.model.generate("/exit")
        assert result is None
        
        # Test /help command
        result = self.model.generate("/help")
        assert "Commands available" in result
    
    def test_generate_with_text_response(self):
        """Test generate method with a simple text response."""
        # Mock the LLM response to return a simple text
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_text_part = MagicMock()
        
        mock_text_part.text = "This is a simple text response."
        mock_content.parts = [mock_text_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Call generate
        result = self.model.generate("Tell me something interesting")
        
        # Verify calls
        self.mock_model_instance.generate_content.assert_called_once()
        assert "This is a simple text response." in result
        
    def test_generate_with_function_call(self):
        """Test generate method with a function call response."""
        # Set up mock response with function call
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        
        # Create function call part
        mock_function_part = MagicMock()
        mock_function_part.text = None
        mock_function_part.function_call = MagicMock()
        mock_function_part.function_call.name = "ls"
        mock_function_part.function_call.args = {"dir": "."}
        
        # Create text part for after function execution
        mock_text_part = MagicMock()
        mock_text_part.text = "Here are the directory contents."
        
        mock_content.parts = [mock_function_part, mock_text_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        
        # Set initial response
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Create a second response for after function execution
        mock_response2 = MagicMock()
        mock_candidate2 = MagicMock()
        mock_content2 = MagicMock()
        mock_text_part2 = MagicMock()
        
        mock_text_part2.text = "Function executed successfully. Here's the result."
        mock_content2.parts = [mock_text_part2]
        mock_candidate2.content = mock_content2
        mock_response2.candidates = [mock_candidate2]
        
        # Set up mock to return different responses on successive calls
        self.mock_model_instance.generate_content.side_effect = [mock_response, mock_response2]
        
        # Call generate
        result = self.model.generate("List the files in this directory")
        
        # Verify tool was looked up and executed
        self.mock_get_tool.assert_called_with("ls")
        self.mock_tool.execute.assert_called_once()
        
        # Verify final response
        assert "Function executed successfully" in result
    
    def test_generate_task_complete_tool(self):
        """Test generate method with task_complete tool call."""
        # Set up mock response with task_complete function call
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        
        # Create function call part
        mock_function_part = MagicMock()
        mock_function_part.text = None
        mock_function_part.function_call = MagicMock()
        mock_function_part.function_call.name = "task_complete"
        mock_function_part.function_call.args = {"summary": "Task completed successfully!"}
        
        mock_content.parts = [mock_function_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        
        # Set the response
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Call generate
        result = self.model.generate("Complete this task")
        
        # Verify result contains the summary
        assert "Task completed successfully!" in result
    
    def test_generate_with_empty_candidates(self):
        """Test generate method with empty candidates response."""
        # Mock response with no candidates
        mock_response = MagicMock()
        mock_response.candidates = []
        
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Call generate
        result = self.model.generate("Generate something")
        
        # Verify error handling
        assert "(Agent received response with no candidates)" in result
    
    def test_generate_with_empty_content(self):
        """Test generate method with empty content in candidate."""
        # Mock response with empty content
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.content = None
        mock_response.candidates = [mock_candidate]
        
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Call generate
        result = self.model.generate("Generate something")
        
        # Verify error handling
        assert "(Agent received response candidate with no content/parts)" in result
    
    def test_generate_with_api_error(self):
        """Test generate method when API throws an error."""
        # Mock API error
        api_error_message = "API Error"
        self.mock_model_instance.generate_content.side_effect = Exception(api_error_message)
        
        # Call generate
        result = self.model.generate("Generate something")
        
        # Verify error handling with specific assertions
        assert "Error calling Gemini API:" in result
        assert api_error_message in result
        
    def test_generate_max_iterations(self):
        """Test generate method with maximum iterations reached."""
        # Set up a response that will always include a function call, forcing iterations
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        
        # Create function call part
        mock_function_part = MagicMock()
        mock_function_part.text = None
        mock_function_part.function_call = MagicMock()
        mock_function_part.function_call.name = "ls"
        mock_function_part.function_call.args = {"dir": "."}
        
        mock_content.parts = [mock_function_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        
        # Make the model always return a function call
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Call generate
        result = self.model.generate("List files recursively")
        
        # Verify we hit the max iterations
        assert self.mock_model_instance.generate_content.call_count <= MAX_AGENT_ITERATIONS + 1
        assert "Maximum iterations reached" in result
        
    def test_generate_with_multiple_tools_per_response(self):
        """Test generate method with multiple tool calls in a single response."""
        # Set up mock response with multiple function calls
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        
        # Create first function call part
        mock_function_part1 = MagicMock()
        mock_function_part1.text = None
        mock_function_part1.function_call = MagicMock()
        mock_function_part1.function_call.name = "ls"
        mock_function_part1.function_call.args = {"dir": "."}
        
        # Create second function call part
        mock_function_part2 = MagicMock()
        mock_function_part2.text = None
        mock_function_part2.function_call = MagicMock()
        mock_function_part2.function_call.name = "view"
        mock_function_part2.function_call.args = {"file_path": "file.txt"}
        
        # Create text part
        mock_text_part = MagicMock()
        mock_text_part.text = "Here are the results."
        
        mock_content.parts = [mock_function_part1, mock_function_part2, mock_text_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        
        # Set up second response for after function execution
        mock_response2 = MagicMock()
        mock_candidate2 = MagicMock()
        mock_content2 = MagicMock()
        mock_text_part2 = MagicMock()
        
        mock_text_part2.text = "All functions executed."
        mock_content2.parts = [mock_text_part2]
        mock_candidate2.content = mock_content2
        mock_response2.candidates = [mock_candidate2]
        
        # Set up mock to return different responses
        self.mock_model_instance.generate_content.side_effect = [mock_response, mock_response2]
        
        # Call generate
        result = self.model.generate("List files and view a file")
        
        # Verify only the first function is executed (since we only process one per turn)
        self.mock_get_tool.assert_called_with("ls")
        assert "All functions executed" in result
    
    def test_manage_context_window_truncation(self):
        """Test specific context window management truncation with many messages."""
        # Add many messages to history
        for i in range(40):  # More than MAX_HISTORY_TURNS
            self.model.add_to_history({"role": "user", "parts": [f"Test message {i}"]})
            self.model.add_to_history({"role": "model", "parts": [f"Test response {i}"]})
        
        # Record length before management
        initial_length = len(self.model.history)
        
        # Call the management function
        self.model._manage_context_window()
        
        # Verify truncation occurred
        assert len(self.model.history) < initial_length
        
        # Verify the first message is still the system prompt with specific content check
        assert "System Prompt" in str(self.model.history[0])
        assert "function calling capabilities" in str(self.model.history[0])
        assert "CLI-Code" in str(self.model.history[0]) 