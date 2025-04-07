"""
Tests specifically for the GeminiModel class to improve code coverage.
This file focuses on increasing coverage for the generate method and its edge cases.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open, call, PropertyMock
import pytest

# Check if running in CI
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Handle imports
try:
    from cli_code.models.gemini import GeminiModel, MAX_AGENT_ITERATIONS, FALLBACK_MODEL
    from rich.console import Console
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    GeminiModel = MagicMock
    Console = MagicMock
    genai = MagicMock
    ResourceExhausted = Exception

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestGeminiModelGenerateMethod:
    """Test suite for GeminiModel generate method, focusing on error paths and edge cases."""
    
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
        
        # Mock get_tool
        self.get_tool_patch = patch('cli_code.models.gemini.get_tool')
        self.mock_get_tool = self.get_tool_patch.start()
        
        # Default tool mock
        self.mock_tool = MagicMock()
        self.mock_tool.execute.return_value = "Tool executed successfully"
        self.mock_get_tool.return_value = self.mock_tool
        
        # Mock questionary confirm
        self.mock_confirm = MagicMock()
        self.questionary_patch = patch('questionary.confirm', return_value=self.mock_confirm)
        self.mock_questionary = self.questionary_patch.start()
        
        # Mock MAX_AGENT_ITERATIONS to limit loop execution
        self.max_iterations_patch = patch('cli_code.models.gemini.MAX_AGENT_ITERATIONS', 1)
        self.mock_max_iterations = self.max_iterations_patch.start()
        
        # Set up basic model
        self.model = GeminiModel("fake-api-key", self.mock_console, "gemini-pro")
        
        # Prepare mock response for basic tests
        self.mock_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        # Set up text part
        text_part = MagicMock()
        text_part.text = "This is a test response"
        
        # Set up content parts
        content.parts = [text_part]
        candidate.content = content
        self.mock_response.candidates = [candidate]
        
        # Setup model to return this response by default
        self.mock_model_instance.generate_content.return_value = self.mock_response
        
    def teardown_method(self):
        """Tear down test fixtures."""
        self.genai_configure_patch.stop()
        self.genai_model_patch.stop()
        self.get_tool_patch.stop()
        self.questionary_patch.stop()
        self.max_iterations_patch.stop()
    
    def test_generate_with_exit_command(self):
        """Test generating with /exit command."""
        result = self.model.generate("/exit")
        assert result is None
    
    def test_generate_with_help_command(self):
        """Test generating with /help command."""
        result = self.model.generate("/help")
        assert "Interactive Commands:" in result
    
    def test_generate_with_simple_text_response(self):
        """Test basic text response generation."""
        # Create a simple text-only response
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        
        # Set up text part that doesn't trigger function calls
        mock_text_part = MagicMock()
        mock_text_part.text = "This is a test response"
        mock_text_part.function_call = None  # Ensure no function call
        
        # Set up content parts with only text
        mock_content.parts = [mock_text_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        
        # Make generate_content return our simple response
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Run the test
        result = self.model.generate("Tell me about Python")
        
        # Verify the call and response
        self.mock_model_instance.generate_content.assert_called_once()
        assert "This is a test response" in result
    
    def test_generate_with_empty_candidates(self):
        """Test handling of empty candidates in response."""
        # Prepare empty candidates
        empty_response = MagicMock()
        empty_response.candidates = []
        self.mock_model_instance.generate_content.return_value = empty_response
        
        result = self.model.generate("Hello")
        
        assert "(Agent received response with no candidates)" in result
    
    def test_generate_with_empty_content(self):
        """Test handling of empty content in response candidate."""
        # Prepare empty content
        empty_response = MagicMock()
        empty_candidate = MagicMock()
        empty_candidate.content = None
        empty_response.candidates = [empty_candidate]
        self.mock_model_instance.generate_content.return_value = empty_response
        
        result = self.model.generate("Hello")
        
        assert "(Agent received response candidate with no content/parts)" in result
    
    def test_generate_with_function_call(self):
        """Test generating with function call in response."""
        # Create function call part
        function_call_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        function_part = MagicMock()
        function_part.function_call = MagicMock()
        function_part.function_call.name = "ls"
        function_part.function_call.args = {"path": "."}
        
        content.parts = [function_part]
        candidate.content = content
        function_call_response.candidates = [candidate]
        
        self.mock_model_instance.generate_content.return_value = function_call_response
        
        # Execute
        result = self.model.generate("List files")
        
        # Verify tool was called
        self.mock_get_tool.assert_called_with("ls")
        self.mock_tool.execute.assert_called_with(path=".")
    
    def test_generate_with_missing_tool(self):
        """Test handling when tool is not found."""
        # Create function call part for non-existent tool
        function_call_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        function_part = MagicMock()
        function_part.function_call = MagicMock()
        function_part.function_call.name = "nonexistent_tool"
        function_part.function_call.args = {}
        
        content.parts = [function_part]
        candidate.content = content
        function_call_response.candidates = [candidate]
        
        self.mock_model_instance.generate_content.return_value = function_call_response
        
        # Set up get_tool to return None
        self.mock_get_tool.return_value = None
        
        # Execute
        result = self.model.generate("Use nonexistent tool")
        
        # Verify error handling
        self.mock_get_tool.assert_called_with("nonexistent_tool")
        self.mock_console.print.assert_any_call(
            "[red] -> Error executing nonexistent_tool: Error: Tool 'nonexistent_tool' is not available....[/red]"
        )
    
    def test_generate_with_tool_execution_error(self):
        """Test handling when tool execution raises an error."""
        # Create function call part
        function_call_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        function_part = MagicMock()
        function_part.function_call = MagicMock()
        function_part.function_call.name = "ls"
        function_part.function_call.args = {"path": "."}
        
        content.parts = [function_part]
        candidate.content = content
        function_call_response.candidates = [candidate]
        
        self.mock_model_instance.generate_content.return_value = function_call_response
        
        # Set up tool to raise exception
        self.mock_tool.execute.side_effect = Exception("Tool execution failed")
        
        # Execute
        result = self.model.generate("List files")
        
        # Verify error handling
        self.mock_get_tool.assert_called_with("ls")
        self.mock_console.print.assert_any_call(
            "[red] -> Error executing ls: Error executing tool ls: Tool execution failed...[/red]"
        )
    
    def test_generate_with_task_complete(self):
        """Test handling of task_complete tool call."""
        # Create function call part for task_complete
        function_call_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        function_part = MagicMock()
        function_part.function_call = MagicMock()
        function_part.function_call.name = "task_complete"
        function_part.function_call.args = {"summary": "Task completed successfully"}
        
        content.parts = [function_part]
        candidate.content = content
        function_call_response.candidates = [candidate]
        
        self.mock_model_instance.generate_content.return_value = function_call_response
        
        # Set up task_complete tool
        task_complete_tool = MagicMock()
        task_complete_tool.execute.return_value = "Task completed successfully with details"
        self.mock_get_tool.return_value = task_complete_tool
        
        # Execute
        result = self.model.generate("Complete task")
        
        # Verify task completion handling
        self.mock_get_tool.assert_called_with("task_complete")
        assert result == "Task completed successfully with details"
    
    def test_generate_with_file_edit_confirmation_accepted(self):
        """Test handling of file edit confirmation when accepted."""
        # Create function call part for edit
        function_call_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        function_part = MagicMock()
        function_part.function_call = MagicMock()
        function_part.function_call.name = "edit"
        function_part.function_call.args = {
            "file_path": "test.py",
            "content": "print('hello world')"
        }
        
        content.parts = [function_part]
        candidate.content = content
        function_call_response.candidates = [candidate]
        
        self.mock_model_instance.generate_content.return_value = function_call_response
        
        # Set up confirmation to return True
        self.mock_confirm.ask.return_value = True
        
        # Execute
        result = self.model.generate("Edit test.py")
        
        # Verify confirmation flow
        self.mock_confirm.ask.assert_called_once()
        self.mock_get_tool.assert_called_with("edit")
        self.mock_tool.execute.assert_called_with(file_path="test.py", content="print('hello world')")
    
    def test_generate_with_file_edit_confirmation_rejected(self):
        """Test handling of file edit confirmation when rejected."""
        # Create function call part for edit
        function_call_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        function_part = MagicMock()
        function_part.function_call = MagicMock()
        function_part.function_call.name = "edit"
        function_part.function_call.args = {
            "file_path": "test.py",
            "content": "print('hello world')"
        }
        
        content.parts = [function_part]
        candidate.content = content
        function_call_response.candidates = [candidate]
        
        self.mock_model_instance.generate_content.return_value = function_call_response
        
        # Set up confirmation to return False
        self.mock_confirm.ask.return_value = False
        
        # Execute
        result = self.model.generate("Edit test.py")
        
        # Verify rejection handling
        self.mock_confirm.ask.assert_called_once()
        # Tool should not be executed if rejected
        self.mock_tool.execute.assert_not_called()
    
    def test_generate_with_quota_exceeded_fallback(self):
        """Test handling of quota exceeded with fallback model."""
        # Temporarily restore MAX_AGENT_ITERATIONS to allow proper fallback
        with patch('cli_code.models.gemini.MAX_AGENT_ITERATIONS', 10):
            # Create a simple text-only response for the fallback model
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            
            # Set up text part
            mock_text_part = MagicMock()
            mock_text_part.text = "This is a test response"
            mock_text_part.function_call = None  # Ensure no function call
            
            # Set up content parts
            mock_content.parts = [mock_text_part]
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]
            
            # Set up first call to raise ResourceExhausted, second call to return our mocked response
            self.mock_model_instance.generate_content.side_effect = [
                ResourceExhausted("Quota exceeded"),
                mock_response
            ]
            
            # Execute
            result = self.model.generate("Hello")
            
            # Verify fallback handling
            assert self.model.current_model_name == FALLBACK_MODEL
            assert "This is a test response" in result
            self.mock_console.print.assert_any_call(
                f"[bold yellow]Quota limit reached for gemini-pro. Switching to fallback model ({FALLBACK_MODEL})...[/bold yellow]"
            )
    
    def test_generate_with_quota_exceeded_on_fallback(self):
        """Test handling when quota is exceeded even on fallback model."""
        # Set the current model to already be the fallback
        self.model.current_model_name = FALLBACK_MODEL
        
        # Set up call to raise ResourceExhausted
        self.mock_model_instance.generate_content.side_effect = ResourceExhausted("Quota exceeded")
        
        # Execute
        result = self.model.generate("Hello")
        
        # Verify fallback failure handling
        assert "Error: API quota exceeded for primary and fallback models" in result
        self.mock_console.print.assert_any_call(
            "[bold red]API quota exceeded for primary and fallback models. Please check your plan/billing.[/bold red]"
        )
    
    def test_generate_with_max_iterations_reached(self):
        """Test handling when max iterations are reached."""
        # Set up responses to keep returning function calls that don't finish the task
        function_call_response = MagicMock()
        candidate = MagicMock()
        content = MagicMock()
        
        function_part = MagicMock()
        function_part.function_call = MagicMock()
        function_part.function_call.name = "ls"
        function_part.function_call.args = {"path": "."}
        
        content.parts = [function_part]
        candidate.content = content
        function_call_response.candidates = [candidate]
        
        # Always return a function call that will continue the loop
        self.mock_model_instance.generate_content.return_value = function_call_response
        
        # Patch MAX_AGENT_ITERATIONS to a smaller value for testing
        with patch('cli_code.models.gemini.MAX_AGENT_ITERATIONS', 3):
            result = self.model.generate("List files recursively")
        
        # Verify max iterations handling
        assert "(Task exceeded max iterations" in result
    
    def test_generate_with_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        # Set up generate_content to raise an exception
        self.mock_model_instance.generate_content.side_effect = Exception("Unexpected error")
        
        # Execute
        result = self.model.generate("Hello")
        
        # Verify exception handling
        assert "Error during agent processing: Unexpected error" in result 