"""
Tests specifically for the GeminiModel class to improve code coverage.
"""

import os
import json
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pytest

# Check if running in CI
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Handle imports
try:
    from cli_code.models.gemini import GeminiModel
    from rich.console import Console
    import google.generativeai as genai
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    GeminiModel = MagicMock
    Console = MagicMock
    genai = MagicMock

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestGeminiModel:
    """Test suite for GeminiModel class, focusing on previously uncovered methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock genai module
        self.genai_configure_patch = patch('google.generativeai.configure')
        self.mock_genai_configure = self.genai_configure_patch.start()
        
        self.genai_model_patch = patch('google.generativeai.GenerativeModel')
        self.mock_genai_model_class = self.genai_model_patch.start()
        self.mock_model_instance = MagicMock()
        self.mock_genai_model_class.return_value = self.mock_model_instance
        
        self.genai_list_models_patch = patch('google.generativeai.list_models')
        self.mock_genai_list_models = self.genai_list_models_patch.start()
        
        # Mock console
        self.mock_console = MagicMock(spec=Console)
        
        # Mock os.path.isdir and os.path.isfile
        self.isdir_patch = patch('os.path.isdir')
        self.isfile_patch = patch('os.path.isfile')
        self.mock_isdir = self.isdir_patch.start()
        self.mock_isfile = self.isfile_patch.start()
        
        # Mock glob
        self.glob_patch = patch('glob.glob')
        self.mock_glob = self.glob_patch.start()
        
        # Mock open
        self.open_patch = patch('builtins.open', mock_open(read_data="# Test content"))
        self.mock_open = self.open_patch.start()
        
        # Mock get_tool
        self.get_tool_patch = patch('cli_code.models.gemini.get_tool')
        self.mock_get_tool = self.get_tool_patch.start()
        
        # Default tool mock
        self.mock_tool = MagicMock()
        self.mock_tool.execute.return_value = "ls output"
        self.mock_get_tool.return_value = self.mock_tool
        
    def teardown_method(self):
        """Tear down test fixtures."""
        self.genai_configure_patch.stop()
        self.genai_model_patch.stop()
        self.genai_list_models_patch.stop()
        self.isdir_patch.stop()
        self.isfile_patch.stop()
        self.glob_patch.stop()
        self.open_patch.stop()
        self.get_tool_patch.stop()
    
    def test_initialization(self):
        """Test initialization of GeminiModel."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Check if genai was configured correctly
        self.mock_genai_configure.assert_called_once_with(api_key="fake-api-key")
        
        # Check if model instance was created correctly
        self.mock_genai_model_class.assert_called_once()
        assert model.api_key == "fake-api-key"
        assert model.current_model_name == "gemini-2.5-pro-exp-03-25"
        
        # Check history initialization
        assert len(model.history) == 2  # System prompt and initial model response
    
    def test_initialize_model_instance(self):
        """Test model instance initialization."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Call the method directly to test
        model._initialize_model_instance()
        
        # Verify model was created with correct parameters
        self.mock_genai_model_class.assert_called_with(
            model_name="gemini-2.5-pro-exp-03-25",
            generation_config=model.generation_config,
            safety_settings=model.safety_settings,
            system_instruction=model.system_instruction
        )
    
    def test_list_models(self):
        """Test listing available models."""
        # Set up mock response
        mock_model1 = MagicMock()
        mock_model1.name = "models/gemini-pro"
        mock_model1.display_name = "Gemini Pro"
        mock_model1.description = "A powerful model"
        mock_model1.supported_generation_methods = ["generateContent"]
        
        mock_model2 = MagicMock()
        mock_model2.name = "models/gemini-2.5-pro-exp-03-25"
        mock_model2.display_name = "Gemini 2.5 Pro"
        mock_model2.description = "An experimental model"
        mock_model2.supported_generation_methods = ["generateContent"]
        
        self.mock_genai_list_models.return_value = [mock_model1, mock_model2]
        
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        result = model.list_models()
        
        # Verify list_models was called
        self.mock_genai_list_models.assert_called_once()
        
        # Verify result format
        assert len(result) == 2
        assert result[0]["id"] == "models/gemini-pro"
        assert result[0]["name"] == "Gemini Pro"
        assert result[1]["id"] == "models/gemini-2.5-pro-exp-03-25"
    
    def test_get_initial_context_with_rules_dir(self):
        """Test getting initial context from .rules directory."""
        # Set up mocks
        self.mock_isdir.return_value = True
        self.mock_glob.return_value = [".rules/context.md", ".rules/tools.md"]
        
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        context = model._get_initial_context()
        
        # Verify directory check
        self.mock_isdir.assert_called_with(".rules")
        
        # Verify glob search
        self.mock_glob.assert_called_with(".rules/*.md")
        
        # Verify files were read
        assert self.mock_open.call_count == 2
        
        # Check result content
        assert "Project rules and guidelines:" in context
        assert "# Content from" in context
    
    def test_get_initial_context_with_readme(self):
        """Test getting initial context from README.md when no .rules directory."""
        # Set up mocks
        self.mock_isdir.return_value = False
        self.mock_isfile.return_value = True
        
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        context = model._get_initial_context()
        
        # Verify README check
        self.mock_isfile.assert_called_with("README.md")
        
        # Verify file reading
        self.mock_open.assert_called_once_with("README.md", "r", encoding="utf-8", errors="ignore")
        
        # Check result content
        assert "Project README:" in context
    
    def test_get_initial_context_with_ls_fallback(self):
        """Test getting initial context via ls when no .rules or README."""
        # Set up mocks
        self.mock_isdir.return_value = False
        self.mock_isfile.return_value = False
        
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        context = model._get_initial_context()
        
        # Verify tool was used
        self.mock_get_tool.assert_called_with("ls")
        self.mock_tool.execute.assert_called_once()
        
        # Check result content
        assert "Current directory contents" in context
        assert "ls output" in context
    
    def test_create_tool_definitions(self):
        """Test creation of tool definitions for Gemini."""
        # Create a mock for AVAILABLE_TOOLS
        with patch('cli_code.models.gemini.AVAILABLE_TOOLS') as mock_available_tools:
            # Sample tool definition
            mock_available_tools.return_value = {
                "test_tool": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "param1": {"type": "string", "description": "A string parameter"},
                        "param2": {"type": "integer", "description": "An integer parameter"}
                    },
                    "required": ["param1"]
                }
            }
            
            model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
            tools = model._create_tool_definitions()
            
            # Verify tools format
            assert len(tools) == 1
            assert tools[0]["name"] == "test_tool"
            assert "description" in tools[0]
            assert "parameters" in tools[0]
    
    def test_create_system_prompt(self):
        """Test creation of system prompt."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        prompt = model._create_system_prompt()
        
        # Verify prompt contains expected content
        assert "function calling capabilities" in prompt
        assert "System Prompt for CLI-Code" in prompt
    
    def test_manage_context_window(self):
        """Test context window management."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Add many messages to force context truncation
        for i in range(30):
            model.add_to_history({"role": "user", "parts": [f"Test message {i}"]})
            model.add_to_history({"role": "model", "parts": [f"Test response {i}"]})
        
        # Record initial length
        initial_length = len(model.history)
        
        # Call context management
        model._manage_context_window()
        
        # Verify history was truncated
        assert len(model.history) < initial_length
    
    def test_extract_text_from_response(self):
        """Test extracting text from Gemini response."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Create mock response with text
        mock_response = MagicMock()
        mock_response.parts = [{"text": "Response text"}]
        
        # Extract text
        result = model._extract_text_from_response(mock_response)
        
        # Verify extraction
        assert result == "Response text"
    
    def test_find_last_model_text(self):
        """Test finding last model text in history."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Clear history
        model.history = []
        
        # Add history entries
        model.add_to_history({"role": "user", "parts": ["User message 1"]})
        model.add_to_history({"role": "model", "parts": ["Model response 1"]})
        model.add_to_history({"role": "user", "parts": ["User message 2"]})
        model.add_to_history({"role": "model", "parts": ["Model response 2"]})
        
        # Find last model text
        result = model._find_last_model_text(model.history)
        
        # Verify result
        assert result == "Model response 2"
    
    def test_add_to_history(self):
        """Test adding messages to history."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Clear history
        model.history = []
        
        # Add a message
        entry = {"role": "user", "parts": ["Test message"]}
        model.add_to_history(entry)
        
        # Verify message was added
        assert len(model.history) == 1
        assert model.history[0] == entry
    
    def test_clear_history(self):
        """Test clearing history."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Add a message
        model.add_to_history({"role": "user", "parts": ["Test message"]})
        
        # Clear history
        model.clear_history()
        
        # Verify history was cleared
        assert len(model.history) == 0
    
    def test_get_help_text(self):
        """Test getting help text."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        help_text = model._get_help_text()
        
        # Verify help text content
        assert "CLI-Code Assistant Help" in help_text
        assert "Commands" in help_text
    
    def test_generate_with_function_calls(self):
        """Test generate method with function calls."""
        # Set up mock response with function call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [
            {
                "functionCall": {
                    "name": "test_tool",
                    "args": {"param1": "value1"}
                }
            }
        ]
        mock_response.candidates[0].finish_reason = "FUNCTION_CALL"
        
        # Set up model instance to return the mock response
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # Mock tool execution
        tool_mock = MagicMock()
        tool_mock.execute.return_value = "Tool execution result"
        self.mock_get_tool.return_value = tool_mock
        
        # Create model
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        
        # Call generate
        result = model.generate("Test prompt")
        
        # Verify model was called
        self.mock_model_instance.generate_content.assert_called()
        
        # Verify tool execution
        tool_mock.execute.assert_called_with(param1="value1")
        
        # There should be a second call to generate_content with the tool result
        assert self.mock_model_instance.generate_content.call_count >= 2 