"""
Tests specifically for the OllamaModel class to improve code coverage.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

# Check if running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"

# Handle imports
try:
    from rich.console import Console

    from cli_code.models.ollama import OllamaModel

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    OllamaModel = MagicMock
    Console = MagicMock

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestOllamaModel:
    """Test suite for OllamaModel class, focusing on previously uncovered methods."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock OpenAI module before initialization
        self.openai_patch = patch("cli_code.models.ollama.OpenAI")
        self.mock_openai = self.openai_patch.start()

        # Mock the OpenAI client instance
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client

        # Mock console
        self.mock_console = MagicMock(spec=Console)

        # Mock os.path.isdir and os.path.isfile
        self.isdir_patch = patch("os.path.isdir")
        self.isfile_patch = patch("os.path.isfile")
        self.mock_isdir = self.isdir_patch.start()
        self.mock_isfile = self.isfile_patch.start()

        # Mock glob
        self.glob_patch = patch("glob.glob")
        self.mock_glob = self.glob_patch.start()

        # Mock open
        self.open_patch = patch("builtins.open", mock_open(read_data="# Test content"))
        self.mock_open = self.open_patch.start()

        # Mock get_tool
        self.get_tool_patch = patch("cli_code.models.ollama.get_tool")
        self.mock_get_tool = self.get_tool_patch.start()

        # Default tool mock
        self.mock_tool = MagicMock()
        self.mock_tool.execute.return_value = "ls output"
        self.mock_get_tool.return_value = self.mock_tool

    def teardown_method(self):
        """Tear down test fixtures."""
        self.openai_patch.stop()
        self.isdir_patch.stop()
        self.isfile_patch.stop()
        self.glob_patch.stop()
        self.open_patch.stop()
        self.get_tool_patch.stop()

    def test_init(self):
        """Test initialization of OllamaModel."""
        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

        # Check if OpenAI client was initialized correctly
        self.mock_openai.assert_called_once_with(base_url="http://localhost:11434", api_key="ollama")

        # Check model attributes
        assert model.api_url == "http://localhost:11434"
        assert model.model_name == "llama3"

        # Check history initialization
        assert len(model.history) == 1
        assert model.history[0]["role"] == "system"

    def test_get_initial_context_with_rules_dir(self):
        """Test getting initial context from .rules directory."""
        # Set up mocks
        self.mock_isdir.return_value = True
        self.mock_glob.return_value = [".rules/context.md", ".rules/tools.md"]

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
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

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
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

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        context = model._get_initial_context()

        # Verify tool was used
        self.mock_get_tool.assert_called_with("ls")
        self.mock_tool.execute.assert_called_once()

        # Check result content
        assert "Current directory contents" in context
        assert "ls output" in context

    def test_prepare_openai_tools(self):
        """Test preparation of tools in OpenAI function format."""
        # Create a mock for AVAILABLE_TOOLS
        with patch("cli_code.models.ollama.AVAILABLE_TOOLS") as mock_available_tools:
            # Sample tool definition
            mock_available_tools.return_value = {
                "test_tool": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "param1": {"type": "string", "description": "A string parameter"},
                        "param2": {"type": "integer", "description": "An integer parameter"},
                    },
                    "required": ["param1"],
                }
            }

            model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
            tools = model._prepare_openai_tools()

            # Verify tools format
            assert len(tools) == 1
            assert tools[0]["type"] == "function"
            assert tools[0]["function"]["name"] == "test_tool"
            assert "parameters" in tools[0]["function"]
            assert "properties" in tools[0]["function"]["parameters"]
            assert "param1" in tools[0]["function"]["parameters"]["properties"]
            assert "param2" in tools[0]["function"]["parameters"]["properties"]
            assert tools[0]["function"]["parameters"]["required"] == ["param1"]

    def test_manage_ollama_context(self):
        """Test context management for Ollama models."""
        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

        # Add many messages to force context truncation
        for i in range(30):
            model.add_to_history({"role": "user", "content": f"Test message {i}"})
            model.add_to_history({"role": "assistant", "content": f"Test response {i}"})

        # Call context management
        model._manage_ollama_context()

        # Verify history was truncated but system message preserved
        assert len(model.history) < 61  # Less than original count
        assert model.history[0]["role"] == "system"  # System message preserved

    def test_add_to_history(self):
        """Test adding messages to history."""
        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

        # Clear existing history
        model.history = []

        # Add a message
        message = {"role": "user", "content": "Test message"}
        model.add_to_history(message)

        # Verify message was added
        assert len(model.history) == 1
        assert model.history[0] == message

    def test_clear_history(self):
        """Test clearing history."""
        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

        # Add some messages
        model.add_to_history({"role": "user", "content": "Test message"})

        # Clear history
        model.clear_history()

        # Verify history was cleared
        assert len(model.history) == 0

    def test_list_models(self):
        """Test listing available models."""
        # Mock the completion response
        mock_response = MagicMock()
        mock_models = [
            {"id": "llama3", "object": "model", "created": 1621880188},
            {"id": "mistral", "object": "model", "created": 1622880188},
        ]
        mock_response.json.return_value = {"data": mock_models}

        # Set up client mock to return response
        self.mock_client.models.list.return_value.data = mock_models

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        result = model.list_models()

        # Verify client method called
        self.mock_client.models.list.assert_called_once()

        # Verify result
        assert result == mock_models

    def test_generate_with_function_calls(self):
        """Test generate method with function calls."""
        # Create response with function calls
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [MagicMock(function=MagicMock(name="test_tool", arguments='{"param1": "value1"}'))]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message, finish_reason="tool_calls")]

        # Set up client mock
        self.mock_client.chat.completions.create.return_value = mock_response

        # Mock get_tool to return a tool that executes successfully
        tool_mock = MagicMock()
        tool_mock.execute.return_value = "Tool execution result"
        self.mock_get_tool.return_value = tool_mock

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        result = model.generate("Test prompt")

        # Verify client method called
        self.mock_client.chat.completions.create.assert_called()

        # Verify tool execution
        tool_mock.execute.assert_called_once_with(param1="value1")

        # Check that there was a second API call with the tool results
        assert self.mock_client.chat.completions.create.call_count == 2
