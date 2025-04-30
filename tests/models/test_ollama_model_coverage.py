"""
Tests specifically for the OllamaModel class to improve code coverage.
This file focuses on testing methods and branches that aren't well covered.
"""

import json
import os
import sys
import unittest
import unittest.mock as mock
from unittest.mock import MagicMock, call, mock_open, patch

import openai
import pytest

# Check if running in CI
IS_CI = os.environ.get("CI", "false").lower() == "true"

# Handle imports
try:
    # Mock the OpenAI import check first
    sys.modules["openai"] = MagicMock()

    import requests
    from rich.console import Console

    from cli_code.models.ollama import OllamaModel

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    OllamaModel = MagicMock
    Console = MagicMock
    requests = MagicMock

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IS_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestOllamaModelCoverage(unittest.TestCase):
    """Test suite for OllamaModel class methods that need more coverage."""

    def setup_method(self, method):
        """Set up mocks for OllamaModel tests."""
        # Removed incorrect patch for load_config
        # self.mock_load_config = patch('cli_code.models.ollama.load_config').start()

        # Patch openai.OpenAI *where it is imported* in the ollama module
        self.mock_openai_class = patch("cli_code.models.ollama.OpenAI").start()
        self.mock_ollama_client = MagicMock(spec=openai.OpenAI)
        self.mock_openai_class.return_value = self.mock_ollama_client
        # Explicitly create the nested 'models' attribute for the spec'd mock
        self.mock_ollama_client.models = MagicMock()

        # Mock Console
        self.mock_console = MagicMock(spec=Console)

        # Mock file system operations used by _get_initial_context
        self.mock_isdir = patch("os.path.isdir").start()
        self.mock_isfile = patch("os.path.isfile").start()
        self.mock_glob = patch("glob.glob").start()
        self.mock_open = patch("builtins.open", mock_open(read_data="Test content")).start()

        # Mock get_tool used by _get_initial_context and generate
        self.mock_get_tool = patch("cli_code.models.ollama.get_tool").start()
        self.mock_tool_instance = MagicMock()
        self.mock_get_tool.return_value = self.mock_tool_instance

        # Create OllamaModel instance with necessary args
        self.model = OllamaModel(
            api_url="http://mock-ollama-url:11434", console=self.mock_console, model_name="mock-llama3"
        )

        self.addCleanup(patch.stopall)

    def test_initialization(self):
        """Test initialization of OllamaModel."""
        # The setup_method now handles instantiation and basic checks
        assert self.model.api_url == "http://mock-ollama-url:11434"
        assert self.model.model_name == "mock-llama3"
        # Assert it's the *exact same* mock instance created in setup
        assert self.model.client is self.mock_ollama_client
        assert len(self.model.history) == 1  # Just the system prompt initially
        self.mock_console.print.assert_not_called()  # No context messages during basic init

    def test_list_models(self):
        """Test listing available models."""
        # Configure the mock response for the client's list method
        mock_model_data = MagicMock()
        mock_model_data.name = "test_model:latest"
        mock_model_data.modified_at = "2023-01-01T10:00:00Z"
        mock_model_data.size = 1234567890

        # Create a mock response object with a .data attribute
        mock_list_response = MagicMock()
        mock_list_response.data = [mock_model_data]  # The code accesses response.data
        self.mock_ollama_client.models.list.return_value = mock_list_response

        result = self.model.list_models()

        self.mock_ollama_client.models.list.assert_called_once()
        # Check the formatted output structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["name"] == "test_model:latest"
        # Check for expected keys in the formatted dictionary
        assert "id" in result[0]
        assert "name" in result[0]
        assert "modified_at" in result[0]
        assert "size" in result[0]
        assert "details" in result[0]

    def test_list_models_with_error(self):
        """Test handling errors when listing models."""
        # Update check for RequestException
        self.mock_ollama_client.models.list.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )  # Simulate OpenAI client error

        models_list = self.model.list_models()

        self.mock_ollama_client.models.list.assert_called_once()
        # The method should now log the error and return None
        assert models_list is None
        # The error handler prints two messages
        assert self.mock_console.print.call_count == 2
        self.mock_console.print.assert_any_call(
            "[bold red]Error contacting Ollama endpoint 'http://mock-ollama-url:11434':[/bold red] Connection error."
        )
        self.mock_console.print.assert_any_call(
            "[yellow]Ensure the Ollama server is running and the API URL is correct.[/yellow]"
        )

    def test_get_initial_context_with_rules_dir(self):
        """Test getting initial context from .rules directory."""
        # Set up mocks
        self.mock_isdir.return_value = True
        self.mock_glob.return_value = [".rules/context.md", ".rules/tools.md"]
        # Configure mock_open specifically for this test
        mock_open_instance = mock_open(read_data="Rule Content")
        with patch("builtins.open", mock_open_instance):
            context = self.model._get_initial_context()

        # Verify directory check
        self.mock_isdir.assert_called_with(".rules")

        # Verify glob search
        self.mock_glob.assert_called_with(".rules/*.md")

        # Verify files were read
        assert mock_open_instance.call_count == 2
        mock_open_instance.assert_any_call(".rules/context.md", "r", encoding="utf-8", errors="ignore")
        mock_open_instance.assert_any_call(".rules/tools.md", "r", encoding="utf-8", errors="ignore")

        # Check result content
        assert "Project rules and guidelines:" in context
        assert "Rule Content" in context
        self.mock_console.print.assert_called_with("[dim]Context initialized from .rules/*.md files.[/dim]")

    def test_get_initial_context_with_readme(self):
        """Test getting initial context from README.md when no .rules directory."""
        # Set up mocks
        self.mock_isdir.return_value = False
        self.mock_isfile.return_value = True
        # Configure mock_open specifically for this test
        mock_open_instance = mock_open(read_data="Readme Content")
        with patch("builtins.open", mock_open_instance):
            context = self.model._get_initial_context()

        # Verify README check
        self.mock_isfile.assert_called_with("README.md")

        # Verify file reading
        mock_open_instance.assert_called_once_with("README.md", "r", encoding="utf-8", errors="ignore")

        # Check result content
        assert "Project README:" in context
        assert "Readme Content" in context
        self.mock_console.print.assert_called_with("[dim]Context initialized from README.md.[/dim]")

    def test_get_initial_context_with_ls_fallback(self):
        """Test getting initial context via ls when no .rules or README."""
        # Set up mocks
        self.mock_isdir.return_value = False
        self.mock_isfile.return_value = False
        self.mock_tool_instance.execute.return_value = "Directory listing content"

        context = self.model._get_initial_context()

        # Verify tool was called
        self.mock_get_tool.assert_called_with("ls")
        self.mock_tool_instance.execute.assert_called_once()
        # Check result content
        assert "Current directory contents" in context
        assert "Directory listing content" in context
        self.mock_console.print.assert_called_with("[dim]Directory context acquired via 'ls'.[/dim]")

    def test_clear_history(self):
        """Test history clearing functionality."""
        # Add some items to history
        self.model.add_to_history({"role": "user", "content": "Test message"})

        # Clear history
        self.model.clear_history()

        # Check that history is reset with just the system prompt
        assert len(self.model.history) == 1
        assert self.model.history[0]["role"] == "system"

    def test_add_to_history(self):
        """Test adding messages to history."""
        initial_length = len(self.model.history)

        # Add a user message
        self.model.add_to_history({"role": "user", "content": "Test user message"})

        # Check that message was added
        assert len(self.model.history) == initial_length + 1
        assert self.model.history[-1]["role"] == "user"
        assert self.model.history[-1]["content"] == "Test user message"
