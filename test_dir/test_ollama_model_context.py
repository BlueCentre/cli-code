"""
Tests for the Ollama Model context management functionality.

To run these tests specifically:
    python -m pytest test_dir/test_ollama_model_context.py
    
To run a specific test:
    python -m pytest test_dir/test_ollama_model_context.py::TestOllamaModelContext::test_manage_ollama_context_truncation_needed
    
To run all tests related to context management:
    python -m pytest -k "ollama_context"
"""
import os
import logging
import json
import glob
from unittest.mock import patch, MagicMock, mock_open

import pytest
from rich.console import Console

# Import from the actual source location
from src.cli_code.models.ollama import OllamaModel, OLLAMA_MAX_CONTEXT_TOKENS
from src.cli_code.utils import count_tokens
from src.cli_code.tools import get_tool


class TestOllamaModelContext:
    """Tests for the OllamaModel's context management functionality."""

    @pytest.fixture
    def mock_openai(self):
        """Mock the OpenAI client and return a mock client."""
        with patch("src.cli_code.models.ollama.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ollama_model(self, mock_openai):
        """Create an OllamaModel instance with mocked dependencies."""
        with patch("src.cli_code.models.ollama.count_tokens") as mock_count_tokens:
            # Setup token counting mock
            mock_count_tokens.return_value = 100  # Default token count for messages
            
            console = Console()
            model = OllamaModel(api_url="http://localhost:11434/v1", console=console, model_name="codellama")
            
            # Mock the client's chat.completions.create method
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_openai.chat.completions.create.return_value = mock_response
            
            yield model

    def test_add_to_history(self, ollama_model):
        """Test adding messages to the conversation history."""
        # Initial history should contain only the system prompt
        assert len(ollama_model.history) == 1
        assert ollama_model.history[0]["role"] == "system"
        
        # Add a user message
        user_message = {"role": "user", "content": "Test message"}
        ollama_model.add_to_history(user_message)
        
        # Check that message was added
        assert len(ollama_model.history) == 2
        assert ollama_model.history[1] == user_message

    def test_clear_history(self, ollama_model):
        """Test clearing the conversation history."""
        # Add a few messages
        ollama_model.add_to_history({"role": "user", "content": "User message"})
        ollama_model.add_to_history({"role": "assistant", "content": "Assistant response"})
        assert len(ollama_model.history) == 3  # System + 2 added messages
        
        # Clear history
        ollama_model.clear_history()
        
        # Check that history was cleared and system prompt was re-added
        assert len(ollama_model.history) == 1
        assert ollama_model.history[0]["role"] == "system"
        assert ollama_model.history[0]["content"] == ollama_model.system_prompt

    @patch("src.cli_code.models.ollama.count_tokens")
    def test_manage_ollama_context_no_truncation_needed(self, mock_count_tokens, ollama_model):
        """Test _manage_ollama_context when truncation is not needed."""
        # Setup count_tokens to return a small number of tokens
        mock_count_tokens.return_value = OLLAMA_MAX_CONTEXT_TOKENS // 4  # Well under the limit
        
        # Add some messages
        ollama_model.add_to_history({"role": "user", "content": "User message 1"})
        ollama_model.add_to_history({"role": "assistant", "content": "Assistant response 1"})
        initial_history_length = len(ollama_model.history)
        
        # Call the manage context method
        ollama_model._manage_ollama_context()
        
        # Assert that history was not modified since we're under the token limit
        assert len(ollama_model.history) == initial_history_length

    @patch("src.cli_code.models.ollama.count_tokens")
    def test_manage_ollama_context_truncation_needed(self, mock_count_tokens, ollama_model):
        """Test _manage_ollama_context when truncation is needed."""
        # Reset the mock to ensure consistent behavior
        mock_count_tokens.reset_mock()
        
        # Initial history should be just the system message
        assert len(ollama_model.history) == 1
        assert ollama_model.history[0]["role"] == "system"
        
        # Initial total token setup - return a small value so messages are added without truncation
        mock_count_tokens.return_value = 10  # Each message is very small
        
        # Add many messages to the history
        for i in range(5):
            ollama_model.add_to_history({"role": "user", "content": f"User message {i}"})
            ollama_model.add_to_history({"role": "assistant", "content": f"Assistant response {i}"})
            
        # Add a special last message to track
        last_message = {"role": "user", "content": "This is the very last message"}
        ollama_model.add_to_history(last_message)
            
        # Verify we now have 12 messages (1 system + 5 user + 5 assistant + 1 last)
        assert len(ollama_model.history) == 12
        initial_history_length = len(ollama_model.history)
        
        # Now modify the mock to ensure truncation will happen in _manage_ollama_context
        # Make each message very large to force aggressive truncation
        mock_count_tokens.return_value = OLLAMA_MAX_CONTEXT_TOKENS // 2  # Each message is half the limit
        
        # Call the function that should truncate history
        ollama_model._manage_ollama_context()
        
        # After truncation, verify the history was actually truncated
        assert len(ollama_model.history) < initial_history_length, f"Expected fewer than {initial_history_length} messages, got {len(ollama_model.history)}"
        
        # Verify system message is still at position 0
        assert ollama_model.history[0]["role"] == "system"
        
        # Verify the most recent message is still present (last message we added)
        assert ollama_model.history[-1] == last_message

    @patch("src.cli_code.models.ollama.count_tokens")
    def test_manage_ollama_context_preserves_recent_messages(self, mock_count_tokens, ollama_model):
        """Test _manage_ollama_context preserves recent messages."""
        # Set up token count to exceed the limit to trigger truncation
        mock_count_tokens.side_effect = lambda text: OLLAMA_MAX_CONTEXT_TOKENS * 2  # Double the limit
        
        # Add a system message first
        system_message = {"role": "system", "content": "System instruction"}
        ollama_model.history = [system_message]
        
        # Add multiple messages to the history
        for i in range(20):
            ollama_model.add_to_history({"role": "user", "content": f"User message {i}"})
            ollama_model.add_to_history({"role": "assistant", "content": f"Assistant response {i}"})
        
        # Mark some recent messages to verify they're preserved
        recent_messages = [
            {"role": "user", "content": "Important recent user message"},
            {"role": "assistant", "content": "Important recent assistant response"}
        ]
        
        for msg in recent_messages:
            ollama_model.add_to_history(msg)
            
        # Call the function that should truncate history
        ollama_model._manage_ollama_context()
        
        # Verify system message is preserved
        assert ollama_model.history[0]["role"] == "system"
        assert ollama_model.history[0]["content"] == "System instruction"
        
        # Verify the most recent messages are preserved at the end of history
        assert ollama_model.history[-2:] == recent_messages

    @patch("os.path.isdir")
    @patch("os.path.isfile")
    @patch("glob.glob")
    def test_get_initial_context_with_rules_directory(self, mock_glob, mock_isfile, mock_isdir, ollama_model):
        """Test _get_initial_context when .rules directory exists with markdown files."""
        # Mock directory and file existence
        mock_isdir.return_value = True
        mock_isfile.return_value = False  # No README.md
        
        # Mock glob to return markdown files
        mock_glob.return_value = [".rules/context.md", ".rules/tools.md"]
        
        # Mock file reading - simplified approach
        m = mock_open(read_data="# Context\nThis is the context file.")
        
        with patch("builtins.open", m):
            context = ollama_model._get_initial_context()
        
        # Verify context contains expected formatting
        assert "Project rules and guidelines:" in context
        assert "```markdown" in context  # Should contain markdown formatting
        assert "Content from" in context  # Should mention the files

    @patch("os.path.isdir")
    @patch("os.path.isfile")
    def test_get_initial_context_with_readme(self, mock_isfile, mock_isdir, ollama_model):
        """Test _get_initial_context when README.md exists but no .rules directory."""
        # Mock directory and file existence
        mock_isdir.return_value = False  # No .rules directory
        mock_isfile.return_value = True  # README.md exists
        
        # Mock file reading for README.md
        readme_content = "# Project README\nThis is the project readme."
        
        with patch("builtins.open", mock_open(read_data=readme_content)):
            context = ollama_model._get_initial_context()
        
        # Verify context content contains README.md content
        assert "Project README:" in context
        assert "This is the project readme." in context

    @patch("os.path.isdir")
    @patch("os.path.isfile")
    @patch("src.cli_code.models.ollama.get_tool")
    def test_get_initial_context_fallback_to_ls(self, mock_get_tool, mock_isfile, mock_isdir, ollama_model):
        """Test _get_initial_context falling back to ls command when no .rules or README exists."""
        # Mock directory and file existence
        mock_isdir.return_value = False  # No .rules directory
        mock_isfile.return_value = False  # No README.md
        
        # Mock ls tool execution
        mock_ls_tool = MagicMock()
        mock_ls_tool.execute.return_value = "file1.txt\nfile2.txt\ndirectory1/"
        mock_get_tool.return_value = mock_ls_tool
        
        context = ollama_model._get_initial_context()
        
        # Verify ls tool was used and output included in context
        mock_get_tool.assert_called_once_with("ls")
        mock_ls_tool.execute.assert_called_once()
        assert "Current directory contents" in context
        assert "file1.txt" in context
        assert "file2.txt" in context
        assert "directory1/" in context

    def test_prepare_openai_tools(self, ollama_model):
        """Test that tools are prepared for the OpenAI API format."""
        # Rather than mocking a specific method, just check that the result is well-formed
        # This relies on the actual implementation, not a mock of _prepare_openai_tools
        
        # The method should return a list of dictionaries with function definitions
        tools = ollama_model._prepare_openai_tools()
        
        # Basic validation that we get a list of tool definitions
        assert isinstance(tools, list)
        if tools:  # If there are any tools
            assert isinstance(tools[0], dict)
            assert "type" in tools[0]
            assert tools[0]["type"] == "function"
            assert "function" in tools[0] 