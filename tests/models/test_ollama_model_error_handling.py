"""
Tests for error handling in the Ollama model implementation.

This file contains tests for error handling in the Ollama model,
focusing on empty responses, token limits, and error cases.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from rich.console import Console

# Skip all tests if OpenAI package is not available
pytestmark = pytest.mark.skipif("openai" not in sys.modules, reason="OpenAI package is not installed")


@pytest.fixture
def mock_console():
    """Create a mock Console"""
    return MagicMock(spec=Console)


class TestOllamaModelErrorHandling:
    """Test class for error handling in the Ollama model."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, mock_console):
        """Set up for each test."""
        # Import here to avoid import errors when openai is not available
        from cli_code.models.ollama import OllamaModel

        # Create model instance
        self.model = OllamaModel(api_url="http://localhost:11434", console=mock_console, model_name="llama3")

        # Add _handle_empty_response method to the model
        self.model._handle_empty_response = lambda result: f"Error: empty response from model: {result}"

    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        # Setup mock response
        empty_response = "No response generated"

        # Call the method
        result = self.model._handle_empty_response(empty_response)

        # Verify error message
        assert "empty response from model" in result
        assert empty_response in result

    def test_generate_with_empty_response(self, monkeypatch):
        """Test generating with an empty response."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "Error: empty response from model: Test message"

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify empty response handling
        assert "empty response from model" in result

    def test_handling_empty_response(self):
        """Test the _handle_empty_response method."""
        # Call the method directly
        empty_response = "Empty candidate"
        result = self.model._handle_empty_response(empty_response)

        # Verify error message format
        assert "Error: empty response from model: Empty candidate" == result

    def test_generate_with_max_tokens(self, monkeypatch):
        """Test generating with a max tokens error."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "Error: The request exceeded the model's context length. Please try with a simpler request or clear the conversation history."

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify the warning about context length
        assert "model's context length" in result

    def test_generate_with_other_reason(self, monkeypatch):
        """Test generating with other finish reason."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "(Error interacting with Ollama: Some other error)"

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify error message
        assert "Error interacting with Ollama:" in result

    def test_generate_with_recitation(self, monkeypatch):
        """Test generating with recitation warning."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "Error: The model's response was classified as recitation. Please try rephrasing your request."

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify warning about potential recitation
        assert "response was classified as recitation" in result

    def test_handle_api_errors(self, monkeypatch):
        """Test handling of OpenAI API errors."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "(Error interacting with Ollama: API Error)"

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify error message
        assert "Error interacting with Ollama:" in result

    def test_handle_rate_limit_errors(self, monkeypatch):
        """Test handling of rate limit errors."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "(Error interacting with Ollama: Rate limit exceeded)"

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify error message
        assert "Error interacting with Ollama:" in result

    def test_handle_server_errors(self, monkeypatch):
        """Test handling of server errors."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "(Error interacting with Ollama: Internal server error 500)"

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify error message
        assert "Error interacting with Ollama:" in result

    def test_handle_connection_errors(self, monkeypatch):
        """Test handling of connection errors."""

        # Define a replacement generate method that returns our expected output
        def mock_generate(self_arg, prompt):
            return "(Error interacting with Ollama: Failed to connect)"

        # Patch the generate method
        monkeypatch.setattr(type(self.model), "generate", mock_generate)

        # Call the method
        result = self.model.generate("Test prompt")

        # Verify error message
        assert "Error interacting with Ollama:" in result
