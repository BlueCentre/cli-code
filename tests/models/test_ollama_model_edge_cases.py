"""
Tests for edge cases in the OllamaModel class to improve test coverage.
Updated with additional fixes to ensure all tests pass.
"""

import json
import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

# Skip tests if openai is not available
try:
    from openai import OpenAIError
    from openai.types.chat import ChatCompletion

    SKIP_OPENAI_TESTS = False
except ImportError:
    SKIP_OPENAI_TESTS = True

# Mark the entire module to be skipped if openai is not available
pytestmark = pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")


@pytest.fixture
def mock_console():
    """Return a Mock rich console"""
    return MagicMock(spec=Console)


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_empty_response_handling(mock_console):
    """Test handling of empty responses without executing actual API calls."""
    from cli_code.models.ollama import OllamaModel

    # Create instance with mocked console
    model = OllamaModel(api_url="http://localhost:11434", console=mock_console, model_name="llama3")

    # Check if the model has the expected method
    assert hasattr(model, "_handle_empty_response")

    # Call the method directly
    result = model._handle_empty_response()

    # Verify result
    assert "empty response" in result.lower()


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_generate_with_empty_response(mock_console):
    """Test generate method's handling of an empty response."""
    from cli_code.models.ollama import OllamaModel

    # Create model instance
    model = OllamaModel(api_url="http://localhost:11434", console=mock_console, model_name="llama3")

    # Mock the OpenAI client creation so it returns our mocked client
    with patch.object(model, "client") as mock_client:
        # Create a mock response with empty content
        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.content = None  # Empty content
        mock_response.role = "assistant"
        mock_response.model_dump = MagicMock(return_value={"role": "assistant", "content": None})

        # Add the tool_calls attribute with value of None or empty list
        mock_response.tool_calls = None

        mock_completion.return_value.choices = [MagicMock(message=mock_response)]
        mock_client.chat.completions.create = mock_completion

        # Call generate
        response = model.generate("test prompt")

        # Check if the response mentions empty response
        assert "empty response" in response.lower()


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_handling_empty_response(mock_console):
    """Test _handle_empty_response method."""
    from cli_code.models.ollama import OllamaModel

    # Create model instance
    model = OllamaModel(api_url="http://localhost:11434", console=mock_console, model_name="llama3")

    # Call method directly
    result = model._handle_empty_response()

    # Verify result
    assert isinstance(result, str)
    assert "empty response" in result.lower()


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_generate_with_max_tokens(mock_console):
    """Test generate method handling max token errors."""
    from cli_code.models.ollama import OllamaModel

    # Create model instance
    model = OllamaModel(api_url="http://localhost:11434", console=mock_console, model_name="llama3")

    # Mock the OpenAI client creation so it returns our mocked client
    with patch.object(model, "client") as mock_client:
        # Create a mock response that will trigger a context length error
        mock_completion = MagicMock()
        mock_client.chat.completions.create = mock_completion

        # Set up a context length error with a regular Exception
        mock_completion.side_effect = Exception("This model's maximum context length is 4097 tokens")

        # Call generate
        response = model.generate("Tell me a long story")

        # Check if the response contains expected content
        assert isinstance(response, str)
        assert re.search("context length|token limit", response.lower())


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_generate_with_other_reason(monkeypatch, mock_console):
    """Test handling other errors in generate method."""
    from cli_code.models.ollama import OllamaModel

    # Mock OpenAI class
    mock_client = MagicMock()
    monkeypatch.setattr("cli_code.models.ollama.OpenAI", lambda **kwargs: mock_client)

    # Create instance with mocked dependencies
    model = OllamaModel("http://localhost:11434", mock_console, "test-model")

    # Mock a general exception
    mock_client.chat.completions.create.side_effect = Exception("Some other error")

    # Execute the test
    result = model.generate("test prompt")

    # Verify result
    assert "Error interacting with Ollama:" in result


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_generate_with_recitation(mock_console):
    """Test generate method when response is detected as recitation."""
    from cli_code.models.ollama import OllamaModel

    # Create model instance
    model = OllamaModel(api_url="http://localhost:11434", console=mock_console, model_name="llama3")

    # Mock the OpenAI client creation so it returns our mocked client
    with patch.object(model, "client") as mock_client:
        # Create a mock response that will trigger a recitation error
        mock_completion = MagicMock()
        mock_client.chat.completions.create = mock_completion

        # Set up a recitation error with a regular Exception
        mock_completion.side_effect = Exception("This response contains recitation which is not allowed")

        # Call generate - OllamaModel handles recitation errors directly in generate, no need to mock _handle_recitation_error
        response = model.generate("Tell me how to break the law")

        # Check if the response contains expected content
        assert isinstance(response, str)
        assert "error" in response.lower()
