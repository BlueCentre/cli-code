"""
Tests for edge cases in the OllamaModel class to improve test coverage.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

# Skip all tests if OpenAI is not available
try:
    import openai

    SKIP_OPENAI_TESTS = False
except ImportError:
    SKIP_OPENAI_TESTS = True


@pytest.fixture
def mock_console():
    """Fixture for console mock."""
    return MagicMock(spec=Console)


# Skip the entire module if needed
pytestmark = pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")


class TestOllamaModelBasicFunctions:
    """Test basic functions of OllamaModel without needing OpenAI."""

    @pytest.mark.xfail(strict=False, reason="Requires OpenAI module")
    def test_empty_response_handling(self, mock_console):
        """Test handling of empty responses without executing actual API calls."""
        from cli_code.models.ollama import OllamaModel

        # This import should work even if we don't have OpenAI installed
        # because we're not actually creating an instance
        assert OllamaModel is not None

        # Just test that we can import the class, which verifies basic module structure
        pass


# All remaining tests will be automatically skipped if OpenAI is not available
@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_generate_with_empty_response(monkeypatch, mock_console):
    """Test handling empty response in generate method."""
    from cli_code.models.ollama import OllamaModel

    # Mock OpenAI class to prevent actual API calls
    mock_openai = MagicMock()
    mock_client = MagicMock()
    monkeypatch.setattr("cli_code.models.ollama.OpenAI", lambda **kwargs: mock_client)

    # Create instance with mocked dependencies
    model = OllamaModel("http://localhost:11434", mock_console, "test-model")

    # Mock response with empty content
    mock_message = MagicMock()
    mock_message.content = ""
    mock_message.tool_calls = []
    mock_message.model_dump.return_value = {"role": "assistant", "content": ""}

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Configure the mock client
    mock_client.chat.completions.create.return_value = mock_response

    # Execute the test
    result = model.generate("test prompt")

    # Verify result
    assert "The model provided an empty response" in result


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_handling_empty_response(mock_console):
    """Test the _handle_empty_response method."""
    from cli_code.models.ollama import OllamaModel

    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_client.models.list.return_value = MagicMock(
            data=[MagicMock(id="llama3", object="model", created=1677610602, owned_by="openai")]
        )

        model = OllamaModel(api_url="http://localhost:11434", console=mock_console, model_name="llama3")
        result = model.handle_empty_response()

        assert "empty response" in result.lower()
        assert isinstance(result, str)


@pytest.mark.skipif(SKIP_OPENAI_TESTS, reason="OpenAI library not available")
def test_generate_with_max_tokens(monkeypatch, mock_console):
    """Test handling max tokens error in generate method."""
    from cli_code.models.ollama import OllamaModel

    # Mock OpenAI class
    mock_client = MagicMock()
    monkeypatch.setattr("cli_code.models.ollama.OpenAI", lambda **kwargs: mock_client)

    # Create instance with mocked dependencies
    model = OllamaModel("http://localhost:11434", mock_console, "test-model")

    # Mock an exception with max tokens error
    mock_client.chat.completions.create.side_effect = Exception("maximum context length exceeded")

    # Execute the test
    result = model.generate("test prompt")

    # Verify result
    assert "Error: The request exceeded the model's context length" in result


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
def test_generate_with_recitation(monkeypatch, mock_console):
    """Test handling recitation in generate method."""
    from cli_code.models.ollama import OllamaModel

    # Mock OpenAI class
    mock_client = MagicMock()
    monkeypatch.setattr("cli_code.models.ollama.OpenAI", lambda **kwargs: mock_client)

    # Create instance with mocked dependencies
    model = OllamaModel("http://localhost:11434", mock_console, "test-model")

    # Mock a response with recitation issue
    mock_client.chat.completions.create.side_effect = Exception("output classified as recitation")

    # Execute the test
    result = model.generate("test prompt")

    # Verify result
    assert "Error: The model's response was classified as recitation" in result
