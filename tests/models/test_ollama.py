"""
Tests for the OllamaModel class.
"""

from types import SimpleNamespace  # Import SimpleNamespace

import pytest

# Import directly to ensure coverage
from src.cli_code.models.ollama import OllamaModel


@pytest.fixture
def mock_console(mocker):
    """Provides a mocked Console object."""
    mock_console = mocker.MagicMock()
    mock_console.status.return_value.__enter__.return_value = None
    mock_console.status.return_value.__exit__.return_value = None
    return mock_console


@pytest.fixture
def ollama_model_with_mocks(mocker, mock_console):
    """Provides an initialized OllamaModel instance with essential mocks."""
    # Mock OpenAI client
    mock_openai = mocker.patch("src.cli_code.models.ollama.OpenAI")
    mock_client = mocker.MagicMock()
    mock_openai.return_value = mock_client

    # Mock os path functions
    mocker.patch("os.path.isdir", return_value=False)
    mocker.patch("os.path.isfile", return_value=False)

    # Mock get_tool for initial context
    mock_tool = mocker.MagicMock()
    mock_tool.execute.return_value = "ls output"
    mocker.patch("src.cli_code.models.ollama.get_tool", return_value=mock_tool)

    # Mock count_tokens to avoid dependencies
    mocker.patch("src.cli_code.models.ollama.count_tokens", return_value=10)

    # Create model instance
    model = OllamaModel("http://localhost:11434", mock_console, "llama3")

    # Reset the client mocks after initialization to test specific functions
    mock_client.reset_mock()

    # Return the model and mocks for test assertions
    return {"model": model, "mock_openai": mock_openai, "mock_client": mock_client, "mock_tool": mock_tool}


def test_init(ollama_model_with_mocks):
    """Test initialization of OllamaModel."""
    model = ollama_model_with_mocks["model"]
    mock_openai = ollama_model_with_mocks["mock_openai"]

    # Check if OpenAI client was initialized correctly
    mock_openai.assert_called_once_with(base_url="http://localhost:11434", api_key="ollama")

    # Check model attributes
    assert model.api_url == "http://localhost:11434"
    assert model.model_name == "llama3"

    # Check history initialization (should have system message)
    assert len(model.history) == 1
    assert model.history[0]["role"] == "system"


def test_get_initial_context_with_ls_fallback(ollama_model_with_mocks):
    """Test getting initial context via ls when no .rules or README."""
    model = ollama_model_with_mocks["model"]
    mock_tool = ollama_model_with_mocks["mock_tool"]

    # Reset mock before specific test call, as __init__ also calls it
    mock_tool.execute.reset_mock()

    # Call method for testing
    context = model._get_initial_context()

    # Verify tool was used during *this* call
    mock_tool.execute.assert_called_once()

    # Check result content
    assert "Current directory contents" in context
    assert "ls output" in context


def test_add_and_clear_history(ollama_model_with_mocks):
    """Test adding messages to history and clearing it."""
    model = ollama_model_with_mocks["model"]

    # Add a test message
    test_message = {"role": "user", "content": "Test message"}
    model.add_to_history(test_message)

    # Verify message was added (in addition to system message)
    assert len(model.history) == 2
    assert model.history[1] == test_message

    # Clear history
    model.clear_history()

    # Verify history was reset to just system message
    assert len(model.history) == 1
    assert model.history[0]["role"] == "system"


def test_list_models(ollama_model_with_mocks, mocker):
    """Test listing available models."""
    model = ollama_model_with_mocks["model"]
    mock_client = ollama_model_with_mocks["mock_client"]

    # Use SimpleNamespace for mock data to avoid mock interaction issues
    mock_model1 = SimpleNamespace(id="llama3", name="Llama 3")
    mock_model2 = SimpleNamespace(id="mistral", name="Mistral")

    # Simulate the response object containing a 'data' attribute with the list
    mock_models_list = SimpleNamespace(data=[mock_model1, mock_model2])

    # Configure client mock to return the simulated response object
    mock_client.models.list.return_value = mock_models_list

    # Call the method
    result = model.list_models()

    # Verify client method called
    mock_client.models.list.assert_called_once()

    # Verify result format matches the method implementation
    assert len(result) == 2
    assert result[0]["id"] == "llama3"
    assert result[0]["name"] == "Llama 3"
    assert result[1]["id"] == "mistral"


def test_generate_simple_response(ollama_model_with_mocks, mocker):
    """Test generating a simple text response."""
    model = ollama_model_with_mocks["model"]
    mock_client = ollama_model_with_mocks["mock_client"]

    # --- Corrected Mock Structure ---
    # 1. Mock the innermost ChatCompletionMessage
    mock_chat_completion_message = mocker.MagicMock()
    mock_chat_completion_message.content = "Test response"
    mock_chat_completion_message.tool_calls = None
    # Mock the model_dump method needed for adding to history
    mock_chat_completion_message.model_dump.return_value = {"role": "assistant", "content": "Test response"}

    # 2. Mock the Choice object containing the message
    mock_choice = mocker.MagicMock()
    mock_choice.message = mock_chat_completion_message
    # Add finish_reason if the code checks it
    mock_choice.finish_reason = "stop"

    # 3. Mock the top-level ChatCompletion object containing choices
    mock_completion = mocker.MagicMock()
    mock_completion.choices = [mock_choice]
    # --- End Corrected Mock Structure ---

    # Override the MAX_OLLAMA_ITERATIONS to ensure test completes with one step
    mocker.patch("src.cli_code.models.ollama.MAX_OLLAMA_ITERATIONS", 1)

    # Reset mock calls made during initialization
    mock_client.chat.completions.create.reset_mock()

    # Configure the client mock to return the structured completion
    mock_client.chat.completions.create.return_value = mock_completion

    # Mock tool preparation as it's not relevant here
    mocker.patch.object(model, "_prepare_openai_tools", return_value=None)

    # Call generate method
    result = model.generate("Test prompt")

    # Verify client method called
    mock_client.chat.completions.create.assert_called_once()

    # Check the final result (should be the text content now)
    assert result == "Test response"


def test_manage_ollama_context(ollama_model_with_mocks, mocker):
    """Test context management for Ollama models."""
    model = ollama_model_with_mocks["model"]

    # Directly modify the max tokens constant for testing
    mocker.patch("src.cli_code.models.ollama.OLLAMA_MAX_CONTEXT_TOKENS", 100)  # Small value to force truncation

    # Mock count_tokens to return large value
    count_tokens_mock = mocker.patch("src.cli_code.models.ollama.count_tokens")
    count_tokens_mock.return_value = 30  # Each message will be 30 tokens

    # Get original system message
    original_system = model.history[0]

    # Add many messages to force context truncation
    for i in range(10):  # 10 messages * 30 tokens = 300 tokens > 100 limit
        model.add_to_history({"role": "user", "content": f"Test message {i}"})

    # Verify history was truncated (should have fewer than 11 messages - system + 10 added)
    assert len(model.history) < 11

    # Verify system message was preserved at the beginning
    assert model.history[0]["role"] == "system"
    assert model.history[0] == original_system
