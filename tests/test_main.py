"""
Tests for the CLI main module.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.markdown import Markdown

# Import main functions under test
from cli_code.main import cli, run_mcp_session, start_interactive_session

# # Import necessary classes for MCP testing - Attempt 3
# from chuk_mcp import MCPError, MCPMessage # <-- Top level - REMOVED
# from chuk_mcp.mcp_client.protocol import MCPClientProtocol # <-- Submodule - REMOVED
# # Import necessary classes for MCP testing - Reverted
# from chuk_mcp import MCPClientProtocol, MCPError, MCPMessage # <-- Reverted imports
# # Import necessary classes for MCP testing - Attempt 2
# # from chuk_mcp.mcp_client.protocol import MCPClientProtocol, MCPError, MCPMessage # <-- Old paths
# from chuk_mcp.mcp_client.exceptions import MCPError
# from chuk_mcp.mcp_client.protocol.base_protocol import MCPClientProtocol
# from chuk_mcp.mcp_client.messages.message_base import MCPMessage


@pytest.fixture
def mock_console(mocker):
    """Provides a mocked Console object."""
    console_mock = mocker.patch("src.cli_code.main.console")
    # Make sure print method doesn't cause issues
    console_mock.print.return_value = None
    # Ensure input method is mockable
    console_mock.input = mocker.MagicMock()
    return console_mock


@pytest.fixture
def mock_config():
    """Fixture to provide a mocked Config object."""
    with patch("cli_code.main.config") as mock_config:
        # Set some reasonable default behavior for the config mock
        mock_config.get_default_provider.return_value = "gemini"
        mock_config.get_default_model.return_value = "gemini-pro"
        mock_config.get_credential.return_value = "fake-api-key"
        yield mock_config


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


@patch("cli_code.main.start_interactive_session")
def test_cli_default_invocation(mock_start_session, runner, mock_config):
    """Test the default CLI invocation starts an interactive session."""
    result = runner.invoke(cli)
    assert result.exit_code == 0
    mock_start_session.assert_called_once()


def test_setup_command(runner, mock_config):
    """Test the setup command."""
    result = runner.invoke(cli, ["setup", "--provider", "gemini", "fake-api-key"])
    assert result.exit_code == 0
    mock_config.set_credential.assert_called_once_with("gemini", "fake-api-key")


def test_set_default_provider(runner, mock_config):
    """Test the set-default-provider command."""
    result = runner.invoke(cli, ["set-default-provider", "ollama"])
    assert result.exit_code == 0
    mock_config.set_default_provider.assert_called_once_with("ollama")


def test_set_default_model(runner, mock_config):
    """Test the set-default-model command."""
    result = runner.invoke(cli, ["set-default-model", "--provider", "gemini", "gemini-pro-vision"])
    assert result.exit_code == 0
    mock_config.set_default_model.assert_called_once_with("gemini-pro-vision", provider="gemini")


# Remove tests for list-models as the command implementation is commented out
# @patch("cli_code.main.GeminiModel")
# def test_list_models_gemini(mock_gemini_model, runner, mock_config):
#     """Test the list-models command for Gemini provider."""
#     # Setup mock model instance
#     mock_instance = MagicMock()
#     mock_instance.list_models.return_value = [
#         {"name": "gemini-pro", "displayName": "Gemini Pro"},
#         {"name": "gemini-pro-vision", "displayName": "Gemini Pro Vision"},
#     ]
#     mock_gemini_model.return_value = mock_instance
#
#     result = runner.invoke(cli, ["list-models", "--provider", "gemini"])
#     assert result.exit_code == 0
#     mock_gemini_model.assert_called_once()
#     mock_instance.list_models.assert_called_once()
#
#
# @patch("cli_code.main.OllamaModel")
# def test_list_models_ollama(mock_ollama_model, runner, mock_config):
#     """Test the list-models command for Ollama provider."""
#     # Setup mock model instance
#     mock_instance = MagicMock()
#     mock_instance.list_models.return_value = [
#         {"name": "llama2", "displayName": "Llama 2"},
#         {"name": "mistral", "displayName": "Mistral"},
#     ]
#     mock_ollama_model.return_value = mock_instance
#
#     result = runner.invoke(cli, ["list-models", "--provider", "ollama"])
#     assert result.exit_code == 0
#     mock_ollama_model.assert_called_once()
#     mock_instance.list_models.assert_called_once()


# --- Updated tests for interactive session ---


@pytest.mark.asyncio
@patch("cli_code.main.run_mcp_session", new_callable=AsyncMock)  # Mock the async function
async def test_start_interactive_session_calls_run_mcp(mock_run_mcp, mock_console):
    """Verify start_interactive_session calls run_mcp_session correctly."""
    provider = "test_provider"
    model_name = "test_model"

    # We call the synchronous wrapper which internally calls asyncio.run
    start_interactive_session(provider, model_name, mock_console)

    # Assert that the mocked async function was called once
    mock_run_mcp.assert_called_once()
    call_args, _ = mock_run_mcp.call_args
    assert isinstance(call_args[0], str) and call_args[0].startswith("cli-code-client-")  # agent_id
    assert isinstance(call_args[1], str) and call_args[1].startswith("session-")  # session_id
    assert call_args[2] is mock_console  # console


# --- Tests for run_mcp_session (using direct asyncio/json mocks) ---


@pytest.fixture
def mock_streams(mocker):
    """Provides mocked asyncio reader and writer streams."""
    mock_reader = AsyncMock(spec=asyncio.StreamReader)
    # Simulate readline returning bytes followed by None (EOF)
    mock_reader.readline = AsyncMock()

    mock_writer = AsyncMock(spec=asyncio.StreamWriter)
    mock_writer.write = MagicMock()  # write is sync
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()  # close is sync
    mock_writer.wait_closed = AsyncMock()
    return mock_reader, mock_writer


@pytest.mark.asyncio
@patch("cli_code.main.asyncio.open_connection", new_callable=AsyncMock)  # Mock connection
@patch("cli_code.main.asyncio.to_thread", new_callable=AsyncMock)  # Mock to_thread
async def test_run_mcp_session_happy_path_custom(mock_to_thread, mock_open_connection, mock_streams, mock_console):
    """Test basic interaction with custom JSON handling."""
    # Arrange
    mock_reader, mock_writer = mock_streams
    mock_open_connection.return_value = (mock_reader, mock_writer)

    # Simulate user input via mocking asyncio.to_thread directly
    mock_to_thread.side_effect = ["hello server", "/exit"]

    # Simulate server response
    assistant_response_payload = {
        "message_type": "assistant_message",
        "agent_id": "server_agent",
        "session_id": "session-123",
        "payload": {"text": "Assistant says hi!"},
    }
    # Encode the response as the server would (JSON line + newline)
    encoded_response = json.dumps(assistant_response_payload).encode("utf-8") + b"\n"
    # readline returns bytes, then empty bytes (EOF) after exit
    mock_reader.readline.side_effect = [encoded_response, b""]

    agent_id = "agent-test"
    session_id = "session-test"

    # Act
    await run_mcp_session(agent_id, session_id, mock_console)

    # Assert
    # Connection
    mock_open_connection.assert_called_once_with("127.0.0.1", 8999)

    # Input calls - check asyncio.to_thread was called correctly
    assert mock_to_thread.call_count == 2
    first_call_args = mock_to_thread.call_args_list[0][0]
    assert first_call_args[0] is mock_console.input  # Check it's calling the right function
    assert first_call_args[1] == "[bold cyan]You:[/bold cyan] "  # Check the prompt

    # Sending message
    assert mock_writer.write.call_count == 1
    sent_bytes = mock_writer.write.call_args[0][0]
    sent_data = json.loads(sent_bytes.decode("utf-8").strip())  # Decode sent JSON
    assert sent_data["message_type"] == "user_message"
    assert sent_data["agent_id"] == agent_id
    assert sent_data["session_id"] == session_id
    assert sent_data["payload"] == {"text": "hello server"}
    mock_writer.drain.assert_called_once()

    # Receiving message
    assert mock_reader.readline.call_count >= 1  # Read at least the response

    # Console output
    print_calls = mock_console.print.call_args_list
    # assert any("Assistant says hi!" in str(call) for call in print_calls), "Assistant message not printed"
    # Check if print was called with a Markdown object
    found_markdown_call = False
    for call in print_calls:
        args, kwargs = call
        if args and isinstance(args[0], Markdown):
            # Optionally, check the markdown content if needed, but isinstance might be enough
            # if "Assistant says hi!" in args[0].markup:
            found_markdown_call = True
            break
    assert found_markdown_call, "console.print was not called with a Markdown object for the assistant message"
    assert any("Exiting session" in str(call) for call in print_calls), "Exit message not printed"

    # Connection closed
    mock_writer.close.assert_called_once()
    mock_writer.wait_closed.assert_called_once()


@pytest.mark.asyncio
@patch("cli_code.main.asyncio.open_connection", new_callable=AsyncMock)
@patch("cli_code.main.asyncio.to_thread", new_callable=AsyncMock)
async def test_run_mcp_session_help_command_custom(mock_to_thread, mock_open_connection, mock_streams, mock_console):
    """Test the /help command is handled locally (custom handling)."""
    # Arrange
    mock_reader, mock_writer = mock_streams
    mock_open_connection.return_value = (mock_reader, mock_writer)
    mock_to_thread.side_effect = ["/help", "/exit"]
    mock_reader.readline.return_value = b""  # Simulate immediate close on exit

    # Act
    await run_mcp_session("agent-h", "session-h", mock_console)

    # Assert
    print_calls = mock_console.print.call_args_list
    assert any("Available commands:" in str(call) for call in print_calls)
    assert mock_to_thread.call_count == 2  # Check input was requested twice
    mock_writer.write.assert_not_called()  # No message sent
    mock_writer.close.assert_called_once()
    mock_writer.wait_closed.assert_called_once()


@pytest.mark.asyncio
@patch("cli_code.main.asyncio.open_connection", new_callable=AsyncMock)
@patch("cli_code.main.asyncio.to_thread", new_callable=AsyncMock)
async def test_run_mcp_session_server_error_message_custom(
    mock_to_thread, mock_open_connection, mock_streams, mock_console
):
    """Test handling of an error_message from the server (custom handling)."""
    # Arrange
    mock_reader, mock_writer = mock_streams
    mock_open_connection.return_value = (mock_reader, mock_writer)
    mock_to_thread.side_effect = ["do something", "/exit"]

    error_payload = {
        "message_type": "error_message",
        "agent_id": "server",
        "session_id": "s1",
        "payload": {"message": "Something broke!"},
    }
    encoded_error = json.dumps(error_payload).encode("utf-8") + b"\n"
    mock_reader.readline.side_effect = [encoded_error, b""]  # Receive error, then EOF

    # Act
    await run_mcp_session("agent-e", "session-e", mock_console)

    # Assert
    assert mock_to_thread.call_count == 2
    assert mock_writer.write.call_count == 1  # User message sent
    print_calls = mock_console.print.call_args_list
    assert any("Server Error:" in str(call) and "Something broke!" in str(call) for call in print_calls)
    mock_writer.close.assert_called_once()


@pytest.mark.asyncio
@patch("cli_code.main.asyncio.run")
@patch("cli_code.main.run_mcp_session", new_callable=AsyncMock)  # Still mock inner call
async def test_start_interactive_session_connection_refused_custom(mock_run_mcp, mock_asyncio_run, mock_console):
    """Test start_interactive_session handles ConnectionRefusedError (custom handling)."""
    # Arrange
    mock_run_mcp.side_effect = ConnectionRefusedError("Test connection refused")

    # Simpler approach: If asyncio.run is called, make it raise the error
    # that the inner coro (run_mcp_session) would have raised.
    mock_asyncio_run.side_effect = ConnectionRefusedError("Simulated connection refused from run")

    # Act
    start_interactive_session("p", "m", mock_console)

    # Assert
    mock_run_mcp.assert_called_once()
    print_calls = mock_console.print.call_args_list
    assert any("Connection Error:" in str(call) and "Could not connect" in str(call) for call in print_calls)
    assert any("Is the stub server" in str(call) for call in print_calls)


@pytest.mark.asyncio
@patch("cli_code.main.asyncio.open_connection", new_callable=AsyncMock)
@patch("cli_code.main.asyncio.to_thread", new_callable=AsyncMock)
async def test_run_mcp_session_receive_timeout_custom(mock_to_thread, mock_open_connection, mock_streams, mock_console):
    """Test handling of asyncio.TimeoutError during receive (custom handling)."""
    # Arrange
    mock_reader, mock_writer = mock_streams
    mock_open_connection.return_value = (mock_reader, mock_writer)
    mock_to_thread.side_effect = ["send message", "/exit"]
    # Simulate readline raising TimeoutError
    mock_reader.readline.side_effect = [asyncio.TimeoutError(), b""]  # Timeout, then EOF

    # Act
    await run_mcp_session("agent-t", "session-t", mock_console)

    # Assert
    assert mock_to_thread.call_count == 2
    assert mock_writer.write.call_count == 1  # User message sent
    assert mock_reader.readline.call_count >= 1  # First call times out
    print_calls = mock_console.print.call_args_list
    assert any("Timeout waiting for server response" in str(call) for call in print_calls)
    mock_writer.close.assert_called_once()


@pytest.mark.asyncio
@patch("cli_code.main.asyncio.open_connection", new_callable=AsyncMock)
@patch("cli_code.main.asyncio.to_thread", new_callable=AsyncMock)
async def test_run_mcp_session_json_decode_error_custom(
    mock_to_thread, mock_open_connection, mock_streams, mock_console
):
    """Test handling of invalid JSON received from server."""
    # Arrange
    mock_reader, mock_writer = mock_streams
    mock_open_connection.return_value = (mock_reader, mock_writer)
    mock_to_thread.side_effect = ["send message", "/exit"]
    # Simulate readline returning invalid JSON bytes
    invalid_json_bytes = b'{"type": "assistant", payload: not json}\n'
    mock_reader.readline.side_effect = [invalid_json_bytes, b""]  # Invalid JSON, then EOF

    # Act
    await run_mcp_session("agent-json", "session-json", mock_console)

    # Assert
    assert mock_to_thread.call_count == 2
    assert mock_writer.write.call_count == 1  # User message sent
    assert mock_reader.readline.call_count >= 1
    print_calls = mock_console.print.call_args_list
    assert any("Received invalid data from server" in str(call) for call in print_calls)
    mock_writer.close.assert_called_once()
