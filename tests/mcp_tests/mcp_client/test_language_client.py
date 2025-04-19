# tests/mcp_tests/mcp_client/test_language_client.py
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_code.mcp_client.language_client import LanguageClient

# Force asyncio only for all tests in this file
# pytestmark = [pytest.mark.asyncio] # Remove this global mark


@pytest.fixture
def server_params():
    """Fixture for basic server parameters."""
    return {"name": "test_server", "type": "stdio"}


@pytest.fixture
def mock_streams():
    """Fixture for mock read/write streams."""
    return (AsyncMock(), AsyncMock())


@pytest.fixture
def server_params_no_name():
    """Fixture for server parameters without a name."""
    # Example: Use a type known to exist but omit 'name'
    return {"type": "stdio", "command": "some_cmd"}


def test_language_client_init(server_params):
    """Test LanguageClient initialization without streams."""
    client = LanguageClient(server_params)
    assert client.server_params == server_params
    assert client.server_name == "test_server"
    assert not client.connected
    assert client.streams is None


def test_language_client_init_with_streams(server_params, mock_streams):
    """Test LanguageClient initialization with pre-existing streams."""
    client = LanguageClient(server_params, streams=mock_streams)
    assert client.server_params == server_params
    assert client.server_name == "test_server"
    assert not client.connected
    assert client.streams == mock_streams


def test_language_client_init_no_name(server_params_no_name):
    """Test LanguageClient initialization when server_params lacks a 'name'."""
    client = LanguageClient(server_params_no_name)
    assert client.server_params == server_params_no_name
    assert client.server_name == "unknown"  # Verify default name
    assert not client.connected
    assert client.streams is None


# Mark only the async tests explicitly
@pytest.mark.asyncio
async def test_language_client_aenter(server_params):
    """Test the __aenter__ method."""
    client = LanguageClient(server_params)
    assert not client.connected
    with patch("logging.debug") as mock_log_debug:
        returned_self = await client.__aenter__()
        assert client.connected
        assert returned_self is client
        mock_log_debug.assert_called_once_with(f"LanguageClient connected to server: {client.server_name}")


@pytest.mark.asyncio
async def test_language_client_aexit(server_params):
    """Test the __aexit__ method."""
    client = LanguageClient(server_params)
    # Simulate entering the context
    await client.__aenter__()
    assert client.connected

    with patch("logging.debug") as mock_log_debug:
        # Test exiting without exception
        handled = await client.__aexit__(None, None, None)
        assert not client.connected
        assert not handled  # Should not suppress exceptions
        mock_log_debug.assert_called_once_with(f"LanguageClient disconnecting from server: {client.server_name}")

    # Reset state and test exiting with an exception
    await client.__aenter__()
    assert client.connected
    mock_log_debug.reset_mock()
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        exc_type, exc_val, exc_tb = type(e), e, e.__traceback__
        with patch("logging.debug") as mock_log_debug_exc:
            handled_exc = await client.__aexit__(exc_type, exc_val, exc_tb)
            assert not client.connected
            assert not handled_exc  # Still should not suppress
            mock_log_debug_exc.assert_called_once_with(
                f"LanguageClient disconnecting from server: {client.server_name}"
            )


@pytest.mark.asyncio
async def test_language_client_context_manager(server_params):
    """Test using LanguageClient as an async context manager."""
    client = LanguageClient(server_params)

    with patch("logging.debug") as mock_log_debug:
        async with client as returned_client:
            assert client.connected
            assert returned_client is client
            mock_log_debug.assert_called_once_with(f"LanguageClient connected to server: {client.server_name}")
            # Check logs inside the context

        # After exiting the context
        assert not client.connected
        # Check that disconnect log was called (second call overall)
        assert mock_log_debug.call_count == 2
        mock_log_debug.assert_called_with(f"LanguageClient disconnecting from server: {client.server_name}")
