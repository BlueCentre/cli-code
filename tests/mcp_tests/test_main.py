from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

# Target module
from src.mcp_code import __main__ as mcp_main
from src.mcp_code.mcp_client.transport.stdio.stdio_server_parameters import StdioServerParameters

# TODO: Add test classes/functions


@pytest.mark.asyncio
async def test_main_success():
    """Test the main function happy path."""
    mock_read_stream = AsyncMock(spec=anyio.streams.memory.MemoryObjectReceiveStream)
    mock_write_stream = AsyncMock(spec=anyio.streams.memory.MemoryObjectSendStream)

    mock_server_params = StdioServerParameters(command="echo hello", cwd=".", env={})

    # Create an AsyncMock that behaves like an async context manager
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = (mock_read_stream, mock_write_stream)
    # __aexit__ needs to be an awaitable that returns something (can be None)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "src.mcp_code.__main__.load_config", new_callable=AsyncMock, return_value=mock_server_params
        ) as mock_load_config,
        patch("src.mcp_code.__main__.stdio_client", return_value=mock_context_manager) as mock_stdio_client,
        patch(
            "src.mcp_code.__main__.send_initialize", new_callable=AsyncMock, return_value={"success": True}
        ) as mock_send_initialize,
        patch("src.mcp_code.__main__.send_ping", new_callable=AsyncMock, return_value=True) as mock_send_ping,
        patch(
            "src.mcp_code.__main__.send_tools_list", new_callable=AsyncMock, return_value=["tool1"]
        ) as mock_send_tools_list,
        patch("builtins.print") as mock_print,
    ):
        # No need to configure __aenter__ here anymore, done above
        # mock_stdio_client.return_value.__aenter__.return_value = (mock_read_stream, mock_write_stream)

        await mcp_main.main()

        mock_load_config.assert_awaited_once_with("server_config.json", "sqlite")
        mock_stdio_client.assert_called_once_with(mock_server_params)
        mock_send_initialize.assert_awaited_once_with(mock_read_stream, mock_write_stream)
        mock_send_ping.assert_awaited_once_with(mock_read_stream, mock_write_stream)
        mock_send_tools_list.assert_awaited_once_with(mock_read_stream, mock_write_stream)

        # Check print calls for success messages
        mock_print.assert_any_call("We're connected!!!")
        mock_print.assert_any_call("Ping successful")
        mock_print.assert_any_call(["tool1"])


def test_run():
    """Test the synchronous run wrapper."""
    with (
        patch("src.mcp_code.__main__.anyio.run") as mock_anyio_run,
        patch("src.mcp_code.__main__.main", new_callable=MagicMock) as mock_main_async,
    ):
        mcp_main.run()
        mock_anyio_run.assert_called_once_with(mock_main_async)


# TODO: Add test for initialization failure
@pytest.mark.asyncio
async def test_main_init_failure():
    """Test the main function when initialization fails."""
    mock_read_stream = AsyncMock(spec=anyio.streams.memory.MemoryObjectReceiveStream)
    mock_write_stream = AsyncMock(spec=anyio.streams.memory.MemoryObjectSendStream)

    mock_server_params = StdioServerParameters(command="echo hello", cwd=".", env={})

    # Create an AsyncMock that behaves like an async context manager
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = (mock_read_stream, mock_write_stream)
    # __aexit__ needs to be an awaitable that returns something (can be None)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "src.mcp_code.__main__.load_config", new_callable=AsyncMock, return_value=mock_server_params
        ) as mock_load_config,
        patch("src.mcp_code.__main__.stdio_client", return_value=mock_context_manager) as mock_stdio_client,
        patch(
            "src.mcp_code.__main__.send_initialize", new_callable=AsyncMock, return_value=None
        ) as mock_send_initialize,
        patch("src.mcp_code.__main__.send_ping", new_callable=AsyncMock) as mock_send_ping,
        patch("src.mcp_code.__main__.send_tools_list", new_callable=AsyncMock) as mock_send_tools_list,
        patch("builtins.print") as mock_print,
    ):
        # No need to configure __aenter__ here anymore, done above
        # mock_stdio_client.return_value.__aenter__.return_value = (mock_read_stream, mock_write_stream)

        await mcp_main.main()

        mock_load_config.assert_awaited_once_with("server_config.json", "sqlite")
        mock_stdio_client.assert_called_once_with(mock_server_params)
        mock_send_initialize.assert_awaited_once_with(mock_read_stream, mock_write_stream)

        # Check that ping and tools list were NOT called
        mock_send_ping.assert_not_awaited()
        mock_send_tools_list.assert_not_awaited()

        # Check print call for failure message
        mock_print.assert_any_call("Server initialization failed")
