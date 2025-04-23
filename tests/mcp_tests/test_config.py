import json
from unittest.mock import mock_open, patch

import pytest

# Target module
from src.mcp_code import config as mcp_config
from src.mcp_code.mcp_client.transport.stdio.stdio_server_parameters import StdioServerParameters

# Sample valid config data
VALID_CONFIG_DATA = {
    "mcpServers": {
        "server1": {"command": "/path/to/server1", "args": ["--port", "8080"], "env": {"VAR1": "value1"}},
        "server2": {
            "command": "/path/to/server2"
            # Optional args and env are missing
        },
    }
}

VALID_CONFIG_JSON = json.dumps(VALID_CONFIG_DATA)


@pytest.mark.asyncio
async def test_load_config_success_full():
    """Test successful loading with command, args, and env."""
    m = mock_open(read_data=VALID_CONFIG_JSON)
    with patch("builtins.open", m):
        config = await mcp_config.load_config("dummy_path.json")
        server_config = config.get("mcpServers", {}).get("server1")
        assert server_config["command"] == "/path/to/server1"
        assert server_config["args"] == ["--port", "8080"]
        assert server_config["env"] == {"VAR1": "value1"}


@pytest.mark.asyncio
async def test_load_config_success_minimal():
    """Test successful loading with only command specified."""
    m = mock_open(read_data=VALID_CONFIG_JSON)
    with patch("builtins.open", m):
        config = await mcp_config.load_config("dummy_path.json")
        server_config = config.get("mcpServers", {}).get("server2")
        assert server_config["command"] == "/path/to/server2"
        assert (
            "args" not in server_config
        )  # Or assert server_config.get("args") is None/[] depending on expected parsing
        assert "env" not in server_config  # Or assert server_config.get("env") is None


@pytest.mark.asyncio
async def test_load_config_file_not_found():
    """Test handling when the config file doesn't exist."""
    m = mock_open()
    m.side_effect = FileNotFoundError("File not found")
    with patch("builtins.open", m):
        with pytest.raises(FileNotFoundError, match="Configuration file not found: dummy_path.json"):
            await mcp_config.load_config("dummy_path.json")


@pytest.mark.asyncio
async def test_load_config_invalid_json():
    """Test handling invalid JSON in the config file."""
    m = mock_open(read_data="{invalid json")
    with patch("builtins.open", m):
        with pytest.raises(
            json.JSONDecodeError,
            match="Invalid JSON in configuration file: Expecting property name enclosed in double quotes",
        ):
            await mcp_config.load_config("dummy_path.json")


@pytest.mark.asyncio
async def test_load_config_server_not_found():
    """Test handling when the server name is not in the config."""
    # This test is less relevant now as load_config doesn't look for server names.
    # The check happens in run_command. We can keep it to ensure the config loads.
    m = mock_open(read_data=VALID_CONFIG_JSON)
    with patch("builtins.open", m):
        config = await mcp_config.load_config("dummy_path.json")
        assert "server1" in config.get("mcpServers", {})
        assert "unknown_server" not in config.get("mcpServers", {})


@pytest.mark.asyncio
async def test_load_config_mcp_servers_key_missing():
    """Test handling when the top-level 'mcpServers' key is missing."""
    # Similar to above, load_config loads the dict, consumer checks keys.
    invalid_config_json = json.dumps({"otherKey": {}})
    m = mock_open(read_data=invalid_config_json)
    with patch("builtins.open", m):
        config = await mcp_config.load_config("dummy_path.json")
        assert "mcpServers" not in config


@pytest.mark.asyncio
async def test_load_config_command_key_missing():
    """Test handling when the 'command' key is missing for a server."""
    config_missing_command = {"mcpServers": {"server1": {"args": ["--port", "8080"], "env": {"VAR1": "value1"}}}}
    invalid_config_json = json.dumps(config_missing_command)
    m = mock_open(read_data=invalid_config_json)
    with patch("builtins.open", m):
        config = await mcp_config.load_config("dummy_path.json")
        assert "command" not in config.get("mcpServers", {}).get("server1", {})


# TODO: Add tests
