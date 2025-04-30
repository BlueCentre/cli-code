"""
Tests for the MCP configuration manager.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

from rich.console import Console

from src.cli_code.mcp.config import MCPConfigurationManager


class TestMCPConfigurationManager(unittest.TestCase):
    """Tests for the MCPConfigurationManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.console = MagicMock(spec=Console)
        # Create a temporary directory for config files
        self.temp_dir = tempfile.TemporaryDirectory()
        # Use a non-existent file path initially for some tests
        self.non_existent_config_path = os.path.join(self.temp_dir.name, "non_existent_config.json")
        # Create a temporary file for other tests
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir.name, suffix=".json")
        self.temp_file.close()
        self.config_path = self.temp_file.name

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    @patch.object(MCPConfigurationManager, "_load_config")
    def test_init_default_config_file(self, mock_load_config):
        """Test initialization uses default config file if none provided."""
        manager = MCPConfigurationManager(self.console)
        self.assertEqual(manager.config_file, MCPConfigurationManager.DEFAULT_CONFIG_FILE)
        mock_load_config.assert_called_once()

    @patch.object(MCPConfigurationManager, "_load_config")
    def test_init_specific_config_file(self, mock_load_config):
        """Test initialization uses specific config file."""
        specific_path = "/path/to/config.json"
        manager = MCPConfigurationManager(self.console, config_file=specific_path)
        self.assertEqual(manager.config_file, specific_path)
        mock_load_config.assert_called_once()

    def test_load_config_file_not_found(self):
        """Test loading configuration when the file does not exist."""
        config_manager = MCPConfigurationManager(self.console, self.non_existent_config_path)
        self.assertEqual(config_manager.config_data, {})
        # No error should be printed if file just doesn't exist
        self.console.print.assert_not_called()

    @patch("src.cli_code.mcp.config.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_config_read_error(self, mock_open, mock_exists):
        """Test loading configuration when file read fails."""
        # Simulate an IOError during the read operation within the context manager
        mock_open.side_effect = IOError("Permission denied")

        config_manager = MCPConfigurationManager(self.console, self.config_path)

        # Verify config_data is empty and warning is printed
        self.assertEqual(config_manager.config_data, {})
        self.console.print.assert_called_once_with("[yellow]Warning:[/] Failed to load config: Permission denied")

    def test_load_config_empty_file(self):
        """Test loading from an empty file."""
        # Write an empty JSON object to the file
        with open(self.config_path, "w") as f:
            f.write("{}")

        config_manager = MCPConfigurationManager(self.console, self.config_path)
        self.assertEqual(config_manager.config_data, {})

    def test_load_config_with_data(self):
        """Test loading configuration with data."""
        test_config = {"mcpServers": {"test-server": {"url": "http://test-server", "api_key": "test-key"}}}

        with open(self.config_path, "w") as f:
            json.dump(test_config, f)

        config_manager = MCPConfigurationManager(self.console, self.config_path)
        self.assertEqual(config_manager.config_data, test_config)

    def test_load_config_invalid_json(self):
        """Test loading with invalid JSON."""
        with open(self.config_path, "w") as f:
            f.write("invalid json")

        config_manager = MCPConfigurationManager(self.console, self.config_path)
        self.assertEqual(config_manager.config_data, {})
        self.console.print.assert_called_once_with(f"[yellow]Warning:[/] Invalid JSON in {self.config_path}")

    def test_save_config(self):
        """Test saving configuration."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {"mcpServers": {"test-server": {"url": "http://test-server"}}}

        result = config_manager.save_config()
        self.assertTrue(result)

        with open(self.config_path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data, config_manager.config_data)

    @patch("builtins.open", new_callable=mock_open)
    def test_save_config_write_error(self, mock_open):
        """Test saving configuration when file write fails."""
        # Make the file handle raise an IOError on write
        mock_open.return_value.write.side_effect = IOError("Disk full")

        # Need to load config first to make the console print call count match
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {"test": "data"}

        result = config_manager.save_config()

        self.assertFalse(result)
        # The first call is the warning from load_config if it fails (which it doesn't here if mock_open is basic)
        # The second call is the error from save_config
        self.console.print.assert_called_with("[red]Error:[/] Failed to save config: Disk full")
        # Check it was called exactly once *with these specific args*
        calls = [
            c for c in self.console.print.call_args_list if c == call("[red]Error:[/] Failed to save config: Disk full")
        ]
        self.assertEqual(len(calls), 1)

    def test_get_server_config_existing(self):
        """Test getting an existing server configuration."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        server_config = {"url": "http://test-server"}
        config_manager.config_data = {"mcpServers": {"test-server": server_config}}

        result = config_manager.get_server_config("test-server")
        self.assertEqual(result, server_config)

    def test_get_server_config_nonexistent(self):
        """Test getting a nonexistent server configuration."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {"mcpServers": {}}

        result = config_manager.get_server_config("nonexistent-server")
        self.assertIsNone(result)

    def test_get_server_config_no_servers_key(self):
        """Test getting server config when 'mcpServers' key is missing."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {}  # No 'mcpServers' key

        result = config_manager.get_server_config("any-server")
        self.assertIsNone(result)

    def test_list_servers(self):
        """Test listing all servers."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        servers = {"server1": {"url": "http://server1"}, "server2": {"url": "http://server2"}}
        config_manager.config_data = {"mcpServers": servers}

        result = config_manager.list_servers()
        self.assertEqual(result, servers)

    def test_list_servers_no_servers_key(self):
        """Test listing servers when 'mcpServers' key is missing."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {}  # No 'mcpServers' key

        result = config_manager.list_servers()
        self.assertEqual(result, {})

    def test_add_server_new(self):
        """Test adding a new server."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {}

        server_config = {"url": "http://new-server"}
        config_manager.add_server("new-server", server_config)

        expected = {"mcpServers": {"new-server": server_config}}
        self.assertEqual(config_manager.config_data, expected)

    def test_add_server_update_existing(self):
        """Test updating an existing server."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {"mcpServers": {"existing-server": {"url": "http://old-url"}}}

        new_config = {"url": "http://new-url"}
        config_manager.add_server("existing-server", new_config)

        expected = {"mcpServers": {"existing-server": new_config}}
        self.assertEqual(config_manager.config_data, expected)

    def test_add_server_when_servers_key_exists(self):
        """Test adding a server when mcpServers key already exists."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {"mcpServers": {"server1": {"url": "http://s1"}}}

        server_config = {"url": "http://new-server"}
        config_manager.add_server("new-server", server_config)

        expected = {"mcpServers": {"server1": {"url": "http://s1"}, "new-server": server_config}}
        self.assertEqual(config_manager.config_data, expected)

    def test_remove_server_existing(self):
        """Test removing an existing server."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {
            "mcpServers": {"server-to-remove": {"url": "http://server"}, "other-server": {"url": "http://other"}}
        }

        result = config_manager.remove_server("server-to-remove")

        self.assertTrue(result)
        expected = {"mcpServers": {"other-server": {"url": "http://other"}}}
        self.assertEqual(config_manager.config_data, expected)

    def test_remove_server_nonexistent(self):
        """Test removing a nonexistent server."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {"mcpServers": {}}

        result = config_manager.remove_server("nonexistent")

        self.assertFalse(result)
        self.assertEqual(config_manager.config_data, {"mcpServers": {}})

    def test_remove_server_no_servers_key(self):
        """Test removing a server when 'mcpServers' key is missing."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {}  # No 'mcpServers' key

        result = config_manager.remove_server("any-server")
        self.assertFalse(result)
        self.assertEqual(config_manager.config_data, {})

    @patch.dict(
        os.environ,
        {
            "MCP_SERVER_TEST_URL": "http://test-from-env",
            "MCP_SERVER_TEST_API_KEY": "key-from-env",
            "UNRELATED_VAR": "unrelated",
        },
    )
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {}

        config_manager.load_from_env()

        expected = {"mcpServers": {"test": {"url": "http://test-from-env", "api_key": "key-from-env"}}}
        self.assertEqual(config_manager.config_data, expected)

    @patch.dict(
        os.environ,
        {
            "MCP_SERVER_MALFORMED": "should be ignored",
            "MCP_SERVER_PROD_URL": "http://prod-url",
            "MCP_SERVER_PROD_TIMEOUT": "30",
            "MCP_SERVER_DEV_URL": "http://dev-url",
        },
    )
    def test_load_from_env_malformed_and_multiple(self):
        """Test loading from env with malformed keys and multiple servers."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {}

        config_manager.load_from_env()

        expected = {
            "mcpServers": {"prod": {"url": "http://prod-url", "timeout": "30"}, "dev": {"url": "http://dev-url"}}
        }
        self.assertEqual(config_manager.config_data, expected)

    @patch.dict(os.environ, {"MCP_SERVER_EXISTING_URL": "http://env-url"})
    def test_load_from_env_overwrites_existing(self):
        """Test that env variables overwrite existing config."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        # Pre-populate config
        config_manager.config_data = {"mcpServers": {"existing": {"url": "http://file-url", "api_key": "file-key"}}}

        config_manager.load_from_env()

        expected = {
            "mcpServers": {
                "existing": {
                    "url": "http://env-url",  # URL updated from env
                    "api_key": "file-key",  # API key remains from file
                }
            }
        }
        self.assertEqual(config_manager.config_data, expected)

    @patch.dict(os.environ, {"MCP_SERVER_MALFORMEDKEY": "ignored"})
    def test_load_from_env_ignores_malformed_keys(self):
        """Test that load_from_env skips keys not matching SERVER_PROP format."""
        config_manager = MCPConfigurationManager(self.console, self.config_path)
        config_manager.config_data = {"mcpServers": {"preexisting": {"url": "abc"}}}

        config_manager.load_from_env()  # Should not add MALFORMEDKEY

        # Ensure only the preexisting data is present
        expected = {"mcpServers": {"preexisting": {"url": "abc"}}}
        self.assertEqual(config_manager.config_data, expected)


if __name__ == "__main__":
    unittest.main()
