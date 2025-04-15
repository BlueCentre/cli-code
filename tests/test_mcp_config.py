"""
Tests for the MCP configuration manager.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

from src.cli_code.mcp.config import MCPConfigurationManager


class TestMCPConfigurationManager(unittest.TestCase):
    """Tests for the MCPConfigurationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.console = MagicMock(spec=Console)
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_config_empty_file(self):
        """Test loading from an empty file."""
        # Write an empty JSON object to the file
        with open(self.temp_file.name, "w") as f:
            f.write("{}")
        
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        self.assertEqual(config_manager.config_data, {})
    
    def test_load_config_with_data(self):
        """Test loading configuration with data."""
        test_config = {
            "mcpServers": {
                "test-server": {
                    "url": "http://test-server",
                    "api_key": "test-key"
                }
            }
        }
        
        with open(self.temp_file.name, "w") as f:
            json.dump(test_config, f)
        
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        self.assertEqual(config_manager.config_data, test_config)
    
    def test_load_config_invalid_json(self):
        """Test loading with invalid JSON."""
        with open(self.temp_file.name, "w") as f:
            f.write("invalid json")
        
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        self.assertEqual(config_manager.config_data, {})
        self.console.print.assert_called_once()
    
    def test_save_config(self):
        """Test saving configuration."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        config_manager.config_data = {
            "mcpServers": {
                "test-server": {
                    "url": "http://test-server"
                }
            }
        }
        
        result = config_manager.save_config()
        self.assertTrue(result)
        
        with open(self.temp_file.name, "r") as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, config_manager.config_data)
    
    def test_get_server_config_existing(self):
        """Test getting an existing server configuration."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        server_config = {"url": "http://test-server"}
        config_manager.config_data = {
            "mcpServers": {
                "test-server": server_config
            }
        }
        
        result = config_manager.get_server_config("test-server")
        self.assertEqual(result, server_config)
    
    def test_get_server_config_nonexistent(self):
        """Test getting a nonexistent server configuration."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        config_manager.config_data = {"mcpServers": {}}
        
        result = config_manager.get_server_config("nonexistent-server")
        self.assertIsNone(result)
    
    def test_list_servers(self):
        """Test listing all servers."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        servers = {
            "server1": {"url": "http://server1"},
            "server2": {"url": "http://server2"}
        }
        config_manager.config_data = {"mcpServers": servers}
        
        result = config_manager.list_servers()
        self.assertEqual(result, servers)
    
    def test_add_server_new(self):
        """Test adding a new server."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        config_manager.config_data = {}
        
        server_config = {"url": "http://new-server"}
        config_manager.add_server("new-server", server_config)
        
        expected = {"mcpServers": {"new-server": server_config}}
        self.assertEqual(config_manager.config_data, expected)
    
    def test_add_server_update_existing(self):
        """Test updating an existing server."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        config_manager.config_data = {
            "mcpServers": {
                "existing-server": {"url": "http://old-url"}
            }
        }
        
        new_config = {"url": "http://new-url"}
        config_manager.add_server("existing-server", new_config)
        
        expected = {"mcpServers": {"existing-server": new_config}}
        self.assertEqual(config_manager.config_data, expected)
    
    def test_remove_server_existing(self):
        """Test removing an existing server."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        config_manager.config_data = {
            "mcpServers": {
                "server-to-remove": {"url": "http://server"},
                "other-server": {"url": "http://other"}
            }
        }
        
        result = config_manager.remove_server("server-to-remove")
        
        self.assertTrue(result)
        expected = {"mcpServers": {"other-server": {"url": "http://other"}}}
        self.assertEqual(config_manager.config_data, expected)
    
    def test_remove_server_nonexistent(self):
        """Test removing a nonexistent server."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        config_manager.config_data = {"mcpServers": {}}
        
        result = config_manager.remove_server("nonexistent")
        
        self.assertFalse(result)
        self.assertEqual(config_manager.config_data, {"mcpServers": {}})
    
    @patch.dict(os.environ, {
        "MCP_SERVER_TEST_URL": "http://test-from-env",
        "MCP_SERVER_TEST_API_KEY": "key-from-env",
        "UNRELATED_VAR": "unrelated"
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config_manager = MCPConfigurationManager(self.console, self.temp_file.name)
        config_manager.config_data = {}
        
        config_manager.load_from_env()
        
        expected = {
            "mcpServers": {
                "test": {
                    "url": "http://test-from-env",
                    "api_key": "key-from-env"
                }
            }
        }
        self.assertEqual(config_manager.config_data, expected)


if __name__ == '__main__':
    unittest.main() 