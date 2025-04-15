"""
Configuration manager for MCP settings.
"""
import json
import os
from typing import Any, Dict, Optional

from rich.console import Console


class MCPConfigurationManager:
    """Manages loading and storing MCP server configurations."""
    
    DEFAULT_CONFIG_FILE = "server_config.json"
    
    def __init__(self, console: Console, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            console: Rich console for output
            config_file: Path to configuration file (uses default if None)
        """
        self.console = console
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config_data = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    self.config_data = json.load(f)
            except json.JSONDecodeError:
                self.console.print(f"[yellow]Warning:[/] Invalid JSON in {self.config_file}")
            except Exception as e:
                self.console.print(f"[yellow]Warning:[/] Failed to load config: {str(e)}")
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config_data, f, indent=2)
            return True
        except Exception as e:
            self.console.print(f"[red]Error:[/] Failed to save config: {str(e)}")
            return False
    
    def get_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server configuration or None if not found
        """
        servers = self.config_data.get("mcpServers", {})
        return servers.get(server_name)
    
    def list_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configured servers.
        
        Returns:
            Dictionary of server names to configurations
        """
        return self.config_data.get("mcpServers", {})
    
    def add_server(self, server_name: str, server_config: Dict[str, Any]) -> None:
        """
        Add or update a server configuration.
        
        Args:
            server_name: Name of the server
            server_config: Server configuration
        """
        if "mcpServers" not in self.config_data:
            self.config_data["mcpServers"] = {}
        
        self.config_data["mcpServers"][server_name] = server_config
    
    def remove_server(self, server_name: str) -> bool:
        """
        Remove a server configuration.
        
        Args:
            server_name: Name of the server to remove
            
        Returns:
            True if server was removed, False if not found
        """
        if "mcpServers" in self.config_data and server_name in self.config_data["mcpServers"]:
            del self.config_data["mcpServers"][server_name]
            return True
        return False
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Example format: MCP_SERVER_myserver_URL=http://localhost:8000
        prefix = "MCP_SERVER_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Extract server name and property
                parts = key[len(prefix):].lower().split("_", 1)
                if len(parts) != 2:
                    continue
                
                server_name, prop = parts
                
                # Initialize server config if needed
                if "mcpServers" not in self.config_data:
                    self.config_data["mcpServers"] = {}
                
                if server_name not in self.config_data["mcpServers"]:
                    self.config_data["mcpServers"][server_name] = {}
                
                # Set property
                self.config_data["mcpServers"][server_name][prop] = value 