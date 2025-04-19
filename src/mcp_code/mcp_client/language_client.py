"""
This module provides the LanguageClient class for connecting to language servers.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union


class LanguageClient:
    """
    A client for connecting to language servers and executing commands.

    This class serves as an async context manager for connecting to servers,
    executing commands, and properly cleaning up resources.
    """

    def __init__(self, server_params: Dict[str, Any], streams: Optional[Tuple] = None):
        """
        Initialize a LanguageClient with the given server parameters.

        Args:
            server_params: Configuration parameters for connecting to the server
            streams: Optional tuple of (read_stream, write_stream) if already established
        """
        self.server_params = server_params
        self.connected = False
        self.streams = streams
        self.server_name = server_params.get("name", "unknown")

    async def __aenter__(self):
        """
        Enter the async context manager, establishing a connection to the server.

        Returns:
            self: The client instance
        """
        self.connected = True
        logging.debug(f"LanguageClient connected to server: {self.server_name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the async context manager, cleaning up resources.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            bool: True if the exception was handled, False otherwise
        """
        logging.debug(f"LanguageClient disconnecting from server: {self.server_name}")
        self.connected = False
        return False  # Don't suppress exceptions
