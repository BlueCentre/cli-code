"""
Model adapter interface for MCP protocol integration.

This module provides adapter classes to bridge between LLM models
and the MCP protocol client.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from rich.console import Console

from ..models.base import AbstractModelAgent
from .client import MCPClient, MCPConfig, MCPMessage, MCPToolCall


class MCPModelAdapter(ABC):
    """
    Abstract base class for MCP model adapters.

    This adapter interfaces between the model-specific code and the
    MCP protocol client.
    """

    def __init__(self, model_agent: AbstractModelAgent, mcp_client: MCPClient):
        """
        Initialize the MCP model adapter.

        Args:
            model_agent: The model-specific agent implementation
            mcp_client: The MCP protocol client
        """
        self.model_agent = model_agent
        self.mcp_client = mcp_client

    @abstractmethod
    def format_for_mcp(self, prompt: str, history: List[Dict[str, Any]]) -> List[MCPMessage]:
        """
        Format a prompt and history for the MCP protocol.

        Args:
            prompt: The user's input prompt
            history: The conversation history in model-specific format

        Returns:
            A list of MCPMessage objects formatted for the MCP protocol
        """
        pass

    @abstractmethod
    def parse_from_mcp(self, message: MCPMessage) -> Dict[str, Any]:
        """
        Parse a response from MCP format to model-specific format.

        Args:
            message: The MCP message to parse

        Returns:
            The parsed message in model-specific format
        """
        pass

    @abstractmethod
    def execute_tool(self, tool_call: MCPToolCall) -> Dict[str, Any]:
        """
        Execute a tool call using model-specific mechanisms.

        Args:
            tool_call: The tool call to execute

        Returns:
            The result of the tool execution
        """
        pass

    @abstractmethod
    def send_request(self, prompt: str) -> Optional[str]:
        """
        Send a request to the LLM through the MCP protocol.

        Args:
            prompt: The user's input prompt

        Returns:
            The generated text response, or None if an error occurs
        """
        pass


class ModelAdapterFactory:
    """Factory for creating model adapters based on model type."""

    @staticmethod
    def create_adapter(
        model_type: str, model_agent: AbstractModelAgent, mcp_config: MCPConfig, console: Console
    ) -> MCPModelAdapter:
        """
        Create an adapter for the specified model type.

        Args:
            model_type: The type of model (e.g., "gemini", "openai")
            model_agent: The model-specific agent implementation
            mcp_config: The MCP configuration
            console: The console for output

        Returns:
            An appropriate MCP model adapter

        Raises:
            ValueError: If the model type is not supported
        """
        if model_type.lower() == "gemini":
            from .adapters.gemini_adapter import GeminiModelAdapter

            return GeminiModelAdapter(model_agent, MCPClient(mcp_config), console)
        elif model_type.lower() == "openai":
            # Future implementation
            raise NotImplementedError("OpenAI adapter not yet implemented")
        elif model_type.lower() == "ollama":
            # Future implementation
            raise NotImplementedError("Ollama adapter not yet implemented")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
