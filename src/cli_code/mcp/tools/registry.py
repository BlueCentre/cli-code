"""
Tool registry for MCP protocol.

This module provides a registry for registering and managing tools.
"""

from typing import Any, Dict, List, Optional

from src.cli_code.mcp.tools.models import Tool


class ToolRegistry:
    """Registry for MCP tools."""

    def __init__(self):
        """Initialize an empty registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool with the registry.

        Args:
            tool: The tool to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered")

        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: The name of the tool to unregister

        Raises:
            ValueError: If no tool with the given name is registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"No tool with name '{tool_name}' is registered")

        del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Tool:
        """
        Get a tool by name.

        Args:
            tool_name: The name of the tool to get

        Returns:
            The requested tool

        Raises:
            ValueError: If no tool with the given name is registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"No tool with name '{tool_name}' is registered")

        return self._tools[tool_name]

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            A list of all registered tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.

        Returns:
            A list of all registered tools
        """
        return list(self._tools.values())

    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get schemas for all registered tools.

        Returns:
            A dictionary mapping tool names to their JSON schemas
        """
        return {
            tool.name: {"name": tool.name, "description": tool.description, "parameters": tool.schema}
            for tool in self._tools.values()
        }
