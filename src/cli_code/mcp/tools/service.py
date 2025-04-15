"""
Tool service for MCP protocol.

This module provides a service for executing tools through the MCP protocol.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import jsonschema

from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.formatter import ToolResponseFormatter
from src.cli_code.mcp.tools.models import ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolService:
    """Service for executing tools through the MCP protocol."""

    def __init__(self, registry: ToolRegistry):
        """
        Initialize the tool service.

        Args:
            registry: The tool registry to use
        """
        self.registry = registry
        self.executor = ToolExecutor(registry)
        self.formatter = ToolResponseFormatter()

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and format the result.

        Args:
            tool_name: The name of the tool to execute
            parameters: The parameters to pass to the tool

        Returns:
            The formatted result of the tool execution
        """
        try:
            # Execute the tool
            result = await self.executor.execute(tool_name, parameters)

            # Format and return the result
            return self.formatter.format_result(result)
        except jsonschema.ValidationError as e:
            # Format validation errors
            return self.formatter.format_error(e, tool_name, parameters)
        except ValueError as e:
            # Format tool not found errors
            return self.formatter.format_error(e, tool_name, parameters)
        except Exception as e:
            # Format other errors
            return self.formatter.format_error(e, tool_name, parameters)

    async def execute_tools(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple tools and format the results.

        Args:
            tools: A list of tool specifications, each containing a name and parameters

        Returns:
            The formatted results of the tool executions
        """
        results = []

        for tool_spec in tools:
            tool_name = tool_spec.get("name")
            parameters = tool_spec.get("parameters", {})

            if not tool_name:
                logger.error(f"Missing tool name in tool specification: {tool_spec}")
                continue

            try:
                # Execute the tool
                result = await self.executor.execute(tool_name, parameters)
                results.append(result)
            except Exception as e:
                # Log the error but continue with other tools
                logger.error(f"Tool execution failed for '{tool_name}': {e}")

                # Create a failed result
                failed_result = ToolResult(
                    tool_name=tool_name, parameters=parameters, success=False, result=None, error=str(e)
                )
                results.append(failed_result)

        # Format and return the results
        return self.formatter.format_results(results)

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available tools.

        Returns:
            A dictionary of tool information
        """
        return self.registry.get_schemas()
