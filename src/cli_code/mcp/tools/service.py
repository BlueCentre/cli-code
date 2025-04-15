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
from src.cli_code.mcp.tools.models import Tool, ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolService:
    """Service for executing tools through the MCP protocol."""

    def __init__(self, registry: ToolRegistry, executor: Optional[ToolExecutor] = None):
        """
        Initialize the tool service.

        Args:
            registry: The tool registry to use
            executor: The tool executor to use, or None to create a new one
        """
        self.registry = registry
        self.executor = executor if executor is not None else ToolExecutor(registry)
        self.formatter = ToolResponseFormatter()

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool.

        Args:
            tool_name: The name of the tool to execute
            parameters: The parameters to pass to the tool

        Returns:
            The result of the tool execution
        """
        try:
            # Execute the tool
            return await self.executor.execute(tool_name, parameters)
        except Exception as e:
            # Create a failed result
            return ToolResult(
                tool_name=tool_name,
                parameters=parameters,
                success=False,
                result=None,
                error=str(e)
            )

    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call in the format provided by LLMs.

        Args:
            tool_call: The tool call to execute

        Returns:
            The formatted result of the tool execution
        """
        tool_call_id = tool_call.get("id", "unknown")
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "unknown")
        arguments = function_data.get("arguments", {})

        # Parse arguments if they are a JSON string
        if isinstance(arguments, str):
            try:
                parameters = json.loads(arguments)
            except json.JSONDecodeError as e:
                return {
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "status": "error",
                    "content": f"Invalid JSON in tool call arguments: {str(e)}"
                }
        else:
            parameters = arguments

        # Execute the tool
        result = await self.execute_tool(tool_name, parameters)

        # Format the result for the tool call response
        return {
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "status": "success" if result.success else "error",
            "content": json.dumps(result.result) if result.success else result.error
        }

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: A list of tool calls to execute

        Returns:
            A list of formatted results of the tool executions
        """
        results = []

        for tool_call in tool_calls:
            result = await self.execute_tool_call(tool_call)
            results.append(result)

        return results

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get definitions for all available tools.

        Returns:
            A list of tool definitions
        """
        tools = self.registry.get_all_tools()
        definitions = []

        for tool in tools:
            # Format the schema to match the expected structure
            params = tool.schema.get("parameters", {})
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": params
            })

        return definitions

    def get_tool_definition_by_name(self, name: str) -> Dict:
        """Get a single tool definition by name."""
        try:
            tool = self.registry.get_tool(name)
            params = tool.schema.get("parameters", {})
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": params,
            }
        except ValueError:
            return None

    def format_result(self, result: ToolResult) -> Dict[str, Any]:
        """
        Format a tool result.

        Args:
            result: The result to format

        Returns:
            The formatted result
        """
        return self.formatter.format_result(result)

    def format_results(self, results: List[ToolResult]) -> Dict[str, Any]:
        """
        Format multiple tool results.

        Args:
            results: The results to format

        Returns:
            The formatted results
        """
        return self.formatter.format_results(results)

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available tools.

        Returns:
            A dictionary of tool information
        """
        return self.registry.get_schemas()
