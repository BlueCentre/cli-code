"""
Tool response formatter for MCP protocol.

This module provides functionality for formatting tool execution results.
"""

import json
from typing import Any, Dict, List, Optional, Union

from src.cli_code.mcp.tools.models import ToolResult


class ToolResponseFormatter:
    """Formatter for MCP tool responses."""

    def format_result(self, result: ToolResult) -> Dict[str, Any]:
        """
        Format a tool result for the MCP protocol response.

        Args:
            result: The tool result to format

        Returns:
            A dictionary containing the formatted result
        """
        return {
            "name": result.tool_name,
            "parameters": result.parameters,
            "result": self._prepare_result_for_json(result.result),
        }

    def format_results(self, results: List[ToolResult]) -> Dict[str, Any]:
        """
        Format multiple tool results for the MCP protocol response.

        Args:
            results: The tool results to format

        Returns:
            A dictionary containing the formatted results
        """
        formatted_results = [self.format_result(result) for result in results]
        return {"results": formatted_results}

    def format_error(self, error: Exception, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format an error that occurred during tool execution.

        Args:
            error: The error that occurred
            tool_name: The name of the tool that failed
            parameters: The parameters passed to the tool

        Returns:
            A dictionary containing the formatted error
        """
        return {
            "name": tool_name,
            "parameters": parameters,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    def _prepare_result_for_json(self, result: Any) -> Any:
        """
        Prepare a result for JSON serialization.

        Args:
            result: The result to prepare

        Returns:
            The prepared result
        """
        if hasattr(result, "to_dict") and callable(result.to_dict):
            return result.to_dict()

        if hasattr(result, "__dict__"):
            return result.__dict__

        # Handle other common types that might need special handling
        if isinstance(result, (list, tuple)):
            return [self._prepare_result_for_json(item) for item in result]
        elif isinstance(result, dict):
            return {k: self._prepare_result_for_json(v) for k, v in result.items()}

        # For basic types, return as is
        return result
