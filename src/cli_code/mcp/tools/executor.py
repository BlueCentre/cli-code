"""
Tool executor for MCP protocol.

This module provides functionality for executing tools with parameter validation.
"""
import json
import logging
from typing import Any, Dict, Optional

import jsonschema

from cli_code.mcp.tools.models import Tool, ToolResult
from cli_code.mcp.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executor for MCP tools."""
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize the tool executor.
        
        Args:
            registry: The tool registry to use
        """
        self.registry = registry
    
    def validate_parameters(self, tool: Tool, parameters: Dict[str, Any]) -> None:
        """
        Validate tool parameters against the tool's schema.
        
        Args:
            tool: The tool to validate parameters for
            parameters: The parameters to validate
            
        Raises:
            jsonschema.ValidationError: If the parameters are invalid
        """
        try:
            jsonschema.validate(instance=parameters, schema=tool.schema)
        except jsonschema.ValidationError as e:
            logger.error(f"Parameter validation failed for tool '{tool.name}': {e}")
            raise
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: The name of the tool to execute
            parameters: The parameters to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
            jsonschema.ValidationError: If the parameters are invalid
            Exception: If the tool execution fails
        """
        try:
            # Get the tool
            tool = self.registry.get_tool(tool_name)
            
            # Validate parameters
            self.validate_parameters(tool, parameters)
            
            # Execute the tool
            logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
            result = await tool.execute(**parameters)
            
            # Format and return the result
            return ToolResult(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                success=True
            )
        except jsonschema.ValidationError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Log and wrap other exceptions
            logger.exception(f"Tool execution failed for '{tool_name}': {e}")
            raise 