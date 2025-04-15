"""
Tool models for the MCP protocol.

This module provides data models for tools used in the MCP protocol.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""

    name: str
    description: str
    type: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


@dataclass
class Tool:
    """Tool definition for MCP protocol."""

    name: str
    description: str
    parameters: List[ToolParameter]
    handler: Callable[..., Awaitable[Any]]
    schema: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Generate schema if not provided."""
        if self.schema is None:
            self.schema = self._generate_schema()

    def _generate_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema for the tool parameters."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}

            if param.enum:
                prop["enum"] = param.enum

            if param.required:
                required.append(param.name)

            properties[param.name] = prop

        return {"type": "object", "properties": properties, "required": required}

    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given parameters.

        Args:
            **kwargs: Parameters to pass to the handler

        Returns:
            The result of the tool execution
        """
        return await self.handler(**kwargs)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None
