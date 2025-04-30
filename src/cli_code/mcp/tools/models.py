"""
Tool models for the MCP protocol.

This module provides data models for tools used in the MCP protocol.
"""

import json
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
    properties: Optional[Dict[str, Any]] = None
    items: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter to dictionary representation."""
        result = {"name": self.name, "description": self.description, "type": self.type}

        if self.required:
            result["required"] = self.required

        if self.enum is not None:
            result["enum"] = self.enum

        if self.properties is not None:
            result["properties"] = self.properties

        if self.items is not None:
            result["items"] = self.items

        return result

    def to_schema(self) -> Dict[str, Any]:
        """Convert parameter to JSON Schema."""
        schema = {"type": self.type, "description": self.description}

        if self.enum is not None:
            schema["enum"] = self.enum

        if self.properties is not None and self.type == "object":
            schema["properties"] = self.properties

        if self.items is not None and self.type == "array":
            schema["items"] = self.items

        return schema


@dataclass
class Tool:
    """Tool definition for MCP protocol."""

    name: str
    description: str
    parameters: List[ToolParameter]
    handler: Callable[[Dict[str, Any]], Awaitable[Any]]

    def __post_init__(self):
        """Generate schema if not provided."""
        self._schema = self._generate_schema()

    def _generate_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema for the tool parameters."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }

    @property
    def schema(self) -> Dict[str, Any]:
        """Get the tool schema."""
        return self._schema

    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given parameters.

        Args:
            parameters: Parameters to pass to the handler

        Returns:
            The result of the tool execution
        """
        return await self.handler(parameters)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None

    def __init__(
        self,
        tool_name: Optional[str] = None,
        name: Optional[str] = None,
        parameters: Dict[str, Any] = None,
        result: Any = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """
        Initialize the tool result.

        Args:
            tool_name: The name of the tool (preferred field)
            name: The name of the tool (deprecated, use tool_name instead)
            parameters: The parameters passed to the tool
            result: The result of the tool execution
            success: Whether the execution was successful
            error: The error message if execution failed
        """
        # Support both name and tool_name for backward compatibility
        if tool_name is not None:
            self.tool_name = tool_name
        elif name is not None:
            self.tool_name = name
        else:
            raise ValueError("Either tool_name or name must be provided")

        self.parameters = parameters if parameters is not None else {}
        self.result = result
        self.success = success
        self.error = error

    @property
    def name(self) -> str:
        """
        Get the name of the tool (backward compatibility property).

        Returns:
            The name of the tool
        """
        return self.tool_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        result_dict = {
            "name": self.tool_name,
            "parameters": self.parameters,
            "result": self.result,
            "success": self.success,
        }

        if not self.success and self.error:
            result_dict["error"] = self.error

        return result_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create tool result from dictionary."""
        # Support both name and tool_name for backward compatibility
        tool_name = data.get("tool_name") or data.get("name")
        if not tool_name:
            raise KeyError("Neither 'tool_name' nor 'name' found in data")

        return cls(
            tool_name=tool_name,
            parameters=data["parameters"],
            result=data["result"],
            success=data.get("success", True),
            error=data.get("error"),
        )

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ToolResult":
        """Create tool result from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
