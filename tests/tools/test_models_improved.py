"""
Improved tests for the tool models.

This module provides comprehensive tests for the Tool, ToolParameter, and ToolResult classes
to improve test coverage.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jsonschema import ValidationError

from src.cli_code.mcp.tools.models import Tool, ToolParameter, ToolResult


class TestToolParameterImproved(unittest.TestCase):
    """Improved tests for the ToolParameter class."""

    def test_parameter_with_defaults(self):
        """Test parameter initialization with default values."""
        param = ToolParameter(name="test_param", description="A test parameter", type="string")

        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.description, "A test parameter")
        self.assertEqual(param.type, "string")
        self.assertFalse(param.required)
        self.assertIsNone(param.default)
        self.assertIsNone(param.enum)
        self.assertIsNone(param.properties)
        self.assertIsNone(param.items)

    def test_parameter_with_all_fields(self):
        """Test parameter initialization with all fields."""
        param = ToolParameter(
            name="test_param",
            description="A test parameter",
            type="object",
            required=True,
            default={"key": "value"},
            enum=["option1", "option2"],
            properties={"prop1": {"type": "string"}, "prop2": {"type": "number"}},
            items={"type": "string"},
        )

        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.description, "A test parameter")
        self.assertEqual(param.type, "object")
        self.assertTrue(param.required)
        self.assertEqual(param.default, {"key": "value"})
        self.assertEqual(param.enum, ["option1", "option2"])
        self.assertEqual(param.properties, {"prop1": {"type": "string"}, "prop2": {"type": "number"}})
        self.assertEqual(param.items, {"type": "string"})

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        param = ToolParameter(name="test_param", description="A test parameter", type="string")

        result = param.to_dict()

        self.assertEqual(result, {"name": "test_param", "description": "A test parameter", "type": "string"})

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        param = ToolParameter(
            name="test_param",
            description="A test parameter",
            type="object",
            required=True,
            enum=["option1", "option2"],
            properties={"prop1": {"type": "string"}, "prop2": {"type": "number"}},
            items={"type": "string"},
        )

        result = param.to_dict()

        self.assertEqual(
            result,
            {
                "name": "test_param",
                "description": "A test parameter",
                "type": "object",
                "required": True,
                "enum": ["option1", "option2"],
                "properties": {"prop1": {"type": "string"}, "prop2": {"type": "number"}},
                "items": {"type": "string"},
            },
        )

    def test_to_schema_string(self):
        """Test to_schema for string parameter."""
        param = ToolParameter(name="test_param", description="A test parameter", type="string")

        schema = param.to_schema()

        self.assertEqual(schema, {"type": "string", "description": "A test parameter"})

    def test_to_schema_number(self):
        """Test to_schema for number parameter."""
        param = ToolParameter(name="test_param", description="A test parameter", type="number")

        schema = param.to_schema()

        self.assertEqual(schema, {"type": "number", "description": "A test parameter"})

    def test_to_schema_object(self):
        """Test to_schema for object parameter."""
        param = ToolParameter(
            name="test_param",
            description="A test parameter",
            type="object",
            properties={"prop1": {"type": "string"}, "prop2": {"type": "number"}},
        )

        schema = param.to_schema()

        self.assertEqual(
            schema,
            {
                "type": "object",
                "description": "A test parameter",
                "properties": {"prop1": {"type": "string"}, "prop2": {"type": "number"}},
            },
        )

    def test_to_schema_array(self):
        """Test to_schema for array parameter."""
        param = ToolParameter(name="test_param", description="A test parameter", type="array", items={"type": "string"})

        schema = param.to_schema()

        self.assertEqual(schema, {"type": "array", "description": "A test parameter", "items": {"type": "string"}})

    def test_to_schema_with_enum(self):
        """Test to_schema for parameter with enum."""
        param = ToolParameter(
            name="test_param", description="A test parameter", type="string", enum=["option1", "option2"]
        )

        schema = param.to_schema()

        self.assertEqual(schema, {"type": "string", "description": "A test parameter", "enum": ["option1", "option2"]})


class TestToolImproved(unittest.TestCase):
    """Improved tests for the Tool class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = AsyncMock(return_value={"result": "success"})

        self.params = [
            ToolParameter(name="param1", description="Parameter 1", type="string", required=True),
            ToolParameter(name="param2", description="Parameter 2", type="number", required=False),
        ]

    def test_init(self):
        """Test initializing a tool."""
        tool = Tool(name="test_tool", description="A test tool", parameters=self.params, handler=self.handler)

        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "A test tool")
        self.assertEqual(tool.parameters, self.params)
        self.assertEqual(tool.handler, self.handler)

    def test_schema_generation(self):
        """Test schema generation."""
        tool = Tool(name="test_tool", description="A test tool", parameters=self.params, handler=self.handler)

        schema = tool.schema

        self.assertEqual(schema["name"], "test_tool")
        self.assertEqual(schema["description"], "A test tool")
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertEqual(len(schema["parameters"]["properties"]), 2)
        self.assertEqual(schema["parameters"]["required"], ["param1"])

        # Check the property definitions
        self.assertEqual(schema["parameters"]["properties"]["param1"]["type"], "string")
        self.assertEqual(schema["parameters"]["properties"]["param1"]["description"], "Parameter 1")
        self.assertEqual(schema["parameters"]["properties"]["param2"]["type"], "number")
        self.assertEqual(schema["parameters"]["properties"]["param2"]["description"], "Parameter 2")

    def test_schema_generation_no_parameters(self):
        """Test schema generation with no parameters."""
        tool = Tool(name="test_tool", description="A test tool", parameters=[], handler=self.handler)

        schema = tool.schema

        self.assertEqual(schema["name"], "test_tool")
        self.assertEqual(schema["description"], "A test tool")
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertEqual(schema["parameters"]["properties"], {})
        self.assertEqual(schema["parameters"]["required"], [])

    def test_schema_generation_with_enum(self):
        """Test schema generation with enum parameters."""
        params = [
            ToolParameter(
                name="operation",
                description="Operation to perform",
                type="string",
                required=True,
                enum=["add", "subtract", "multiply", "divide"],
            )
        ]

        tool = Tool(name="calculator", description="A calculator tool", parameters=params, handler=self.handler)

        schema = tool.schema

        self.assertEqual(
            schema["parameters"]["properties"]["operation"]["enum"], ["add", "subtract", "multiply", "divide"]
        )

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test executing a tool."""
        tool = Tool(name="test_tool", description="A test tool", parameters=self.params, handler=self.handler)

        result = await tool.execute({"param1": "value1", "param2": 42})

        self.assertEqual(result, {"result": "success"})
        self.handler.assert_called_once_with({"param1": "value1", "param2": 42})

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test executing a tool that raises an error."""
        handler = AsyncMock(side_effect=ValueError("Test error"))

        tool = Tool(
            name="error_tool", description="A tool that raises an error", parameters=self.params, handler=handler
        )

        with self.assertRaises(ValueError) as context:
            await tool.execute({"param1": "value1", "param2": 42})

        self.assertEqual(str(context.exception), "Test error")
        handler.assert_called_once_with({"param1": "value1", "param2": 42})


class TestToolResultImproved(unittest.TestCase):
    """Improved tests for the ToolResult class."""

    def test_init_with_tool_name(self):
        """Test initializing with tool_name."""
        result = ToolResult(tool_name="test_tool", parameters={"param1": "value1"}, result={"status": "success"})

        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"status": "success"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_init_with_name(self):
        """Test initializing with name (deprecated)."""
        result = ToolResult(name="test_tool", parameters={"param1": "value1"}, result={"status": "success"})

        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.name, "test_tool")  # Check name property
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"status": "success"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_init_with_error(self):
        """Test initializing with an error."""
        result = ToolResult(
            tool_name="test_tool", parameters={"param1": "value1"}, result=None, success=False, error="Test error"
        )

        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Test error")

    def test_init_with_empty_parameters(self):
        """Test initializing with empty parameters."""
        result = ToolResult(tool_name="test_tool", parameters=None, result={"status": "success"})

        self.assertEqual(result.parameters, {})

    def test_init_without_name(self):
        """Test initializing without a name."""
        with self.assertRaises(ValueError) as context:
            ToolResult(parameters={"param1": "value1"}, result={"status": "success"})

        self.assertIn("Either tool_name or name must be provided", str(context.exception))

    def test_to_dict_success(self):
        """Test to_dict for a successful result."""
        result = ToolResult(tool_name="test_tool", parameters={"param1": "value1"}, result={"status": "success"})

        result_dict = result.to_dict()

        self.assertEqual(
            result_dict,
            {"name": "test_tool", "parameters": {"param1": "value1"}, "result": {"status": "success"}, "success": True},
        )

    def test_to_dict_error(self):
        """Test to_dict for a result with an error."""
        result = ToolResult(
            tool_name="test_tool", parameters={"param1": "value1"}, result=None, success=False, error="Test error"
        )

        result_dict = result.to_dict()

        self.assertEqual(
            result_dict,
            {
                "name": "test_tool",
                "parameters": {"param1": "value1"},
                "result": None,
                "success": False,
                "error": "Test error",
            },
        )

    def test_from_dict_success(self):
        """Test from_dict for a successful result."""
        data = {
            "tool_name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": {"status": "success"},
            "success": True,
        }

        result = ToolResult.from_dict(data)

        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"status": "success"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_from_dict_with_name(self):
        """Test from_dict with name instead of tool_name."""
        data = {
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": {"status": "success"},
            "success": True,
        }

        result = ToolResult.from_dict(data)

        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"status": "success"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_from_dict_error(self):
        """Test from_dict for a result with an error."""
        data = {
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": None,
            "success": False,
            "error": "Test error",
        }

        result = ToolResult.from_dict(data)

        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Test error")

    def test_from_dict_without_name(self):
        """Test from_dict without a name."""
        data = {"parameters": {"param1": "value1"}, "result": {"status": "success"}}

        with self.assertRaises(KeyError) as context:
            ToolResult.from_dict(data)

        self.assertIn("Neither 'tool_name' nor 'name' found in data", str(context.exception))

    def test_to_json(self):
        """Test to_json serialization."""
        result = ToolResult(tool_name="test_tool", parameters={"param1": "value1"}, result={"status": "success"})

        json_str = result.to_json()

        # Parse the JSON string back to a dict for comparison
        parsed = json.loads(json_str)
        self.assertEqual(
            parsed,
            {"name": "test_tool", "parameters": {"param1": "value1"}, "result": {"status": "success"}, "success": True},
        )

    def test_from_json(self):
        """Test from_json deserialization."""
        json_str = json.dumps(
            {"name": "test_tool", "parameters": {"param1": "value1"}, "result": {"status": "success"}, "success": True}
        )

        result = ToolResult.from_json(json_str)

        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"status": "success"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_from_json_invalid(self):
        """Test from_json with invalid JSON."""
        with self.assertRaises(json.JSONDecodeError):
            ToolResult.from_json("invalid json")
