"""
Tests for the tools models module.

This module provides comprehensive tests for the ToolParameter, Tool, and ToolResult classes.
"""

import asyncio
import json
import unittest
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

from src.cli_code.mcp.tools.models import Tool, ToolParameter, ToolResult


# Dummy async function for testing
async def dummy_async_handler(params):
    return {"processed": True, "params": params}


class TestToolParameter(unittest.TestCase):
    """Test case for the ToolParameter class."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        param = ToolParameter(name="test_param", description="A test param", type="string")
        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.description, "A test param")
        self.assertEqual(param.type, "string")
        self.assertFalse(param.required)
        self.assertIsNone(param.default)
        self.assertIsNone(param.enum)
        self.assertIsNone(param.properties)
        self.assertIsNone(param.items)

    def test_init_full(self):
        """Test full initialization with all optional fields."""
        param = ToolParameter(
            name="full_param",
            description="Another test param",
            type="array",
            required=True,
            default=[1],
            enum=[[1], [2]],
            properties=None,  # Not applicable for array
            items={"type": "integer"},
        )
        self.assertTrue(param.required)
        self.assertEqual(param.default, [1])
        self.assertEqual(param.enum, [[1], [2]])
        self.assertEqual(param.items, {"type": "integer"})
        self.assertIsNone(param.properties)  # Ensure properties wasn't set incorrectly

    def test_to_dict(self):
        """Test converting ToolParameter to dictionary."""
        param = ToolParameter(
            name="dict_param",
            description="For dict test",
            type="object",
            required=True,
            enum=None,
            properties={"key": {"type": "string"}},
            items=None,
        )
        expected_dict = {
            "name": "dict_param",
            "description": "For dict test",
            "type": "object",
            "required": True,
            "properties": {"key": {"type": "string"}},
        }
        self.assertEqual(param.to_dict(), expected_dict)

        # Test minimal conversion
        param_min = ToolParameter(name="min_param", description="Min", type="boolean")
        expected_min_dict = {
            "name": "min_param",
            "description": "Min",
            "type": "boolean",
        }
        self.assertEqual(param_min.to_dict(), expected_min_dict)

        # Test with enum and items
        param_enum_items = ToolParameter(
            name="ei_param", description="EnumItems", type="array", enum=[[1], [2]], items={"type": "integer"}
        )
        expected_ei_dict = {
            "name": "ei_param",
            "description": "EnumItems",
            "type": "array",
            "enum": [[1], [2]],
            "items": {"type": "integer"},
        }
        self.assertEqual(param_enum_items.to_dict(), expected_ei_dict)

    def test_to_schema(self):
        """Test converting ToolParameter to JSON Schema fragment."""
        param_obj = ToolParameter(
            name="obj_param", description="Obj", type="object", properties={"prop": {"type": "string"}}, required=True
        )
        expected_obj_schema = {"type": "object", "description": "Obj", "properties": {"prop": {"type": "string"}}}
        self.assertEqual(param_obj.to_schema(), expected_obj_schema)

        param_arr = ToolParameter(
            name="arr_param", description="Arr", type="array", items={"type": "number"}, enum=[[1.0], [2.0]]
        )
        expected_arr_schema = {
            "type": "array",
            "description": "Arr",
            "items": {"type": "number"},
            "enum": [[1.0], [2.0]],
        }
        self.assertEqual(param_arr.to_schema(), expected_arr_schema)

        param_simple = ToolParameter(name="simple", description="Simple", type="string", enum=["a", "b"])
        expected_simple_schema = {"type": "string", "description": "Simple", "enum": ["a", "b"]}
        self.assertEqual(param_simple.to_schema(), expected_simple_schema)

        # Ensure properties/items only added for correct types
        param_wrong_type = ToolParameter(name="wrong", description="Wrong", type="string", properties={}, items={})
        expected_wrong_schema = {"type": "string", "description": "Wrong"}
        self.assertEqual(param_wrong_type.to_schema(), expected_wrong_schema)


class TestTool(unittest.TestCase):
    """Test case for the Tool class."""

    def setUp(self):
        """Set up test fixtures."""
        self.param1 = ToolParameter(name="p1", description="Param 1", type="string", required=True)
        self.param2 = ToolParameter(name="p2", description="Param 2", type="integer")
        self.handler_mock = AsyncMock(return_value={"result": "Success"})
        self.tool = Tool(
            name="test_tool",
            description="A test tool for testing",
            parameters=[self.param1, self.param2],
            handler=self.handler_mock,
        )

    def test_init(self):
        """Test initializing a Tool."""
        self.assertEqual(self.tool.name, "test_tool")
        self.assertEqual(self.tool.description, "A test tool for testing")
        self.assertEqual(self.tool.parameters, [self.param1, self.param2])
        self.assertEqual(self.tool.handler, self.handler_mock)
        # Check if schema was generated in __post_init__
        self.assertIsNotNone(self.tool.schema)

    def test_schema_generation(self):
        """Test the generated JSON schema."""
        schema = self.tool.schema
        expected_schema = {
            "name": "test_tool",
            "description": "A test tool for testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "p1": {"type": "string", "description": "Param 1"},
                    "p2": {"type": "integer", "description": "Param 2"},
                },
                "required": ["p1"],
            },
        }
        self.assertEqual(schema, expected_schema)

    def test_schema_property(self):
        """Test accessing the schema property multiple times."""
        schema1 = self.tool.schema
        schema2 = self.tool.schema
        self.assertIs(schema1, schema2)  # Should return the same generated schema object
        self.assertIsNotNone(schema1)

    async def test_execute(self):
        """Test executing the tool's handler."""
        test_params = {"p1": "value1", "p2": 123}
        result = await self.tool.execute(test_params)
        self.handler_mock.assert_awaited_once_with(test_params)
        self.assertEqual(result, {"result": "Success"})

        # Test execution with different params
        await self.tool.execute({"p1": "another"})
        self.handler_mock.assert_awaited_with({"p1": "another"})


class TestToolResult(unittest.TestCase):
    """Test case for the ToolResult class."""

    def test_init_success_tool_name(self):
        """Test successful initialization using tool_name."""
        result = ToolResult(tool_name="calc", parameters={"a": 1}, result=1)
        self.assertEqual(result.tool_name, "calc")
        self.assertEqual(result.name, "calc")  # Check backward compatibility property
        self.assertEqual(result.parameters, {"a": 1})
        self.assertEqual(result.result, 1)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_init_success_name(self):
        """Test successful initialization using deprecated name."""
        result = ToolResult(name="calc", parameters={"b": 2}, result=4)
        self.assertEqual(result.tool_name, "calc")
        self.assertEqual(result.name, "calc")
        self.assertEqual(result.parameters, {"b": 2})
        self.assertEqual(result.result, 4)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_init_failure(self):
        """Test initialization for a failed execution."""
        result = ToolResult(
            tool_name="fail_tool", parameters={}, result=None, success=False, error="Something went wrong"
        )
        self.assertEqual(result.tool_name, "fail_tool")
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Something went wrong")

    def test_init_no_name(self):
        """Test initialization fails if neither tool_name nor name is provided."""
        with self.assertRaisesRegex(ValueError, "Either tool_name or name must be provided"):
            ToolResult(parameters={}, result=None)

    def test_init_default_params(self):
        """Test initialization uses default empty dict for parameters."""
        result = ToolResult(tool_name="default_params")
        self.assertEqual(result.parameters, {})

    def test_to_dict_success(self):
        """Test converting a successful result to dict."""
        result = ToolResult(tool_name="calc", parameters={"a": 1}, result=1)
        expected = {"name": "calc", "parameters": {"a": 1}, "result": 1, "success": True}
        self.assertEqual(result.to_dict(), expected)

    def test_to_dict_failure(self):
        """Test converting a failed result to dict."""
        result = ToolResult(tool_name="fail", parameters={}, success=False, error="Bad times")
        expected = {"name": "fail", "parameters": {}, "result": None, "success": False, "error": "Bad times"}
        self.assertEqual(result.to_dict(), expected)

    def test_from_dict_success_tool_name(self):
        """Test creating a successful result from dict using tool_name."""
        data = {"tool_name": "dict_tool", "parameters": {"x": "y"}, "result": {"status": "ok"}, "success": True}
        result = ToolResult.from_dict(data)
        self.assertEqual(result.tool_name, "dict_tool")
        self.assertEqual(result.parameters, {"x": "y"})
        self.assertEqual(result.result, {"status": "ok"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_from_dict_success_name(self):
        """Test creating a successful result from dict using name."""
        data = {
            "name": "dict_tool2",  # Using deprecated name field
            "parameters": {"z": 1},
            "result": [1, 2],
            # success defaults to True if omitted
        }
        result = ToolResult.from_dict(data)
        self.assertEqual(result.tool_name, "dict_tool2")
        self.assertEqual(result.parameters, {"z": 1})
        self.assertEqual(result.result, [1, 2])
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_from_dict_failure(self):
        """Test creating a failed result from dict."""
        data = {"tool_name": "fail_dict", "parameters": {}, "result": None, "success": False, "error": "Load failed"}
        result = ToolResult.from_dict(data)
        self.assertEqual(result.tool_name, "fail_dict")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Load failed")

    def test_from_dict_missing_name(self):
        """Test from_dict raises KeyError if name is missing."""
        data = {"parameters": {}, "result": None}
        with self.assertRaisesRegex(KeyError, "Neither 'tool_name' nor 'name' found in data"):
            ToolResult.from_dict(data)

    def test_to_json(self):
        """Test converting result to JSON string."""
        result = ToolResult(tool_name="json_tool", parameters={"p": True}, result="json_res")
        json_str = result.to_json()
        # Check if it's a valid JSON string and contains expected data
        data = json.loads(json_str)
        self.assertEqual(data["name"], "json_tool")
        self.assertEqual(data["parameters"], {"p": True})
        self.assertEqual(data["result"], "json_res")
        self.assertTrue(data["success"])
        self.assertNotIn("error", data)

    def test_from_json_success(self):
        """Test creating result from JSON string."""
        json_str = '{"name": "json_tool", "parameters": {"p": true}, "result": "json_res", "success": true}'
        result = ToolResult.from_json(json_str)
        self.assertEqual(result.tool_name, "json_tool")
        self.assertEqual(result.parameters, {"p": True})
        self.assertEqual(result.result, "json_res")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_from_json_failure(self):
        """Test creating failed result from JSON string."""
        json_str = (
            '{"tool_name": "json_fail", "parameters": {}, "result": null, "success": false, "error": "json error"}'
        )
        result = ToolResult.from_json(json_str)
        self.assertEqual(result.tool_name, "json_fail")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "json error")

    def test_from_json_invalid(self):
        """Test from_json raises error for invalid JSON."""
        with self.assertRaises(json.JSONDecodeError):
            ToolResult.from_json("this is not json")


# Use unittest's async capabilities for TestTool
class TestToolAsync(unittest.IsolatedAsyncioTestCase):
    async def test_tool_execute_async(self):
        param1 = ToolParameter(name="p1", description="Param 1", type="string", required=True)
        handler_mock = AsyncMock(return_value={"processed": True})
        tool = Tool(
            name="async_test_tool",
            description="An async test tool",
            parameters=[param1],
            handler=handler_mock,
        )
        test_params = {"p1": "async_value"}
        result = await tool.execute(test_params)
        handler_mock.assert_awaited_once_with(test_params)
        self.assertEqual(result, {"processed": True})


if __name__ == "__main__":
    unittest.main()
