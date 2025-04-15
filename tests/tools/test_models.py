"""
Tests for the Tool models.
"""

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jsonschema import ValidationError

from src.cli_code.mcp.tools.models import Tool, ToolParameter, ToolResult


class TestToolParameter(unittest.TestCase):
    """Tests for the ToolParameter class."""

    def test_init_with_required_fields(self):
        """Test initialization with only required fields."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="string"
        )
        
        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.description, "Test parameter")
        self.assertEqual(param.type, "string")
        self.assertFalse(param.required)
        self.assertIsNone(param.enum)
        self.assertIsNone(param.properties)
        self.assertIsNone(param.items)
    
    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="object",
            required=True,
            enum=["value1", "value2"],
            properties={
                "prop1": {"type": "string"},
                "prop2": {"type": "number"}
            },
            items={"type": "string"}
        )
        
        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.description, "Test parameter")
        self.assertEqual(param.type, "object")
        self.assertTrue(param.required)
        self.assertEqual(param.enum, ["value1", "value2"])
        self.assertEqual(param.properties, {
            "prop1": {"type": "string"},
            "prop2": {"type": "number"}
        })
        self.assertEqual(param.items, {"type": "string"})
    
    def test_to_dict_with_required_fields(self):
        """Test to_dict with only required fields."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="string"
        )
        
        expected = {
            "name": "test_param",
            "description": "Test parameter",
            "type": "string"
        }
        
        self.assertEqual(param.to_dict(), expected)
    
    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="object",
            required=True,
            enum=["value1", "value2"],
            properties={
                "prop1": {"type": "string"},
                "prop2": {"type": "number"}
            },
            items={"type": "string"}
        )
        
        expected = {
            "name": "test_param",
            "description": "Test parameter",
            "type": "object",
            "required": True,
            "enum": ["value1", "value2"],
            "properties": {
                "prop1": {"type": "string"},
                "prop2": {"type": "number"}
            },
            "items": {"type": "string"}
        }
        
        self.assertEqual(param.to_dict(), expected)
    
    def test_to_schema_string_type(self):
        """Test to_schema with string type."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="string"
        )
        
        expected = {
            "type": "string",
            "description": "Test parameter"
        }
        
        self.assertEqual(param.to_schema(), expected)
    
    def test_to_schema_object_type(self):
        """Test to_schema with object type."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="object",
            properties={
                "prop1": {"type": "string"},
                "prop2": {"type": "number"}
            }
        )
        
        expected = {
            "type": "object",
            "description": "Test parameter",
            "properties": {
                "prop1": {"type": "string"},
                "prop2": {"type": "number"}
            }
        }
        
        self.assertEqual(param.to_schema(), expected)
    
    def test_to_schema_array_type(self):
        """Test to_schema with array type."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="array",
            items={"type": "string"}
        )
        
        expected = {
            "type": "array",
            "description": "Test parameter",
            "items": {"type": "string"}
        }
        
        self.assertEqual(param.to_schema(), expected)
    
    def test_to_schema_with_enum(self):
        """Test to_schema with enum."""
        param = ToolParameter(
            name="test_param",
            description="Test parameter",
            type="string",
            enum=["value1", "value2"]
        )
        
        expected = {
            "type": "string",
            "description": "Test parameter",
            "enum": ["value1", "value2"]
        }
        
        self.assertEqual(param.to_schema(), expected)


class TestTool(unittest.TestCase):
    """Tests for the Tool class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = AsyncMock(return_value={"result": "success"})
        
        self.params = [
            ToolParameter(
                name="param1",
                description="Parameter 1",
                type="string",
                required=True
            ),
            ToolParameter(
                name="param2",
                description="Parameter 2",
                type="number",
                required=False
            )
        ]
        
        self.tool = Tool(
            name="test_tool",
            description="Test tool",
            parameters=self.params,
            handler=self.handler
        )
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.tool.name, "test_tool")
        self.assertEqual(self.tool.description, "Test tool")
        self.assertEqual(len(self.tool.parameters), 2)
        self.assertEqual(self.tool.parameters[0].name, "param1")
        self.assertEqual(self.tool.parameters[1].name, "param2")
        self.assertEqual(self.tool.handler, self.handler)
    
    def test_schema(self):
        """Test schema generation."""
        expected = {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter 1"
                    },
                    "param2": {
                        "type": "number",
                        "description": "Parameter 2"
                    }
                },
                "required": ["param1"]
            }
        }
        
        self.assertEqual(self.tool.schema, expected)
    
    @pytest.mark.asyncio
    async def test_execute_with_valid_parameters(self):
        """Test execution with valid parameters."""
        parameters = {
            "param1": "value1",
            "param2": 42
        }
        
        result = await self.tool.execute(parameters)
        
        self.assertEqual(result, {"result": "success"})
        self.handler.assert_called_once_with(parameters)
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_required_parameter(self):
        """Test execution with missing required parameter."""
        parameters = {
            "param2": 42
        }
        
        with self.assertRaises(ValidationError):
            await self.tool.execute(parameters)
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_parameter_type(self):
        """Test execution with invalid parameter type."""
        parameters = {
            "param1": "value1",
            "param2": "not_a_number"
        }
        
        with self.assertRaises(ValidationError):
            await self.tool.execute(parameters)
    
    @pytest.mark.asyncio
    async def test_execute_with_extra_parameters(self):
        """Test execution with extra parameters."""
        parameters = {
            "param1": "value1",
            "param2": 42,
            "param3": "extra"
        }
        
        result = await self.tool.execute(parameters)
        
        self.assertEqual(result, {"result": "success"})
        self.handler.assert_called_once()
        # The handler should still be called with all parameters, including extra ones
        actual_parameters = self.handler.call_args[0][0]
        self.assertEqual(actual_parameters["param3"], "extra")
    
    def test_tool_with_no_parameters(self):
        """Test a tool with no parameters."""
        tool = Tool(
            name="no_params_tool",
            description="Tool with no parameters",
            parameters=[],
            handler=self.handler
        )
        
        expected_schema = {
            "name": "no_params_tool",
            "description": "Tool with no parameters",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        self.assertEqual(tool.schema, expected_schema)


class TestToolResult(unittest.TestCase):
    """Tests for the ToolResult class."""
    
    def test_init_success(self):
        """Test initialization with success."""
        result = ToolResult(
            name="test_tool",
            parameters={"param1": "value1"},
            result={"key": "value"},
            success=True
        )
        
        self.assertEqual(result.name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"key": "value"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    def test_init_failure(self):
        """Test initialization with failure."""
        result = ToolResult(
            name="test_tool",
            parameters={"param1": "value1"},
            result=None,
            success=False,
            error="Something went wrong"
        )
        
        self.assertEqual(result.name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Something went wrong")
    
    def test_to_dict_success(self):
        """Test to_dict with success."""
        result = ToolResult(
            name="test_tool",
            parameters={"param1": "value1"},
            result={"key": "value"},
            success=True
        )
        
        expected = {
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": {"key": "value"},
            "success": True
        }
        
        self.assertEqual(result.to_dict(), expected)
    
    def test_to_dict_failure(self):
        """Test to_dict with failure."""
        result = ToolResult(
            name="test_tool",
            parameters={"param1": "value1"},
            result=None,
            success=False,
            error="Something went wrong"
        )
        
        expected = {
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": None,
            "success": False,
            "error": "Something went wrong"
        }
        
        self.assertEqual(result.to_dict(), expected)
    
    def test_from_dict_success(self):
        """Test from_dict with success."""
        data = {
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": {"key": "value"},
            "success": True
        }
        
        result = ToolResult.from_dict(data)
        
        self.assertEqual(result.name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"key": "value"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    def test_from_dict_failure(self):
        """Test from_dict with failure."""
        data = {
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": None,
            "success": False,
            "error": "Something went wrong"
        }
        
        result = ToolResult.from_dict(data)
        
        self.assertEqual(result.name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Something went wrong")
    
    def test_to_json_success(self):
        """Test to_json with success."""
        result = ToolResult(
            name="test_tool",
            parameters={"param1": "value1"},
            result={"key": "value"},
            success=True
        )
        
        expected = json.dumps({
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": {"key": "value"},
            "success": True
        })
        
        self.assertEqual(result.to_json(), expected)
    
    def test_to_json_failure(self):
        """Test to_json with failure."""
        result = ToolResult(
            name="test_tool",
            parameters={"param1": "value1"},
            result=None,
            success=False,
            error="Something went wrong"
        )
        
        expected = json.dumps({
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": None,
            "success": False,
            "error": "Something went wrong"
        })
        
        self.assertEqual(result.to_json(), expected)
    
    def test_from_json_success(self):
        """Test from_json with success."""
        json_data = json.dumps({
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": {"key": "value"},
            "success": True
        })
        
        result = ToolResult.from_json(json_data)
        
        self.assertEqual(result.name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertEqual(result.result, {"key": "value"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    def test_from_json_failure(self):
        """Test from_json with failure."""
        json_data = json.dumps({
            "name": "test_tool",
            "parameters": {"param1": "value1"},
            "result": None,
            "success": False,
            "error": "Something went wrong"
        })
        
        result = ToolResult.from_json(json_data)
        
        self.assertEqual(result.name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Something went wrong") 