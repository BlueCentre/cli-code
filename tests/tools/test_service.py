"""
Tests for the ToolService class.
"""

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli_code.mcp.tools.models import Tool, ToolParameter, ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.service import ToolService


class TestToolService(unittest.TestCase):
    """Tests for the ToolService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock tools for testing
        self.calculator_handler = AsyncMock(return_value={"result": 42})
        self.weather_handler = AsyncMock(return_value={"temperature": 72, "condition": "sunny"})
        
        # Create calculator tool with parameters
        calculator_params = [
            ToolParameter(
                name="operation",
                description="The operation to perform",
                type="string",
                required=True,
                enum=["add", "subtract", "multiply", "divide"]
            ),
            ToolParameter(
                name="a",
                description="First operand",
                type="number",
                required=True
            ),
            ToolParameter(
                name="b",
                description="Second operand",
                type="number",
                required=True
            )
        ]
        
        self.calculator_tool = Tool(
            name="calculator",
            description="Perform a calculation",
            parameters=calculator_params,
            handler=self.calculator_handler
        )
        
        # Create weather tool with parameters
        weather_params = [
            ToolParameter(
                name="location",
                description="The location to get weather for",
                type="string",
                required=True
            ),
            ToolParameter(
                name="units",
                description="Temperature units",
                type="string",
                enum=["celsius", "fahrenheit"],
                required=False
            )
        ]
        
        self.weather_tool = Tool(
            name="weather",
            description="Get weather information",
            parameters=weather_params,
            handler=self.weather_handler
        )
        
        # Create a registry with both tools
        self.registry = ToolRegistry()
        self.registry.register(self.calculator_tool)
        self.registry.register(self.weather_tool)
        
        # Create an executor
        self.executor = ToolExecutor(self.registry)
        
        # Create the service
        self.service = ToolService(self.registry, self.executor)
    
    def test_initialization(self):
        """Test ToolService initialization."""
        self.assertEqual(self.service.registry, self.registry)
        self.assertEqual(self.service.executor, self.executor)
    
    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        definitions = self.service.get_tool_definitions()
        
        # Should be a list with 2 tools
        self.assertEqual(len(definitions), 2)
        
        # Check that each tool has the expected fields
        for tool_def in definitions:
            self.assertIn("name", tool_def)
            self.assertIn("description", tool_def)
            self.assertIn("parameters", tool_def)
            
            # Tools should be either calculator or weather
            self.assertIn(tool_def["name"], ["calculator", "weather"])
            
            # Check parameters structure
            params = tool_def["parameters"]
            self.assertIn("type", params)
            self.assertEqual(params["type"], "object")
            self.assertIn("properties", params)
            self.assertIn("required", params)
            
            if tool_def["name"] == "calculator":
                # Calculator should have 3 parameters
                self.assertEqual(len(params["properties"]), 3)
                self.assertIn("operation", params["properties"])
                self.assertIn("a", params["properties"])
                self.assertIn("b", params["properties"])
                self.assertEqual(len(params["required"]), 3)
            
            elif tool_def["name"] == "weather":
                # Weather should have 2 parameters
                self.assertEqual(len(params["properties"]), 2)
                self.assertIn("location", params["properties"])
                self.assertIn("units", params["properties"])
                self.assertEqual(len(params["required"]), 1)
    
    def test_get_tool_definition_by_name_existing(self):
        """Test getting a specific tool definition by name when it exists."""
        calculator_def = self.service.get_tool_definition_by_name("calculator")
        
        self.assertIsNotNone(calculator_def)
        self.assertEqual(calculator_def["name"], "calculator")
        self.assertEqual(calculator_def["description"], "Perform a calculation")
        
        # Check parameters
        params = calculator_def["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertEqual(len(params["properties"]), 3)
        self.assertEqual(len(params["required"]), 3)
    
    def test_get_tool_definition_by_name_nonexistent(self):
        """Test getting a specific tool definition by name when it doesn't exist."""
        nonexistent_def = self.service.get_tool_definition_by_name("nonexistent")
        
        self.assertIsNone(nonexistent_def)
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test executing a tool successfully."""
        # Mock the executor to return a successful result
        result_dict = {
            "name": "calculator",
            "parameters": {"operation": "add", "a": 1, "b": 2},
            "result": {"result": 3},
            "success": True
        }
        mock_result = ToolResult.from_dict(result_dict)
        self.executor.execute = AsyncMock(return_value=mock_result)
        
        # Execute the tool
        result = await self.service.execute_tool(
            "calculator",
            {"operation": "add", "a": 1, "b": 2}
        )
        
        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.name, "calculator")
        self.assertEqual(result.parameters, {"operation": "add", "a": 1, "b": 2})
        self.assertEqual(result.result, {"result": 3})
        
        # Verify that the executor was called with the right arguments
        self.executor.execute.assert_called_once_with("calculator", {"operation": "add", "a": 1, "b": 2})
    
    @pytest.mark.asyncio
    async def test_execute_tool_failure(self):
        """Test executing a tool that fails."""
        # Mock the executor to return a failed result
        result_dict = {
            "name": "calculator",
            "parameters": {"operation": "divide", "a": 1, "b": 0},
            "result": None,
            "success": False,
            "error": "Division by zero"
        }
        mock_result = ToolResult.from_dict(result_dict)
        self.executor.execute = AsyncMock(return_value=mock_result)
        
        # Execute the tool
        result = await self.service.execute_tool(
            "calculator",
            {"operation": "divide", "a": 1, "b": 0}
        )
        
        # Check the result
        self.assertFalse(result.success)
        self.assertEqual(result.name, "calculator")
        self.assertEqual(result.parameters, {"operation": "divide", "a": 1, "b": 0})
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Division by zero")
        
        # Verify that the executor was called with the right arguments
        self.executor.execute.assert_called_once_with("calculator", {"operation": "divide", "a": 1, "b": 0})
    
    @pytest.mark.asyncio
    async def test_execute_tool_nonexistent(self):
        """Test executing a tool that doesn't exist."""
        # Mock the executor to raise an exception for a non-existent tool
        self.executor.execute = AsyncMock(side_effect=ValueError("Tool 'nonexistent' not found"))
        
        # Execute the tool
        result = await self.service.execute_tool("nonexistent", {})
        
        # Check the result
        self.assertFalse(result.success)
        self.assertEqual(result.name, "nonexistent")
        self.assertEqual(result.parameters, {})
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Tool 'nonexistent' not found")
        
        # Verify that the executor was called with the right arguments
        self.executor.execute.assert_called_once_with("nonexistent", {})
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_json_args(self):
        """Test executing a tool call with JSON arguments."""
        # Create a tool call object with JSON arguments
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": json.dumps({
                    "operation": "add",
                    "a": 1,
                    "b": 2
                })
            }
        }
        
        # Mock the executor to return a successful result
        result_dict = {
            "name": "calculator",
            "parameters": {"operation": "add", "a": 1, "b": 2},
            "result": {"result": 3},
            "success": True
        }
        mock_result = ToolResult.from_dict(result_dict)
        self.executor.execute = AsyncMock(return_value=mock_result)
        
        # Execute the tool call
        result = await self.service.execute_tool_call(tool_call)
        
        # Check the result
        self.assertEqual(result["tool_call_id"], "call_123")
        self.assertEqual(result["name"], "calculator")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], '{"result": 3}')
        
        # Verify that the executor was called with the right arguments
        self.executor.execute.assert_called_once_with(
            "calculator", 
            {"operation": "add", "a": 1, "b": 2}
        )
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_object_args(self):
        """Test executing a tool call with object arguments."""
        # Create a tool call object with object arguments
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": {
                    "operation": "add",
                    "a": 1,
                    "b": 2
                }
            }
        }
        
        # Mock the executor to return a successful result
        result_dict = {
            "name": "calculator",
            "parameters": {"operation": "add", "a": 1, "b": 2},
            "result": {"result": 3},
            "success": True
        }
        mock_result = ToolResult.from_dict(result_dict)
        self.executor.execute = AsyncMock(return_value=mock_result)
        
        # Execute the tool call
        result = await self.service.execute_tool_call(tool_call)
        
        # Check the result
        self.assertEqual(result["tool_call_id"], "call_123")
        self.assertEqual(result["name"], "calculator")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], '{"result": 3}')
        
        # Verify that the executor was called with the right arguments
        self.executor.execute.assert_called_once_with(
            "calculator", 
            {"operation": "add", "a": 1, "b": 2}
        )
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_invalid_json(self):
        """Test executing a tool call with invalid JSON."""
        tool_call = {
            "id": "test_id",
            "type": "function",
            "function": {
                "name": "calc_tool",
                "arguments": "invalid json"
            }
        }

        result = await self.service.execute_tool_call(tool_call)

        self.assertEqual(result["tool_call_id"], "test_id")
        self.assertEqual(result["name"], "calc_tool")
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid JSON", result["content"])

    @pytest.mark.asyncio
    async def test_execute_tool_call_error_handling(self):
        """Test error handling in execute_tool_call."""
        # Setup mock to raise exception
        self.executor.execute.side_effect = ValueError("Test error")
        
        tool_call = {
            "id": "test_id",
            "type": "function",
            "function": {
                "name": "calc_tool",
                "arguments": {"param1": "value1"}
            }
        }

        result = await self.service.execute_tool_call(tool_call)

        self.assertEqual(result["tool_call_id"], "test_id")
        self.assertEqual(result["name"], "calc_tool")
        self.assertEqual(result["status"], "error")
        self.assertIn("Test error", result["content"])

    @pytest.mark.asyncio
    async def test_execute_tool_call_without_id(self):
        """Test executing a tool call without an ID."""
        tool_call = {
            "type": "function",
            "function": {
                "name": "calc_tool",
                "arguments": {"param1": "value1"}
            }
        }

        result = await self.service.execute_tool_call(tool_call)

        self.assertEqual(result["tool_call_id"], "unknown")
        self.assertEqual(result["name"], "calc_tool")
        self.assertEqual(result["status"], "success")

    @pytest.mark.asyncio
    async def test_execute_tool_call_without_name(self):
        """Test executing a tool call without a name."""
        tool_call = {
            "id": "test_id",
            "type": "function",
            "function": {
                "arguments": {"param1": "value1"}
            }
        }

        result = await self.service.execute_tool_call(tool_call)

        self.assertEqual(result["tool_call_id"], "test_id")
        self.assertEqual(result["name"], "unknown")
        self.assertEqual(result["status"], "error")

    @pytest.mark.asyncio
    async def test_execute_tool_with_exception(self):
        """Test execute_tool with an exception."""
        # Setup to raise exception during execution
        self.executor.execute.side_effect = Exception("Unexpected error")
        
        result = await self.service.execute_tool("calc_tool", {"param1": "value1"})
        
        # Verify the result
        self.assertEqual(result.tool_name, "calc_tool")
        self.assertEqual(result.parameters, {"param1": "value1"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Unexpected error")
        self.assertIsNone(result.result)

    @pytest.mark.asyncio
    async def test_format_result(self):
        """Test the format_result method."""
        # Create a ToolResult
        tool_result = ToolResult(
            tool_name="calc_tool",
            parameters={"param1": "value1"},
            result={"answer": 42},
            success=True
        )
        
        # Format the result
        formatted = self.service.format_result(tool_result)
        
        # Verify the format
        self.assertEqual(formatted["name"], "calc_tool")
        self.assertEqual(formatted["parameters"], {"param1": "value1"})
        self.assertEqual(formatted["result"], {"answer": 42})
        self.assertEqual(formatted["success"], True)
        self.assertNotIn("error", formatted)

    @pytest.mark.asyncio
    async def test_format_result_error(self):
        """Test formatting an error result."""
        # Create a failed ToolResult
        tool_result = ToolResult(
            tool_name="calc_tool",
            parameters={"param1": "value1"},
            result=None,
            success=False,
            error="Division by zero"
        )
        
        # Format the result
        formatted = self.service.format_result(tool_result)
        
        # Verify the format
        self.assertEqual(formatted["name"], "calc_tool")
        self.assertEqual(formatted["parameters"], {"param1": "value1"})
        self.assertIsNone(formatted["result"])
        self.assertEqual(formatted["success"], False)
        self.assertEqual(formatted["error"], "Division by zero")
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_single(self):
        """Test executing a single tool call."""
        # Create a tool call object
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": json.dumps({
                    "operation": "add",
                    "a": 1,
                    "b": 2
                })
            }
        }
        
        # Mock the executor to return a successful result
        result_dict = {
            "name": "calculator",
            "parameters": {"operation": "add", "a": 1, "b": 2},
            "result": {"result": 3},
            "success": True
        }
        mock_result = ToolResult.from_dict(result_dict)
        self.executor.execute = AsyncMock(return_value=mock_result)
        
        # Execute the tool calls
        results = await self.service.execute_tool_calls([tool_call])
        
        # Check the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool_call_id"], "call_123")
        self.assertEqual(results[0]["name"], "calculator")
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(results[0]["content"], '{"result": 3}')
        
        # Verify that the executor was called with the right arguments
        self.executor.execute.assert_called_once_with(
            "calculator", 
            {"operation": "add", "a": 1, "b": 2}
        )
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_multiple(self):
        """Test executing multiple tool calls."""
        # Create tool call objects
        calculator_call = {
            "id": "calc_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": json.dumps({
                    "operation": "add",
                    "a": 1,
                    "b": 2
                })
            }
        }
        
        weather_call = {
            "id": "weather_456",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": json.dumps({
                    "location": "New York",
                    "units": "celsius"
                })
            }
        }
        
        # Mock the executor to return different results for different tools
        async def mock_execute(name, params):
            if name == "calculator":
                return ToolResult(
                    name="calculator",
                    parameters={"operation": "add", "a": 1, "b": 2},
                    result={"result": 3},
                    success=True
                )
            elif name == "weather":
                return ToolResult(
                    name="weather",
                    parameters={"location": "New York", "units": "celsius"},
                    result={"temperature": 20, "condition": "sunny"},
                    success=True
                )
        
        self.executor.execute = AsyncMock(side_effect=mock_execute)
        
        # Execute the tool calls
        results = await self.service.execute_tool_calls([calculator_call, weather_call])
        
        # Check the results
        self.assertEqual(len(results), 2)
        
        # Check calculator result
        calc_result = next(r for r in results if r["tool_call_id"] == "calc_123")
        self.assertEqual(calc_result["name"], "calculator")
        self.assertEqual(calc_result["status"], "success")
        self.assertEqual(calc_result["content"], '{"result": 3}')
        
        # Check weather result
        weather_result = next(r for r in results if r["tool_call_id"] == "weather_456")
        self.assertEqual(weather_result["name"], "weather")
        self.assertEqual(weather_result["status"], "success")
        self.assertEqual(weather_result["content"], '{"temperature": 20, "condition": "sunny"}')
        
        # Verify that the executor was called twice with the right arguments
        self.assertEqual(self.executor.execute.call_count, 2)
        call_args_list = self.executor.execute.call_args_list
        
        # First call should be for calculator
        self.assertEqual(call_args_list[0][0][0], "calculator")
        self.assertEqual(call_args_list[0][0][1], {"operation": "add", "a": 1, "b": 2})
        
        # Second call should be for weather
        self.assertEqual(call_args_list[1][0][0], "weather")
        self.assertEqual(call_args_list[1][0][1], {"location": "New York", "units": "celsius"})
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_empty(self):
        """Test executing an empty list of tool calls."""
        # Execute empty tool calls
        results = await self.service.execute_tool_calls([])
        
        # Check the results
        self.assertEqual(len(results), 0)
        
        # Verify that the executor was not called
        self.executor.execute.assert_not_called() 