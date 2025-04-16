"""
Tests for the tools service module.

This module provides comprehensive tests for the ToolService class.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import jsonschema

from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.formatter import ToolResponseFormatter
from src.cli_code.mcp.tools.models import Tool, ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService


class TestToolService(unittest.TestCase):
    """Test case for the ToolService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.registry = MagicMock(spec=ToolRegistry)
        self.executor = MagicMock(spec=ToolExecutor)
        self.formatter = MagicMock(spec=ToolResponseFormatter)

        # Set up the service
        self.service = ToolService(self.registry, self.executor)
        self.service.formatter = self.formatter

        # Create mock tools
        self.tool1 = MagicMock(spec=Tool)
        self.tool1.name = "test_tool1"
        self.tool1.description = "Test tool 1"
        self.tool1.schema = {"parameters": {"type": "object", "properties": {"param1": {"type": "string"}}}}

        self.tool2 = MagicMock(spec=Tool)
        self.tool2.name = "test_tool2"
        self.tool2.description = "Test tool 2"
        self.tool2.schema = {"parameters": {"type": "object", "properties": {"param2": {"type": "integer"}}}}

        # Set up the registry to return mock tools
        self.registry.get_all_tools.return_value = [self.tool1, self.tool2]
        self.registry.get_tool.side_effect = lambda name: {"test_tool1": self.tool1, "test_tool2": self.tool2}.get(name)
        self.registry.get_schemas.return_value = {
            "test_tool1": {
                "name": "test_tool1",
                "description": "Test tool 1",
                "parameters": {"type": "object", "properties": {"param1": {"type": "string"}}},
            },
            "test_tool2": {
                "name": "test_tool2",
                "description": "Test tool 2",
                "parameters": {"type": "object", "properties": {"param2": {"type": "integer"}}},
            },
        }

    def test_init_with_executor(self):
        """Test initializing with an executor."""
        service = ToolService(self.registry, self.executor)
        self.assertEqual(service.registry, self.registry)
        self.assertEqual(service.executor, self.executor)
        self.assertIsInstance(service.formatter, ToolResponseFormatter)

    def test_init_without_executor(self):
        """Test initializing without an executor."""
        with patch("src.cli_code.mcp.tools.service.ToolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor

            service = ToolService(self.registry)

            self.assertEqual(service.registry, self.registry)
            self.assertEqual(service.executor, mock_executor)
            mock_executor_class.assert_called_once_with(self.registry)

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        definitions = self.service.get_tool_definitions()

        self.assertEqual(len(definitions), 2)
        self.assertEqual(definitions[0]["name"], "test_tool1")
        self.assertEqual(definitions[0]["description"], "Test tool 1")
        self.assertEqual(definitions[0]["parameters"], {"type": "object", "properties": {"param1": {"type": "string"}}})
        self.assertEqual(definitions[1]["name"], "test_tool2")
        self.assertEqual(definitions[1]["description"], "Test tool 2")
        self.assertEqual(
            definitions[1]["parameters"], {"type": "object", "properties": {"param2": {"type": "integer"}}}
        )

    def test_get_tool_definition_by_name_existing(self):
        """Test getting a tool definition by name for an existing tool."""
        definition = self.service.get_tool_definition_by_name("test_tool1")

        self.assertEqual(definition["name"], "test_tool1")
        self.assertEqual(definition["description"], "Test tool 1")
        self.assertEqual(definition["parameters"], {"type": "object", "properties": {"param1": {"type": "string"}}})

    def test_get_tool_definition_by_name_nonexistent(self):
        """Test getting a tool definition by name for a nonexistent tool."""
        self.registry.get_tool.side_effect = ValueError("Tool not found")

        definition = self.service.get_tool_definition_by_name("nonexistent_tool")

        self.assertIsNone(definition)

    def test_get_available_tools(self):
        """Test getting available tools."""
        tools = self.service.get_available_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools["test_tool1"]["name"], "test_tool1")
        self.assertEqual(tools["test_tool1"]["description"], "Test tool 1")
        self.assertEqual(tools["test_tool2"]["name"], "test_tool2")
        self.assertEqual(tools["test_tool2"]["description"], "Test tool 2")

    def test_format_result(self):
        """Test formatting a result."""
        # Create a result
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param": "value"},
            success=True,
            result={"output": "test_output"},
            error=None,
        )

        # Set up the formatter mock
        formatted_result = {"name": "test_tool", "parameters": {"param": "value"}, "result": {"output": "test_output"}}
        self.formatter.format_result.return_value = formatted_result

        # Format the result
        formatted = self.service.format_result(result)

        # Check the result
        self.assertEqual(formatted, formatted_result)
        self.formatter.format_result.assert_called_once_with(result)

    def test_format_results(self):
        """Test formatting multiple results."""
        # Create results
        results = [
            ToolResult(
                tool_name="test_tool1",
                parameters={"param1": "value1"},
                success=True,
                result={"output1": "test_output1"},
                error=None,
            ),
            ToolResult(
                tool_name="test_tool2",
                parameters={"param2": "value2"},
                success=False,
                result=None,
                error="Error message",
            ),
        ]

        # Set up the formatter mock
        formatted_results = {
            "results": [
                {"name": "test_tool1", "parameters": {"param1": "value1"}, "result": {"output1": "test_output1"}},
                {
                    "name": "test_tool2",
                    "parameters": {"param2": "value2"},
                    "error": {"type": "RuntimeError", "message": "Error message"},
                },
            ]
        }
        self.formatter.format_results.return_value = formatted_results

        # Format the results
        formatted = self.service.format_results(results)

        # Check the result
        self.assertEqual(formatted, formatted_results)
        self.formatter.format_results.assert_called_once_with(results)

    def test_execute_tool_success(self):
        """Test executing a tool successfully."""
        # Create a success result
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param": "value"},
            success=True,
            result={"output": "test_output"},
            error=None,
        )

        # Set up the executor mock to return the success result
        self.executor.execute = AsyncMock(return_value=result)

        # Execute the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            actual_result = loop.run_until_complete(self.service.execute_tool("test_tool", {"param": "value"}))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(actual_result, result)
        self.executor.execute.assert_called_once_with("test_tool", {"param": "value"})

    def test_execute_tool_error(self):
        """Test executing a tool that raises an error."""
        # Set up the executor mock to raise an exception
        error = Exception("Test error")
        self.executor.execute = AsyncMock(side_effect=error)

        # Execute the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.service.execute_tool("test_tool", {"param": "value"}))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param": "value"})
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Test error")
        self.executor.execute.assert_called_once_with("test_tool", {"param": "value"})

    def test_execute_tool_call_success(self):
        """Test executing a tool call successfully."""
        # Create a success result
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param": "value"},
            success=True,
            result={"output": "test_output"},
            error=None,
        )

        # Set up the service's execute_tool method to return the success result
        self.service.execute_tool = AsyncMock(return_value=result)

        # Create a tool call
        tool_call = {"id": "call_123", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}

        # Execute the tool call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.service.execute_tool_call(tool_call))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(response["tool_call_id"], "call_123")
        self.assertEqual(response["name"], "test_tool")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["content"], json.dumps({"output": "test_output"}))
        self.service.execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    def test_execute_tool_call_invalid_json(self):
        """Test executing a tool call with invalid JSON arguments."""
        # Save the original method to restore it later
        original_execute_tool = self.service.execute_tool

        # Create a mock for execute_tool to track calls
        mock_execute_tool = AsyncMock()
        self.service.execute_tool = mock_execute_tool

        try:
            # Create a tool call with invalid JSON
            tool_call = {"id": "call_123", "function": {"name": "test_tool", "arguments": "{invalid json"}}

            # Execute the tool call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self.service.execute_tool_call(tool_call))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

            # Check the result
            self.assertEqual(response["tool_call_id"], "call_123")
            self.assertEqual(response["name"], "test_tool")
            self.assertEqual(response["status"], "error")
            self.assertTrue("Invalid JSON in tool call arguments" in response["content"])
            mock_execute_tool.assert_not_called()
        finally:
            # Restore the original method
            self.service.execute_tool = original_execute_tool

    def test_execute_tool_call_dict_arguments(self):
        """Test executing a tool call with dictionary arguments."""
        # Create a success result
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param": "value"},
            success=True,
            result={"output": "test_output"},
            error=None,
        )

        # Set up the service's execute_tool method to return the success result
        self.service.execute_tool = AsyncMock(return_value=result)

        # Create a tool call with dictionary arguments
        tool_call = {"id": "call_123", "function": {"name": "test_tool", "arguments": {"param": "value"}}}

        # Execute the tool call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.service.execute_tool_call(tool_call))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(response["tool_call_id"], "call_123")
        self.assertEqual(response["name"], "test_tool")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["content"], json.dumps({"output": "test_output"}))
        self.service.execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    def test_execute_tool_call_error(self):
        """Test executing a tool call that results in an error."""
        # Create an error result
        result = ToolResult(
            tool_name="test_tool", parameters={"param": "value"}, success=False, result=None, error="Test error"
        )

        # Set up the service's execute_tool method to return the error result
        self.service.execute_tool = AsyncMock(return_value=result)

        # Create a tool call
        tool_call = {"id": "call_123", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}

        # Execute the tool call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.service.execute_tool_call(tool_call))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(response["tool_call_id"], "call_123")
        self.assertEqual(response["name"], "test_tool")
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["content"], "Test error")
        self.service.execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    def test_execute_tool_calls(self):
        """Test executing multiple tool calls."""
        # Set up the service's execute_tool_call method to return mock responses
        responses = [
            {
                "tool_call_id": "call_1",
                "name": "test_tool1",
                "status": "success",
                "content": json.dumps({"output": "test_output1"}),
            },
            {"tool_call_id": "call_2", "name": "test_tool2", "status": "error", "content": "Error message"},
        ]
        self.service.execute_tool_call = AsyncMock(side_effect=responses)

        # Create tool calls
        tool_calls = [
            {"id": "call_1", "function": {"name": "test_tool1", "arguments": json.dumps({"param1": "value1"})}},
            {"id": "call_2", "function": {"name": "test_tool2", "arguments": json.dumps({"param2": "value2"})}},
        ]

        # Execute the tool calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self.service.execute_tool_calls(tool_calls))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results, responses)
        self.assertEqual(self.service.execute_tool_call.call_count, 2)
        self.service.execute_tool_call.assert_any_call(tool_calls[0])
        self.service.execute_tool_call.assert_any_call(tool_calls[1])


if __name__ == "__main__":
    unittest.main()
