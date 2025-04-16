"""
Tests for the ToolService class in src.cli_code.mcp.tools.service.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.formatter import ToolResponseFormatter
from src.cli_code.mcp.tools.models import Tool, ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService


class TestToolService(unittest.TestCase):
    """Tests for the ToolService class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = MagicMock(spec=ToolRegistry)
        self.executor = MagicMock(spec=ToolExecutor)
        self.formatter = MagicMock(spec=ToolResponseFormatter)

        # Create the service with mocked dependencies
        with patch("src.cli_code.mcp.tools.service.ToolResponseFormatter", return_value=self.formatter):
            self.service = ToolService(self.registry, self.executor)

        # Sample tool for testing
        self.tool_schema = {
            "parameters": {
                "properties": {"param1": {"type": "string"}, "param2": {"type": "integer"}},
                "required": ["param1"],
            }
        }
        self.tool = MagicMock(spec=Tool)
        self.tool.name = "test_tool"
        self.tool.description = "Test tool for testing"
        self.tool.schema = self.tool_schema

        # Sample parameters
        self.params = {"param1": "value1", "param2": 42}

        # Sample tool result
        self.success_result = ToolResult(
            tool_name="test_tool", parameters=self.params, success=True, result={"output": "success"}, error=None
        )

        self.error_result = ToolResult(
            tool_name="test_tool", parameters=self.params, success=False, result=None, error="Test error"
        )

    def test_init(self):
        """Test initialization of the service."""
        # Test with provided executor
        service = ToolService(self.registry, self.executor)
        self.assertEqual(service.registry, self.registry)
        self.assertEqual(service.executor, self.executor)
        self.assertIsInstance(service.formatter, ToolResponseFormatter)

        # Test with default executor
        with patch("src.cli_code.mcp.tools.service.ToolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            service = ToolService(self.registry)
            self.assertEqual(service.registry, self.registry)
            self.assertEqual(service.executor, mock_executor)
            mock_executor_class.assert_called_once_with(self.registry)

    @patch("src.cli_code.mcp.tools.service.ToolExecutor")
    def test_init_creates_executor(self, mock_executor_class):
        """Test that init creates an executor if none is provided."""
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor

        service = ToolService(self.registry)
        mock_executor_class.assert_called_once_with(self.registry)
        self.assertEqual(service.executor, mock_executor)

    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        self.executor.execute = AsyncMock(return_value=self.success_result)

        result = await self.service.execute_tool("test_tool", self.params)

        self.executor.execute.assert_called_once_with("test_tool", self.params)
        self.assertEqual(result, self.success_result)
        self.assertTrue(result.success)
        self.assertEqual(result.result, {"output": "success"})
        self.assertIsNone(result.error)

    async def test_execute_tool_failure(self):
        """Test tool execution failure."""
        # Mock executor to raise an exception
        self.executor.execute = AsyncMock(side_effect=ValueError("Test error"))

        result = await self.service.execute_tool("test_tool", self.params)

        self.executor.execute.assert_called_once_with("test_tool", self.params)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Test error")

    async def test_execute_tool_call_json_string(self):
        """Test execution of a tool call with JSON string arguments."""
        # Set up mock to return a successful result
        self.executor.execute = AsyncMock(return_value=self.success_result)

        # Create a tool call with JSON string arguments
        tool_call = {"id": "call123", "function": {"name": "test_tool", "arguments": json.dumps(self.params)}}

        result = await self.service.execute_tool_call(tool_call)

        self.executor.execute.assert_called_once_with("test_tool", self.params)
        self.assertEqual(result["tool_call_id"], "call123")
        self.assertEqual(result["name"], "test_tool")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], json.dumps({"output": "success"}))

    async def test_execute_tool_call_dict_arguments(self):
        """Test execution of a tool call with dictionary arguments."""
        self.executor.execute = AsyncMock(return_value=self.success_result)

        # Create a tool call with dictionary arguments
        tool_call = {"id": "call123", "function": {"name": "test_tool", "arguments": self.params}}

        result = await self.service.execute_tool_call(tool_call)

        self.executor.execute.assert_called_once_with("test_tool", self.params)
        self.assertEqual(result["tool_call_id"], "call123")
        self.assertEqual(result["name"], "test_tool")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], json.dumps({"output": "success"}))

    async def test_execute_tool_call_invalid_json(self):
        """Test execution of a tool call with invalid JSON arguments."""
        # Create a tool call with invalid JSON arguments
        tool_call = {"id": "call123", "function": {"name": "test_tool", "arguments": "{invalid json"}}

        result = await self.service.execute_tool_call(tool_call)

        self.assertEqual(result["tool_call_id"], "call123")
        self.assertEqual(result["name"], "test_tool")
        self.assertEqual(result["status"], "error")
        self.assertTrue("Invalid JSON in tool call arguments" in result["content"])
        self.executor.execute.assert_not_called()

    async def test_execute_tool_call_error(self):
        """Test execution of a tool call that results in an error."""
        self.executor.execute = AsyncMock(return_value=self.error_result)

        tool_call = {"id": "call123", "function": {"name": "test_tool", "arguments": self.params}}

        result = await self.service.execute_tool_call(tool_call)

        self.executor.execute.assert_called_once_with("test_tool", self.params)
        self.assertEqual(result["tool_call_id"], "call123")
        self.assertEqual(result["name"], "test_tool")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["content"], "Test error")

    async def test_execute_tool_calls(self):
        """Test execution of multiple tool calls."""
        # Set up the mock to handle multiple calls
        self.service.execute_tool_call = AsyncMock(
            side_effect=[
                {"tool_call_id": "call1", "name": "tool1", "status": "success", "content": "result1"},
                {"tool_call_id": "call2", "name": "tool2", "status": "error", "content": "error2"},
            ]
        )

        tool_calls = [
            {"id": "call1", "function": {"name": "tool1", "arguments": {}}},
            {"id": "call2", "function": {"name": "tool2", "arguments": {}}},
        ]

        results = await self.service.execute_tool_calls(tool_calls)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["tool_call_id"], "call1")
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(results[1]["tool_call_id"], "call2")
        self.assertEqual(results[1]["status"], "error")

        self.service.execute_tool_call.assert_any_call(tool_calls[0])
        self.service.execute_tool_call.assert_any_call(tool_calls[1])

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        # Set up the registry to return our test tool
        self.registry.get_all_tools.return_value = [self.tool]

        definitions = self.service.get_tool_definitions()

        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "test_tool")
        self.assertEqual(definitions[0]["description"], "Test tool for testing")
        self.assertEqual(definitions[0]["parameters"], self.tool_schema["parameters"])
        self.registry.get_all_tools.assert_called_once()

    def test_get_tool_definition_by_name_found(self):
        """Test getting a tool definition by name when it exists."""
        # Set up the registry to return our test tool
        self.registry.get_tool.return_value = self.tool

        definition = self.service.get_tool_definition_by_name("test_tool")

        self.assertIsNotNone(definition)
        self.assertEqual(definition["name"], "test_tool")
        self.assertEqual(definition["description"], "Test tool for testing")
        self.assertEqual(definition["parameters"], self.tool_schema["parameters"])
        self.registry.get_tool.assert_called_once_with("test_tool")

    def test_get_tool_definition_by_name_not_found(self):
        """Test getting a tool definition by name when it doesn't exist."""
        # Set up the registry to raise a ValueError for nonexistent tool
        self.registry.get_tool.side_effect = ValueError("Tool not found")

        definition = self.service.get_tool_definition_by_name("nonexistent_tool")

        self.assertIsNone(definition)
        self.registry.get_tool.assert_called_once_with("nonexistent_tool")

    def test_format_result(self):
        """Test formatting a single tool result."""
        # Set up the formatter mock
        self.formatter.format_result.return_value = {"formatted": "result"}

        formatted = self.service.format_result(self.success_result)

        self.assertEqual(formatted, {"formatted": "result"})
        self.formatter.format_result.assert_called_once_with(self.success_result)

    def test_format_results(self):
        """Test formatting multiple tool results."""
        # Set up the formatter mock
        self.formatter.format_results.return_value = {"formatted": "results"}

        formatted = self.service.format_results([self.success_result, self.error_result])

        self.assertEqual(formatted, {"formatted": "results"})
        self.formatter.format_results.assert_called_once_with([self.success_result, self.error_result])

    def test_get_available_tools(self):
        """Test getting information about available tools."""
        # Set up the registry mock
        self.registry.get_schemas.return_value = {"test_tool": {"schema": "info"}}

        tools = self.service.get_available_tools()

        self.assertEqual(tools, {"test_tool": {"schema": "info"}})
        self.registry.get_schemas.assert_called_once()


if __name__ == "__main__":
    unittest.main()
