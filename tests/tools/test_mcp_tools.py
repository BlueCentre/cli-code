"""
Tests for MCP tools modules.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.formatter import ToolResponseFormatter
from src.cli_code.mcp.tools.models import Tool, ToolParameter, ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService


class TestToolRegistry(unittest.TestCase):
    """Tests for the ToolRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()

        # Create a mock tool
        self.tool_handler = MagicMock()
        self.tool = Tool(
            name="test_tool",
            description="Test tool for testing",
            parameters=[ToolParameter(name="param1", description="Test parameter", type="string", required=True)],
            handler=self.tool_handler,
        )

    def test_register_tool(self):
        """Test registering a tool."""
        self.registry.register(self.tool)
        self.assertIn("test_tool", self.registry.list_tools())

    def test_register_duplicate_tool(self):
        """Test registering a duplicate tool raises an error."""
        self.registry.register(self.tool)

        # Create a duplicate tool
        duplicate_tool = Tool(name="test_tool", description="Duplicate test tool", parameters=[], handler=MagicMock())

        with self.assertRaises(ValueError):
            self.registry.register(duplicate_tool)

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        self.registry.register(self.tool)
        self.registry.unregister("test_tool")
        self.assertNotIn("test_tool", self.registry.list_tools())

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a nonexistent tool raises an error."""
        with self.assertRaises(ValueError):
            self.registry.unregister("nonexistent_tool")

    def test_get_tool(self):
        """Test getting a tool by name."""
        self.registry.register(self.tool)
        retrieved_tool = self.registry.get_tool("test_tool")
        self.assertEqual(retrieved_tool, self.tool)

    def test_get_nonexistent_tool(self):
        """Test getting a nonexistent tool raises an error."""
        with self.assertRaises(ValueError):
            self.registry.get_tool("nonexistent_tool")

    def test_list_tools(self):
        """Test listing all tool names."""
        self.registry.register(self.tool)

        # Create another tool
        another_tool = Tool(name="another_tool", description="Another test tool", parameters=[], handler=MagicMock())
        self.registry.register(another_tool)

        tool_names = self.registry.list_tools()
        self.assertEqual(len(tool_names), 2)
        self.assertIn("test_tool", tool_names)
        self.assertIn("another_tool", tool_names)

    def test_get_all_tools(self):
        """Test getting all tools."""
        self.registry.register(self.tool)
        tools = self.registry.get_all_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0], self.tool)

    def test_get_schemas(self):
        """Test getting schemas for all tools."""
        self.registry.register(self.tool)
        schemas = self.registry.get_schemas()
        self.assertIn("test_tool", schemas)
        self.assertEqual(schemas["test_tool"]["name"], "test_tool")
        self.assertEqual(schemas["test_tool"]["description"], "Test tool for testing")


class TestToolExecutor(unittest.TestCase):
    """Tests for the ToolExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

        # Create a mock tool with an async handler
        self.tool_result = "Test result"
        self.tool_handler = AsyncMock(return_value=self.tool_result)
        self.tool = Tool(
            name="test_tool",
            description="Test tool for testing",
            parameters=[ToolParameter(name="param1", description="Test parameter", type="string", required=True)],
            handler=self.tool_handler,
        )
        self.registry.register(self.tool)

    def test_validate_parameters_valid(self):
        """Test validating valid parameters."""
        parameters = {"param1": "test_value"}
        # Should not raise an exception
        self.executor.validate_parameters(self.tool, parameters)

    def test_validate_parameters_invalid(self):
        """Test validating invalid parameters raises an error."""
        # Missing required parameter
        parameters = {}
        with self.assertRaises(ValueError):
            self.executor.validate_parameters(self.tool, parameters)

    async def test_execute_success(self):
        """Test executing a tool successfully."""
        parameters = {"param1": "test_value"}
        result = await self.executor.execute("test_tool", parameters)

        # Verify that the handler was called with the correct parameters
        self.tool_handler.assert_called_with(**parameters)

        # Verify the result
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, parameters)
        self.assertEqual(result.result, self.tool_result)

    async def test_execute_tool_not_found(self):
        """Test executing a nonexistent tool raises an error."""
        with self.assertRaises(ValueError):
            await self.executor.execute("nonexistent_tool", {})

    async def test_execute_validation_error(self):
        """Test executing a tool with invalid parameters raises an error."""
        # Missing required parameter
        parameters = {}
        with self.assertRaises(ValueError):
            await self.executor.execute("test_tool", parameters)


class TestToolResponseFormatter(unittest.TestCase):
    """Tests for the ToolResponseFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ToolResponseFormatter()

        # Create a test result
        self.result = ToolResult(tool_name="test_tool", parameters={"param1": "test_value"}, result="Test result")

    def test_format_result(self):
        """Test formatting a single result."""
        formatted = self.formatter.format_result(self.result)
        self.assertEqual(formatted["name"], "test_tool")
        self.assertEqual(formatted["parameters"], {"param1": "test_value"})
        self.assertEqual(formatted["result"], "Test result")

    def test_format_results(self):
        """Test formatting multiple results."""
        formatted = self.formatter.format_results([self.result])
        self.assertEqual(len(formatted["results"]), 1)
        self.assertEqual(formatted["results"][0]["name"], "test_tool")

    def test_format_error(self):
        """Test formatting an error."""
        error = ValueError("Test error")
        formatted = self.formatter.format_error(error, "test_tool", {"param1": "test_value"})
        self.assertEqual(formatted["name"], "test_tool")
        self.assertEqual(formatted["parameters"], {"param1": "test_value"})
        self.assertEqual(formatted["error"]["type"], "ValueError")
        self.assertEqual(formatted["error"]["message"], "Test error")


class TestToolService(unittest.TestCase):
    """Tests for the ToolService class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.service = ToolService(self.registry)

        # Create a mock tool with an async handler
        self.tool_result = "Test result"
        self.tool_handler = AsyncMock(return_value=self.tool_result)
        self.tool = Tool(
            name="test_tool",
            description="Test tool for testing",
            parameters=[ToolParameter(name="param1", description="Test parameter", type="string", required=True)],
            handler=self.tool_handler,
        )
        self.registry.register(self.tool)

    async def test_execute_tool_success(self):
        """Test executing a tool successfully."""
        parameters = {"param1": "test_value"}
        result = await self.service.execute_tool("test_tool", parameters)

        # Verify the result
        self.assertEqual(result["name"], "test_tool")
        self.assertEqual(result["parameters"], parameters)
        self.assertEqual(result["result"], self.tool_result)

    async def test_execute_tool_not_found(self):
        """Test executing a nonexistent tool returns an error."""
        result = await self.service.execute_tool("nonexistent_tool", {})

        # Verify the error
        self.assertEqual(result["name"], "nonexistent_tool")
        self.assertEqual(result["parameters"], {})
        self.assertIn("error", result)
        self.assertEqual(result["error"]["type"], "ValueError")

    async def test_execute_tools(self):
        """Test executing multiple tools."""
        tools = [{"name": "test_tool", "parameters": {"param1": "test_value"}}]
        result = await self.service.execute_tools(tools)

        # Verify the result
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["name"], "test_tool")
        self.assertEqual(result["results"][0]["result"], self.tool_result)

    async def test_execute_tools_with_errors(self):
        """Test executing multiple tools with errors."""
        tools = [
            {"name": "test_tool", "parameters": {"param1": "test_value"}},
            {"name": "nonexistent_tool", "parameters": {}},
        ]
        result = await self.service.execute_tools(tools)

        # Verify the result
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["name"], "test_tool")
        self.assertEqual(result["results"][0]["result"], self.tool_result)

        # The second tool doesn't exist, so it should create a ToolResult with success=False
        self.assertEqual(result["results"][1]["name"], "nonexistent_tool")
        self.assertFalse(result["results"][1]["success"])

    def test_get_available_tools(self):
        """Test getting available tools."""
        tools = self.service.get_available_tools()
        self.assertIn("test_tool", tools)
        self.assertEqual(tools["test_tool"]["name"], "test_tool")


# Run the async tests
def run_async_test(coro):
    """Run an async test coroutine."""
    return asyncio.run(coro)
