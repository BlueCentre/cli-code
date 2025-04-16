"""
Tests for the tool registry.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli_code.mcp.tools.models import Tool, ToolParameter
from src.cli_code.mcp.tools.registry import ToolRegistry


class TestToolRegistry(unittest.TestCase):
    """Tests for the ToolRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()

        # Create a test tool
        self.handler = AsyncMock(return_value={"result": "success"})
        self.params = [
            ToolParameter(name="param1", description="Parameter 1", type="string", required=True),
            ToolParameter(name="param2", description="Parameter 2", type="number", required=False),
        ]

        self.test_tool = Tool(name="test_tool", description="Test tool", parameters=self.params, handler=self.handler)

        # Second test tool
        self.second_tool = Tool(
            name="second_tool",
            description="Second test tool",
            parameters=[
                ToolParameter(
                    name="option",
                    description="Option parameter",
                    type="string",
                    required=True,
                    enum=["option1", "option2"],
                )
            ],
            handler=AsyncMock(return_value={"status": "done"}),
        )

    def test_init(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        self.assertEqual(len(registry.list_tools()), 0)
        self.assertEqual(registry.get_all_tools(), [])
        self.assertEqual(registry.get_schemas(), {})

    def test_register_tool(self):
        """Test registering a tool."""
        self.registry.register(self.test_tool)

        self.assertEqual(len(self.registry.list_tools()), 1)
        self.assertIn("test_tool", self.registry.list_tools())

        # Get tool and verify it's the same
        retrieved_tool = self.registry.get_tool("test_tool")
        self.assertEqual(retrieved_tool.name, "test_tool")
        self.assertEqual(retrieved_tool.description, "Test tool")
        self.assertEqual(len(retrieved_tool.parameters), 2)

    def test_register_duplicate_tool(self):
        """Test registering a tool with the same name twice raises an error."""
        # Register the first time
        self.registry.register(self.test_tool)

        # Register again with the same name
        with self.assertRaises(ValueError) as context:
            self.registry.register(self.test_tool)

        self.assertIn("is already registered", str(context.exception))

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        # Register and then unregister
        self.registry.register(self.test_tool)
        self.assertIn("test_tool", self.registry.list_tools())

        self.registry.unregister("test_tool")
        self.assertNotIn("test_tool", self.registry.list_tools())
        self.assertEqual(len(self.registry.list_tools()), 0)

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a tool that does not exist raises an error."""
        with self.assertRaises(ValueError) as context:
            self.registry.unregister("nonexistent_tool")

        self.assertIn("No tool with name", str(context.exception))

    def test_get_tool(self):
        """Test getting a tool by name."""
        self.registry.register(self.test_tool)

        tool = self.registry.get_tool("test_tool")
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "Test tool")
        self.assertEqual(len(tool.parameters), 2)

    def test_get_nonexistent_tool(self):
        """Test getting a tool that does not exist raises an error."""
        with self.assertRaises(ValueError) as context:
            self.registry.get_tool("nonexistent_tool")

        self.assertIn("No tool with name", str(context.exception))

    def test_list_tools(self):
        """Test listing all tool names."""
        # Empty registry
        self.assertEqual(self.registry.list_tools(), [])

        # Add tools
        self.registry.register(self.test_tool)
        self.registry.register(self.second_tool)

        # Check the list
        tool_names = self.registry.list_tools()
        self.assertEqual(len(tool_names), 2)
        self.assertIn("test_tool", tool_names)
        self.assertIn("second_tool", tool_names)

    def test_get_all_tools(self):
        """Test getting all tools."""
        # Empty registry
        self.assertEqual(self.registry.get_all_tools(), [])

        # Add tools
        self.registry.register(self.test_tool)
        self.registry.register(self.second_tool)

        # Check the list
        tools = self.registry.get_all_tools()
        self.assertEqual(len(tools), 2)

        # Check tool names
        tool_names = [tool.name for tool in tools]
        self.assertIn("test_tool", tool_names)
        self.assertIn("second_tool", tool_names)

    def test_get_schemas(self):
        """Test getting schemas for all tools."""
        # Empty registry
        self.assertEqual(self.registry.get_schemas(), {})

        # Add tools
        self.registry.register(self.test_tool)
        self.registry.register(self.second_tool)

        # Get schemas
        schemas = self.registry.get_schemas()
        self.assertEqual(len(schemas), 2)
        self.assertIn("test_tool", schemas)
        self.assertIn("second_tool", schemas)

        # Check schema structure
        test_tool_schema = schemas["test_tool"]
        self.assertEqual(test_tool_schema["name"], "test_tool")
        self.assertEqual(test_tool_schema["description"], "Test tool")
        self.assertIn("parameters", test_tool_schema)

        second_tool_schema = schemas["second_tool"]
        self.assertEqual(second_tool_schema["name"], "second_tool")
        self.assertEqual(second_tool_schema["description"], "Second test tool")
        self.assertIn("parameters", second_tool_schema)
