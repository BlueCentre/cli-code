"""
Tests for the tools models module.

This module provides comprehensive tests for the Tool and ToolRegistry classes.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.cli_code.mcp.tools.models import Tool, ToolRegistry


class TestTool(unittest.TestCase):
    """Test case for the Tool class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample parameter schema
        self.parameter_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

        # Create a sample function
        self.func = MagicMock(return_value={"result": "Success"})

        # Create a tool for testing
        self.tool = Tool(
            name="test_tool",
            description="A test tool for testing",
            parameter_schema=self.parameter_schema,
            func=self.func,
        )

    def test_init(self):
        """Test initializing a Tool."""
        # Check that the attributes were set correctly
        self.assertEqual(self.tool.name, "test_tool")
        self.assertEqual(self.tool.description, "A test tool for testing")
        self.assertEqual(self.tool.parameter_schema, self.parameter_schema)
        self.assertEqual(self.tool.func, self.func)

    def test_get_definition(self):
        """Test getting the tool definition."""
        # Get the definition
        definition = self.tool.get_definition()

        # Check the definition
        self.assertEqual(definition["name"], "test_tool")
        self.assertEqual(definition["description"], "A test tool for testing")
        self.assertEqual(definition["parameters"], self.parameter_schema)

    def test_execute_sync(self):
        """Test executing a synchronous tool function."""
        # Configure the function to return a synchronous result
        self.func.return_value = {"result": "Sync success"}

        # Execute the tool with parameters
        result = self.tool.execute({"name": "John", "age": 30})

        # Verify the function was called with the correct parameters
        self.func.assert_called_once_with({"name": "John", "age": 30})

        # Check the result
        self.assertEqual(result, {"result": "Sync success"})

    @patch("asyncio.iscoroutinefunction")
    @patch("asyncio.run")
    def test_execute_async(self, mock_asyncio_run, mock_iscoroutinefunction):
        """Test executing an asynchronous tool function."""
        # Configure the mocks for async execution
        mock_iscoroutinefunction.return_value = True
        mock_asyncio_run.return_value = {"result": "Async success"}

        # Execute the tool with parameters
        result = self.tool.execute({"name": "Jane", "age": 25})

        # Verify the mocks were called correctly
        mock_iscoroutinefunction.assert_called_once_with(self.func)
        mock_asyncio_run.assert_called_once()

        # Check the result
        self.assertEqual(result, {"result": "Async success"})

    def test_str_representation(self):
        """Test the string representation of a Tool."""
        # Check the string representation
        self.assertEqual(str(self.tool), "Tool(name='test_tool')")

    def test_repr_representation(self):
        """Test the repr representation of a Tool."""
        # Check the repr representation
        self.assertEqual(repr(self.tool), "Tool(name='test_tool', description='A test tool for testing')")


class TestToolRegistry(unittest.TestCase):
    """Test case for the ToolRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a registry for testing
        self.registry = ToolRegistry()

        # Create a sample tool
        self.tool = MagicMock(spec=Tool)
        self.tool.name = "test_tool"
        self.tool.description = "A test tool for testing"
        self.tool.get_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool for testing",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        }

    def test_init(self):
        """Test initializing a ToolRegistry."""
        # Check that the registry starts empty
        self.assertEqual(len(self.registry.tools), 0)

    def test_add_tool(self):
        """Test adding a tool to the registry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Check that the tool was added
        self.assertEqual(len(self.registry.tools), 1)
        self.assertIn("test_tool", self.registry.tools)
        self.assertEqual(self.registry.tools["test_tool"], self.tool)

    def test_add_duplicate_tool(self):
        """Test adding a tool with a duplicate name."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Create another tool with the same name
        duplicate_tool = MagicMock(spec=Tool)
        duplicate_tool.name = "test_tool"

        # Add the duplicate tool
        with self.assertRaises(ValueError):
            self.registry.add_tool(duplicate_tool)

    def test_get_tool(self):
        """Test getting a tool from the registry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Get the tool
        retrieved_tool = self.registry.get_tool("test_tool")

        # Check the result
        self.assertEqual(retrieved_tool, self.tool)

    def test_get_nonexistent_tool(self):
        """Test getting a nonexistent tool from the registry."""
        # Get a nonexistent tool
        retrieved_tool = self.registry.get_tool("nonexistent_tool")

        # Check the result
        self.assertIsNone(retrieved_tool)

    def test_get_tool_definitions(self):
        """Test getting tool definitions from the registry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Get the tool definitions
        definitions = self.registry.get_tool_definitions()

        # Check the result
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "test_tool")
        self.assertEqual(definitions[0]["description"], "A test tool for testing")
        self.assertIn("parameters", definitions[0])

    def test_get_tool_definitions_empty(self):
        """Test getting tool definitions from an empty registry."""
        # Get the tool definitions
        definitions = self.registry.get_tool_definitions()

        # Check the result
        self.assertEqual(len(definitions), 0)

    def test_remove_tool(self):
        """Test removing a tool from the registry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Verify the tool is in the registry
        self.assertIn("test_tool", self.registry.tools)

        # Remove the tool
        self.registry.remove_tool("test_tool")

        # Check that the tool was removed
        self.assertNotIn("test_tool", self.registry.tools)

    def test_remove_nonexistent_tool(self):
        """Test removing a nonexistent tool from the registry."""
        # Remove a nonexistent tool
        with self.assertRaises(KeyError):
            self.registry.remove_tool("nonexistent_tool")

    def test_contains(self):
        """Test checking if a tool is in the registry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Check that the tool is in the registry
        self.assertTrue("test_tool" in self.registry)
        self.assertFalse("nonexistent_tool" in self.registry)

    def test_len(self):
        """Test getting the number of tools in the registry."""
        # Check the initial length
        self.assertEqual(len(self.registry), 0)

        # Add a tool
        self.registry.add_tool(self.tool)

        # Check the length after adding a tool
        self.assertEqual(len(self.registry), 1)

    def test_iter(self):
        """Test iterating over the registry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Iterate over the registry
        tools = list(self.registry)

        # Check the result
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0], self.tool)

    def test_clear(self):
        """Test clearing the registry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Verify the tool is in the registry
        self.assertIn("test_tool", self.registry.tools)

        # Clear the registry
        self.registry.clear()

        # Check that the registry is empty
        self.assertEqual(len(self.registry.tools), 0)

    def test_str_representation(self):
        """Test the string representation of a ToolRegistry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Check the string representation
        self.assertEqual(str(self.registry), "ToolRegistry(tools=['test_tool'])")

    def test_repr_representation(self):
        """Test the repr representation of a ToolRegistry."""
        # Add a tool
        self.registry.add_tool(self.tool)

        # Check the repr representation
        self.assertEqual(repr(self.registry), "ToolRegistry(tools=['test_tool'])")
