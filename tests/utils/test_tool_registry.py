import unittest
from unittest.mock import MagicMock, patch

from cli_code.utils.tool_registry import ToolRegistry


class TestToolRegistry(unittest.TestCase):
    """Tests for the ToolRegistry class"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = ToolRegistry()

    def test_init(self):
        """Test that the registry initializes with an empty dictionary"""
        self.assertEqual(self.registry.tools, {})

    def test_register_tool(self):
        """Test registering a tool"""
        mock_tool = MagicMock()

        self.registry.register_tool("test_tool", mock_tool)

        self.assertIn("test_tool", self.registry.tools)
        self.assertEqual(self.registry.tools["test_tool"], mock_tool)

    def test_get_tool(self):
        """Test getting a registered tool"""
        mock_tool = MagicMock()

        self.registry.tools["test_tool"] = mock_tool

        result = self.registry.get_tool("test_tool")

        self.assertEqual(result, mock_tool)

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist"""
        result = self.registry.get_tool("nonexistent_tool")

        self.assertIsNone(result)

    def test_get_declarations(self):
        """Test getting declarations for all tools"""
        # Create mock tools with declarations
        mock_tool1 = MagicMock()
        mock_tool1.get_function_declaration.return_value = {"name": "tool1", "description": "Test tool 1"}

        mock_tool2 = MagicMock()
        mock_tool2.get_function_declaration.return_value = {"name": "tool2", "description": "Test tool 2"}

        # Register the tools
        self.registry.tools = {"tool1": mock_tool1, "tool2": mock_tool2}

        declarations = self.registry.get_declarations()

        # Verify the result has both tool declarations
        self.assertEqual(len(declarations), 2)
        self.assertIn({"name": "tool1", "description": "Test tool 1"}, declarations)
        self.assertIn({"name": "tool2", "description": "Test tool 2"}, declarations)

    def test_keys(self):
        """Test the keys method"""
        mock_tool1 = MagicMock()
        mock_tool2 = MagicMock()

        self.registry.tools = {"tool1": mock_tool1, "tool2": mock_tool2}

        keys = self.registry.keys()

        self.assertEqual(set(keys), {"tool1", "tool2"})


if __name__ == "__main__":
    unittest.main()
