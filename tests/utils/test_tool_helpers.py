"""
Tests for the tool_helpers utility module.
"""

import unittest
from unittest.mock import MagicMock, patch

from cli_code.utils.tool_helpers import execute_tool


class TestToolHelpers(unittest.TestCase):
    """Test cases for the tool_helpers module."""

    @patch("cli_code.tools.get_tool")
    def test_execute_tool_with_args(self, mock_get_tool):
        """Test executing a tool with arguments."""
        # Setup mock
        mock_tool = MagicMock()
        mock_get_tool.return_value = mock_tool
        mock_tool.execute.return_value = "Tool executed successfully"

        # Call the function
        args = {"arg1": "value1", "arg2": "value2"}
        result = execute_tool("test_tool", args)

        # Verify
        mock_get_tool.assert_called_once_with("test_tool")
        mock_tool.execute.assert_called_once_with(**args)
        self.assertEqual(result, "Tool executed successfully")

    @patch("cli_code.tools.get_tool")
    def test_execute_tool_without_args(self, mock_get_tool):
        """Test executing a tool without arguments."""
        # Setup mock
        mock_tool = MagicMock()
        mock_get_tool.return_value = mock_tool
        mock_tool.execute.return_value = "Tool executed successfully"

        # Call the function
        result = execute_tool("test_tool")

        # Verify
        mock_get_tool.assert_called_once_with("test_tool")
        mock_tool.execute.assert_called_once_with()
        self.assertEqual(result, "Tool executed successfully")

    @patch("cli_code.tools.get_tool")
    def test_execute_tool_with_none_args(self, mock_get_tool):
        """Test executing a tool with None for args."""
        # Setup mock
        mock_tool = MagicMock()
        mock_get_tool.return_value = mock_tool
        mock_tool.execute.return_value = "Tool executed successfully"

        # Call the function with explicit None
        result = execute_tool("test_tool", None)

        # Verify
        mock_get_tool.assert_called_once_with("test_tool")
        mock_tool.execute.assert_called_once_with()
        self.assertEqual(result, "Tool executed successfully")

    @patch("cli_code.tools.get_tool")
    def test_execute_tool_not_found(self, mock_get_tool):
        """Test executing a tool that doesn't exist."""
        # Setup mock
        mock_get_tool.return_value = None

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            execute_tool("nonexistent_tool")

        # Check error message
        self.assertIn("Tool 'nonexistent_tool' not found", str(context.exception))
        mock_get_tool.assert_called_once_with("nonexistent_tool")


if __name__ == "__main__":
    unittest.main()
