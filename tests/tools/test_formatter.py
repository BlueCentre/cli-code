"""
Tests for the ToolResponseFormatter.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.cli_code.mcp.tools.formatter import ToolResponseFormatter
from src.cli_code.mcp.tools.models import ToolResult


class TestToolResponseFormatter(unittest.TestCase):
    """Tests for the ToolResponseFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ToolResponseFormatter()

    def test_format_success_result(self):
        """Test formatting a successful result."""
        # Create a success result
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            result={"data": "test_data"},
            success=True,
            error=None,
        )

        # Format the result
        formatted = self.formatter.format_result(result)

        # Check the formatted result
        self.assertEqual(formatted["result"], "success")
        self.assertEqual(formatted["tool_name"], "test_tool")
        self.assertEqual(formatted["data"], {"data": "test_data"})
        self.assertEqual(formatted["parameters"], {"param1": "value1"})
        self.assertNotIn("error", formatted)

    def test_format_error_result(self):
        """Test formatting an error result."""
        # Create an error result
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            result=None,
            success=False,
            error="Test error message",
        )

        # Format the result
        formatted = self.formatter.format_result(result)

        # Check the formatted result
        self.assertEqual(formatted["result"], "error")
        self.assertEqual(formatted["tool_name"], "test_tool")
        self.assertEqual(formatted["error"], "Test error message")
        self.assertEqual(formatted["parameters"], {"param1": "value1"})
        self.assertNotIn("data", formatted)

    def test_format_multiple_results(self):
        """Test formatting multiple results."""
        # Create success and error results
        success_result = ToolResult(
            tool_name="success_tool",
            parameters={"param1": "value1"},
            result={"data": "success_data"},
            success=True,
            error=None,
        )

        error_result = ToolResult(
            tool_name="error_tool", parameters={"param2": "value2"}, result=None, success=False, error="Error message"
        )

        # Format multiple results
        formatted = self.formatter.format_results([success_result, error_result])

        # Check the formatted results
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["result"], "success")
        self.assertEqual(formatted[0]["tool_name"], "success_tool")
        self.assertEqual(formatted[0]["data"], {"data": "success_data"})
        self.assertEqual(formatted[0]["parameters"], {"param1": "value1"})
        self.assertNotIn("error", formatted[0])

        self.assertEqual(formatted[1]["result"], "error")
        self.assertEqual(formatted[1]["tool_name"], "error_tool")
        self.assertEqual(formatted[1]["error"], "Error message")
        self.assertEqual(formatted[1]["parameters"], {"param2": "value2"})
        self.assertNotIn("data", formatted[1])

    def test_format_empty_results(self):
        """Test formatting an empty list of results."""
        formatted = self.formatter.format_results([])
        self.assertEqual(formatted, [])

    def test_format_error_without_message(self):
        """Test formatting an error without a message."""
        error_formatted = self.formatter.format_error(None)
        self.assertEqual(error_formatted["result"], "error")
        self.assertEqual(error_formatted["error"], "Unknown error")

    def test_format_error_with_message(self):
        """Test formatting an error with a message."""
        error_formatted = self.formatter.format_error("Custom error message")
        self.assertEqual(error_formatted["result"], "error")
        self.assertEqual(error_formatted["error"], "Custom error message")

    def test_format_result_with_none_result_data(self):
        """Test formatting a result with None data."""
        # Create a success result with None data
        result = ToolResult(
            tool_name="test_tool", parameters={"param1": "value1"}, result=None, success=True, error=None
        )

        # Format the result
        formatted = self.formatter.format_result(result)

        # Check the formatted result
        self.assertEqual(formatted["result"], "success")
        self.assertEqual(formatted["tool_name"], "test_tool")
        self.assertEqual(formatted["data"], None)
        self.assertEqual(formatted["parameters"], {"param1": "value1"})
        self.assertNotIn("error", formatted)

    def test_format_result_with_primitive_result_data(self):
        """Test formatting a result with primitive data types."""
        # Test with string
        string_result = ToolResult(tool_name="string_tool", parameters={}, result="string value", success=True)
        string_formatted = self.formatter.format_result(string_result)
        self.assertEqual(string_formatted["data"], "string value")

        # Test with number
        number_result = ToolResult(tool_name="number_tool", parameters={}, result=42, success=True)
        number_formatted = self.formatter.format_result(number_result)
        self.assertEqual(number_formatted["data"], 42)

        # Test with boolean
        bool_result = ToolResult(tool_name="bool_tool", parameters={}, result=True, success=True)
        bool_formatted = self.formatter.format_result(bool_result)
        self.assertEqual(bool_formatted["data"], True)

    def test_format_result_with_list_result_data(self):
        """Test formatting a result with list data."""
        # Create a success result with list data
        result = ToolResult(tool_name="list_tool", parameters={}, result=["item1", "item2", "item3"], success=True)

        # Format the result
        formatted = self.formatter.format_result(result)

        # Check the formatted result
        self.assertEqual(formatted["result"], "success")
        self.assertEqual(formatted["tool_name"], "list_tool")
        self.assertEqual(formatted["data"], ["item1", "item2", "item3"])
        self.assertEqual(formatted["parameters"], {})
        self.assertNotIn("error", formatted)

    def test_format_result_with_complex_nested_data(self):
        """Test formatting a result with complex nested data."""
        # Create a success result with complex nested data
        complex_data = {
            "user": {
                "name": "John Doe",
                "age": 30,
                "address": {"street": "123 Main St", "city": "Anytown", "zip": "12345"},
                "contacts": [{"type": "email", "value": "john@example.com"}, {"type": "phone", "value": "555-1234"}],
            },
            "status": "active",
            "permissions": ["read", "write"],
        }

        result = ToolResult(tool_name="complex_tool", parameters={"userId": "123"}, result=complex_data, success=True)

        # Format the result
        formatted = self.formatter.format_result(result)

        # Check the formatted result
        self.assertEqual(formatted["result"], "success")
        self.assertEqual(formatted["tool_name"], "complex_tool")
        self.assertEqual(formatted["data"], complex_data)
        self.assertEqual(formatted["parameters"], {"userId": "123"})
        self.assertNotIn("error", formatted)

    def test_format_error_with_exception(self):
        """Test formatting an error from an exception."""
        try:
            # Raise an exception
            raise ValueError("Test exception")
        except Exception as e:
            # Format the exception
            error_formatted = self.formatter.format_error(e)

        self.assertEqual(error_formatted["result"], "error")
        self.assertEqual(error_formatted["error"], "Test exception")

    def test_custom_formatter(self):
        """Test using a custom formatter."""

        # Create a custom formatter function
        def custom_formatter(result):
            return {
                "status": "OK" if result.success else "FAILED",
                "tool": result.tool_name.upper(),
                "params": result.parameters,
                "output": result.result,
                "message": result.error,
            }

        # Create a formatter with the custom formatter
        custom_fmt = ToolResponseFormatter(result_formatter=custom_formatter)

        # Create a result
        result = ToolResult(
            tool_name="test_tool", parameters={"param1": "value1"}, result={"data": "test_data"}, success=True
        )

        # Format the result
        formatted = custom_fmt.format_result(result)

        # Check the custom format
        self.assertEqual(formatted["status"], "OK")
        self.assertEqual(formatted["tool"], "TEST_TOOL")
        self.assertEqual(formatted["params"], {"param1": "value1"})
        self.assertEqual(formatted["output"], {"data": "test_data"})
        self.assertIsNone(formatted["message"])
