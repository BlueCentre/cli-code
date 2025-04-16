"""
Tests for the tool response formatter.
"""

import json
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

from src.cli_code.mcp.tools.formatter import ToolResponseFormatter
from src.cli_code.mcp.tools.models import ToolResult


class TestToolResponseFormatter(unittest.TestCase):
    """Tests for the ToolResponseFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ToolResponseFormatter()

    def test_format_result_basic(self):
        """Test formatting a basic result."""
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param1": "value1", "param2": 42},
            result={"status": "success", "value": 123},
        )

        formatted = self.formatter.format_result(result)

        self.assertEqual(formatted["name"], "test_tool")
        self.assertEqual(formatted["parameters"], {"param1": "value1", "param2": 42})
        self.assertEqual(formatted["result"], {"status": "success", "value": 123})

    def test_format_result_complex(self):
        """Test formatting a result with complex data types."""
        # Create a complex nested structure
        complex_result = {
            "string": "value",
            "number": 42,
            "list": [1, 2, "three", [4, 5]],
            "dict": {"nested": {"deeper": [{"key": "value"}]}},
        }

        result = ToolResult(tool_name="complex_tool", parameters={"operation": "process"}, result=complex_result)

        formatted = self.formatter.format_result(result)

        self.assertEqual(formatted["name"], "complex_tool")
        self.assertEqual(formatted["parameters"], {"operation": "process"})
        self.assertEqual(formatted["result"], complex_result)

    def test_format_results_empty(self):
        """Test formatting an empty list of results."""
        formatted = self.formatter.format_results([])

        self.assertEqual(formatted, {"results": []})

    def test_format_results_multiple(self):
        """Test formatting multiple results."""
        result1 = ToolResult(tool_name="tool1", parameters={"param": "value1"}, result={"status": "success"})

        result2 = ToolResult(tool_name="tool2", parameters={"param": "value2"}, result={"status": "pending"})

        formatted = self.formatter.format_results([result1, result2])

        self.assertEqual(len(formatted["results"]), 2)
        self.assertEqual(formatted["results"][0]["name"], "tool1")
        self.assertEqual(formatted["results"][1]["name"], "tool2")

    def test_format_error(self):
        """Test formatting an error."""
        error = ValueError("Invalid parameter value")

        formatted = self.formatter.format_error(
            error=error, tool_name="test_tool", parameters={"param": "invalid_value"}
        )

        self.assertEqual(formatted["name"], "test_tool")
        self.assertEqual(formatted["parameters"], {"param": "invalid_value"})
        self.assertEqual(formatted["error"]["type"], "ValueError")
        self.assertEqual(formatted["error"]["message"], "Invalid parameter value")

    def test_prepare_result_to_dict_method(self):
        """Test formatting a result with a custom class that has to_dict method."""

        @dataclass
        class CustomResult:
            value: int

            def to_dict(self):
                return {"custom_value": self.value}

        result = ToolResult(tool_name="custom_tool", parameters={}, result=CustomResult(42))

        formatted = self.formatter.format_result(result)

        self.assertEqual(formatted["result"], {"custom_value": 42})

    def test_prepare_result_with_dict_attr(self):
        """Test formatting a result with a custom class that has __dict__."""

        class PlainObject:
            def __init__(self, value):
                self.attribute = value

        result = ToolResult(tool_name="plain_object_tool", parameters={}, result=PlainObject("test_value"))

        formatted = self.formatter.format_result(result)

        self.assertEqual(formatted["result"], {"attribute": "test_value"})

    def test_prepare_result_list(self):
        """Test formatting a result with a list."""
        result = ToolResult(tool_name="list_tool", parameters={}, result=[1, "two", 3.0])

        formatted = self.formatter.format_result(result)

        self.assertEqual(formatted["result"], [1, "two", 3.0])

    def test_prepare_result_nested_objects(self):
        """Test formatting a result with nested objects."""

        # Simple class containing a string attribute
        class SimpleObject:
            def __init__(self, value):
                self.value = value

        # Create the object to test with
        obj = SimpleObject("test value")

        # Create the tool result
        result = ToolResult(tool_name="simple_object_tool", parameters={}, result=obj)

        # Get the formatted result
        formatted = self.formatter.format_result(result)

        # Just check that the result was converted to some kind of dictionary
        # and contains the value attribute
        self.assertIsInstance(formatted["result"], dict)
        self.assertIn("value", formatted["result"])
        self.assertEqual(formatted["result"]["value"], "test value")

    def test_prepare_result_dict_with_objects(self):
        """Test formatting a result with a dictionary containing objects."""

        @dataclass
        class CustomObject:
            value: str

            def to_dict(self):
                return {"custom_value": self.value}

        result = ToolResult(
            tool_name="dict_with_objects_tool", parameters={}, result={"primitive": 123, "object": CustomObject("test")}
        )

        formatted = self.formatter.format_result(result)

        self.assertEqual(formatted["result"]["primitive"], 123)
        self.assertEqual(formatted["result"]["object"], {"custom_value": "test"})

    def test_prepare_result_primitive_types(self):
        """Test formatting primitive result types."""
        # Test with string
        string_result = ToolResult(tool_name="string_tool", parameters={}, result="string_value")
        self.assertEqual(self.formatter.format_result(string_result)["result"], "string_value")

        # Test with number
        number_result = ToolResult(tool_name="number_tool", parameters={}, result=42)
        self.assertEqual(self.formatter.format_result(number_result)["result"], 42)

        # Test with boolean
        bool_result = ToolResult(tool_name="bool_tool", parameters={}, result=True)
        self.assertEqual(self.formatter.format_result(bool_result)["result"], True)

        # Test with None
        none_result = ToolResult(tool_name="none_tool", parameters={}, result=None)
        self.assertEqual(self.formatter.format_result(none_result)["result"], None)
