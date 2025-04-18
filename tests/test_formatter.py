"""
Tests for the tools formatter module.

This module provides comprehensive tests for the ToolResponseFormatter class.
"""

import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.cli_code.mcp.tools.formatter import ToolResponseFormatter
from src.cli_code.mcp.tools.models import ToolResult


class CustomObject:
    """A custom object for testing the formatter."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def to_dict(self):
        """Convert the object to a dictionary."""
        return {"name": self.name, "value": self.value}


@dataclass
class CustomObjectWithoutDict:
    """A custom object without a to_dict method."""

    name: str
    value: int


class TestToolResponseFormatter(unittest.TestCase):
    """Test case for the ToolResponseFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ToolResponseFormatter()

    def test_format_result_success(self):
        """Test formatting a successful result."""
        # Create a successful result
        result = ToolResult(
            tool_name="test_tool",
            parameters={"param": "value"},
            success=True,
            result={"output": "test_output"},
            error=None,
        )

        # Format the result
        formatted = self.formatter.format_result(result)

        # Check the formatted result
        self.assertEqual(formatted["name"], "test_tool")
        self.assertEqual(formatted["parameters"], {"param": "value"})
        self.assertEqual(formatted["result"], {"output": "test_output"})

    def test_format_result_error(self):
        """Test formatting a result with an error."""
        # Create a result with an error
        result = ToolResult(
            tool_name="test_tool", parameters={"param": "value"}, success=False, result=None, error="Test error"
        )

        # Format the result
        formatted = self.formatter.format_result(result)

        # Check the formatted result
        self.assertEqual(formatted["name"], "test_tool")
        self.assertEqual(formatted["parameters"], {"param": "value"})
        self.assertEqual(formatted["result"], None)

    def test_format_results_multiple(self):
        """Test formatting multiple results."""
        # Create multiple results
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

        # Format the results
        formatted = self.formatter.format_results(results)

        # Check the formatted results
        self.assertIn("results", formatted)
        self.assertEqual(len(formatted["results"]), 2)

        # Check the first result
        self.assertEqual(formatted["results"][0]["name"], "test_tool1")
        self.assertEqual(formatted["results"][0]["parameters"], {"param1": "value1"})
        self.assertEqual(formatted["results"][0]["result"], {"output1": "test_output1"})

        # Check the second result
        self.assertEqual(formatted["results"][1]["name"], "test_tool2")
        self.assertEqual(formatted["results"][1]["parameters"], {"param2": "value2"})
        self.assertEqual(formatted["results"][1]["result"], None)

    def test_format_results_empty(self):
        """Test formatting an empty list of results."""
        # Format empty results
        formatted = self.formatter.format_results([])

        # Check the formatted results
        self.assertIn("results", formatted)
        self.assertEqual(len(formatted["results"]), 0)

    def test_format_error(self):
        """Test formatting an error."""
        # Create an error
        error = ValueError("Invalid value")
        tool_name = "test_tool"
        parameters = {"param": "value"}

        # Format the error
        formatted = self.formatter.format_error(error, tool_name, parameters)

        # Check the formatted error
        self.assertEqual(formatted["name"], "test_tool")
        self.assertEqual(formatted["parameters"], {"param": "value"})
        self.assertIn("error", formatted)
        self.assertEqual(formatted["error"]["type"], "ValueError")
        self.assertEqual(formatted["error"]["message"], "Invalid value")

    def test_prepare_result_for_json_basic_types(self):
        """Test preparing basic types for JSON serialization."""
        # Test with a string
        self.assertEqual(self.formatter._prepare_result_for_json("test"), "test")

        # Test with an integer
        self.assertEqual(self.formatter._prepare_result_for_json(123), 123)

        # Test with a float
        self.assertEqual(self.formatter._prepare_result_for_json(123.45), 123.45)

        # Test with a boolean
        self.assertEqual(self.formatter._prepare_result_for_json(True), True)

        # Test with None
        self.assertIsNone(self.formatter._prepare_result_for_json(None))

    def test_prepare_result_for_json_with_to_dict(self):
        """Test preparing an object with a to_dict method."""
        # Create an object with a to_dict method
        obj = CustomObject("test", 123)

        # Prepare the object for JSON serialization
        prepared = self.formatter._prepare_result_for_json(obj)

        # Check the prepared object
        self.assertIsInstance(prepared, dict)
        self.assertEqual(prepared["name"], "test")
        self.assertEqual(prepared["value"], 123)

    def test_prepare_result_for_json_with_dict_attr(self):
        """Test preparing an object with a __dict__ attribute."""
        # Create an object without a to_dict method
        obj = CustomObjectWithoutDict("test", 123)

        # Prepare the object for JSON serialization
        prepared = self.formatter._prepare_result_for_json(obj)

        # Check the prepared object
        self.assertIsInstance(prepared, dict)
        self.assertEqual(prepared["name"], "test")
        self.assertEqual(prepared["value"], 123)

    def test_prepare_result_for_json_list(self):
        """Test preparing a list for JSON serialization."""
        # Create a list of mixed objects
        lst = ["string", 123, CustomObject("obj1", 1), CustomObjectWithoutDict("obj2", 2), [1, 2, 3], {"key": "value"}]

        # Prepare the list for JSON serialization
        prepared = self.formatter._prepare_result_for_json(lst)

        # Check the prepared list
        self.assertIsInstance(prepared, list)
        self.assertEqual(len(prepared), 6)
        self.assertEqual(prepared[0], "string")
        self.assertEqual(prepared[1], 123)
        self.assertEqual(prepared[2], {"name": "obj1", "value": 1})
        self.assertEqual(prepared[3], {"name": "obj2", "value": 2})
        self.assertEqual(prepared[4], [1, 2, 3])
        self.assertEqual(prepared[5], {"key": "value"})

    def test_prepare_result_for_json_dict(self):
        """Test preparing a dictionary for JSON serialization."""
        # Create a dictionary with mixed values
        dct = {
            "string": "value",
            "number": 123,
            "object": CustomObject("obj", 1),
            "object_without_dict": CustomObjectWithoutDict("obj2", 2),
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        # Prepare the dictionary for JSON serialization
        prepared = self.formatter._prepare_result_for_json(dct)

        # Check the prepared dictionary
        self.assertIsInstance(prepared, dict)
        self.assertEqual(len(prepared), 6)
        self.assertEqual(prepared["string"], "value")
        self.assertEqual(prepared["number"], 123)
        self.assertEqual(prepared["object"], {"name": "obj", "value": 1})
        self.assertEqual(prepared["object_without_dict"], {"name": "obj2", "value": 2})
        self.assertEqual(prepared["list"], [1, 2, 3])
        self.assertEqual(prepared["dict"], {"nested": "value"})

    def test_prepare_result_for_json_nested(self):
        """Test preparing a nested structure for JSON serialization."""
        # Create a nested structure
        nested = {
            "level1": {
                "level2": {
                    "level3": CustomObject("nested", 3),
                    "list": [CustomObject("item1", 1), CustomObject("item2", 2)],
                }
            }
        }

        # Prepare the nested structure for JSON serialization
        prepared = self.formatter._prepare_result_for_json(nested)

        # Check the prepared structure
        self.assertIsInstance(prepared, dict)
        self.assertIn("level1", prepared)
        self.assertIn("level2", prepared["level1"])
        self.assertIn("level3", prepared["level1"]["level2"])
        self.assertEqual(prepared["level1"]["level2"]["level3"], {"name": "nested", "value": 3})
        self.assertEqual(prepared["level1"]["level2"]["list"][0], {"name": "item1", "value": 1})
        self.assertEqual(prepared["level1"]["level2"]["list"][1], {"name": "item2", "value": 2})


if __name__ == "__main__":
    unittest.main()
