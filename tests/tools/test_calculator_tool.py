"""
Tests for the Calculator Tool.
"""

import unittest
from unittest.mock import AsyncMock, patch

import pytest

from src.cli_code.mcp.tools.examples.calculator import CalculatorTool, calculator_handler
from src.cli_code.mcp.tools.models import Tool


class TestCalculatorHandler(unittest.TestCase):
    """Tests for the calculator_handler function."""

    @pytest.mark.asyncio
    async def test_add(self):
        """Test addition operation."""
        result = await calculator_handler({"operation": "add", "a": 5, "b": 3})
        self.assertEqual(result, {"operation": "add", "a": 5, "b": 3, "result": 8})

    @pytest.mark.asyncio
    async def test_subtract(self):
        """Test subtraction operation."""
        result = await calculator_handler({"operation": "subtract", "a": 5, "b": 3})
        self.assertEqual(result, {"operation": "subtract", "a": 5, "b": 3, "result": 2})

    @pytest.mark.asyncio
    async def test_multiply(self):
        """Test multiplication operation."""
        result = await calculator_handler({"operation": "multiply", "a": 5, "b": 3})
        self.assertEqual(result, {"operation": "multiply", "a": 5, "b": 3, "result": 15})

    @pytest.mark.asyncio
    async def test_divide(self):
        """Test division operation."""
        result = await calculator_handler({"operation": "divide", "a": 6, "b": 3})
        self.assertEqual(result, {"operation": "divide", "a": 6, "b": 3, "result": 2})

    @pytest.mark.asyncio
    async def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            await calculator_handler({"operation": "divide", "a": 5, "b": 0})
        self.assertIn("Division by zero", str(context.exception))

    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test invalid operation raises ValueError."""
        with self.assertRaises(ValueError) as context:
            await calculator_handler({"operation": "power", "a": 5, "b": 2})
        self.assertIn("Unsupported operation", str(context.exception))
        
    @pytest.mark.asyncio
    async def test_missing_operation(self):
        """Test missing operation parameter raises ValueError."""
        with self.assertRaises(ValueError) as context:
            await calculator_handler({"a": 5, "b": 3})
        self.assertIn("Operation parameter is required", str(context.exception))
        
    @pytest.mark.asyncio
    async def test_missing_a_parameter(self):
        """Test missing 'a' parameter raises ValueError."""
        with self.assertRaises(ValueError) as context:
            await calculator_handler({"operation": "add", "b": 3})
        self.assertIn("Parameter 'a' is required", str(context.exception))
        
    @pytest.mark.asyncio
    async def test_missing_b_parameter(self):
        """Test missing 'b' parameter raises ValueError."""
        with self.assertRaises(ValueError) as context:
            await calculator_handler({"operation": "add", "a": 5})
        self.assertIn("Parameter 'b' is required", str(context.exception))


class TestCalculatorTool(unittest.TestCase):
    """Tests for the CalculatorTool class."""

    def test_create(self):
        """Test the creation of a calculator tool."""
        tool = CalculatorTool.create()

        # Check that a Tool instance was returned
        self.assertIsInstance(tool, Tool)

        # Check the tool properties
        self.assertEqual(tool.name, "calculator")
        self.assertEqual(tool.description, "Performs basic arithmetic operations (add, subtract, multiply, divide)")

        # Check the parameters
        parameters = {param.name: param for param in tool.parameters}

        # Check operation parameter
        self.assertIn("operation", parameters)
        self.assertTrue(parameters["operation"].required)
        operation_schema = parameters["operation"].to_schema()
        self.assertEqual(operation_schema["type"], "string")
        self.assertEqual(operation_schema["enum"], ["add", "subtract", "multiply", "divide"])

        # Check a parameter
        self.assertIn("a", parameters)
        self.assertTrue(parameters["a"].required)
        a_schema = parameters["a"].to_schema()
        self.assertEqual(a_schema["type"], "number")

        # Check b parameter
        self.assertIn("b", parameters)
        self.assertTrue(parameters["b"].required)
        b_schema = parameters["b"].to_schema()
        self.assertEqual(b_schema["type"], "number")

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test the execution of the calculator tool."""
        # Create the tool
        tool = CalculatorTool.create()

        # Test addition
        result = await tool.execute({"operation": "add", "a": 10, "b": 5})
        self.assertEqual(result, {"operation": "add", "a": 10, "b": 5, "result": 15})

        # Test subtraction
        result = await tool.execute({"operation": "subtract", "a": 10, "b": 5})
        self.assertEqual(result, {"operation": "subtract", "a": 10, "b": 5, "result": 5})

        # Test multiplication
        result = await tool.execute({"operation": "multiply", "a": 10, "b": 5})
        self.assertEqual(result, {"operation": "multiply", "a": 10, "b": 5, "result": 50})

        # Test division
        result = await tool.execute({"operation": "divide", "a": 10, "b": 5})
        self.assertEqual(result, {"operation": "divide", "a": 10, "b": 5, "result": 2})
