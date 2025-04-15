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

    async def test_add(self):
        """Test addition operation."""
        result = await calculator_handler(operation="add", a=5, b=3)
        self.assertEqual(result, {"result": 8})

    async def test_subtract(self):
        """Test subtraction operation."""
        result = await calculator_handler(operation="subtract", a=5, b=3)
        self.assertEqual(result, {"result": 2})

    async def test_multiply(self):
        """Test multiplication operation."""
        result = await calculator_handler(operation="multiply", a=5, b=3)
        self.assertEqual(result, {"result": 15})

    async def test_divide(self):
        """Test division operation."""
        result = await calculator_handler(operation="divide", a=6, b=3)
        self.assertEqual(result, {"result": 2})

    async def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            await calculator_handler(operation="divide", a=5, b=0)
        self.assertIn("Division by zero", str(context.exception))

    async def test_invalid_operation(self):
        """Test invalid operation raises ValueError."""
        with self.assertRaises(ValueError) as context:
            await calculator_handler(operation="power", a=5, b=2)
        self.assertIn("Unsupported operation", str(context.exception))


class TestCalculatorTool(unittest.TestCase):
    """Tests for the CalculatorTool class."""

    def test_create(self):
        """Test the creation of a calculator tool."""
        tool = CalculatorTool.create()

        # Check that a Tool instance was returned
        self.assertIsInstance(tool, Tool)

        # Check the tool properties
        self.assertEqual(tool.name, "calculator")
        self.assertEqual(tool.description, "Perform basic arithmetic operations")

        # Check the parameters
        parameters = {param.name: param for param in tool.parameters}

        # Check operation parameter
        self.assertIn("operation", parameters)
        self.assertTrue(parameters["operation"].required)
        self.assertEqual(parameters["operation"].schema["type"], "string")
        self.assertEqual(parameters["operation"].schema["enum"], ["add", "subtract", "multiply", "divide"])

        # Check a parameter
        self.assertIn("a", parameters)
        self.assertTrue(parameters["a"].required)
        self.assertEqual(parameters["a"].schema["type"], "number")

        # Check b parameter
        self.assertIn("b", parameters)
        self.assertTrue(parameters["b"].required)
        self.assertEqual(parameters["b"].schema["type"], "number")

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test the execution of the calculator tool."""
        # Create the tool
        tool = CalculatorTool.create()

        # Test addition
        result = await tool.handler(operation="add", a=10, b=5)
        self.assertEqual(result, {"result": 15})

        # Test subtraction
        result = await tool.handler(operation="subtract", a=10, b=5)
        self.assertEqual(result, {"result": 5})

        # Test multiplication
        result = await tool.handler(operation="multiply", a=10, b=5)
        self.assertEqual(result, {"result": 50})

        # Test division
        result = await tool.handler(operation="divide", a=10, b=5)
        self.assertEqual(result, {"result": 2})
