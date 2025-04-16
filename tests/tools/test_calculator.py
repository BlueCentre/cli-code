"""
Tests for the calculator tool.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli_code.mcp.tools.examples.calculator import CalculatorTool, calculator_handler
from src.cli_code.mcp.tools.models import Tool


class TestCalculatorHandler(unittest.TestCase):
    """Tests for the calculator_handler function."""

    @pytest.mark.asyncio
    async def test_addition(self):
        """Test addition operation."""
        params = {"operation": "add", "a": 5, "b": 3}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], 8)
        self.assertEqual(result["operation"], "add")
        self.assertEqual(result["a"], 5)
        self.assertEqual(result["b"], 3)

    @pytest.mark.asyncio
    async def test_subtraction(self):
        """Test subtraction operation."""
        params = {"operation": "subtract", "a": 10, "b": 4}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], 6)
        self.assertEqual(result["operation"], "subtract")
        self.assertEqual(result["a"], 10)
        self.assertEqual(result["b"], 4)

    @pytest.mark.asyncio
    async def test_multiplication(self):
        """Test multiplication operation."""
        params = {"operation": "multiply", "a": 6, "b": 7}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], 42)
        self.assertEqual(result["operation"], "multiply")
        self.assertEqual(result["a"], 6)
        self.assertEqual(result["b"], 7)

    @pytest.mark.asyncio
    async def test_division(self):
        """Test division operation."""
        params = {"operation": "divide", "a": 15, "b": 3}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], 5)
        self.assertEqual(result["operation"], "divide")
        self.assertEqual(result["a"], 15)
        self.assertEqual(result["b"], 3)

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        """Test division by zero raises an error."""
        params = {"operation": "divide", "a": 10, "b": 0}

        with self.assertRaises(ValueError) as context:
            await calculator_handler(params)

        self.assertIn("Division by zero", str(context.exception))

    @pytest.mark.asyncio
    async def test_unsupported_operation(self):
        """Test unsupported operation raises an error."""
        params = {"operation": "modulo", "a": 10, "b": 3}

        with self.assertRaises(ValueError) as context:
            await calculator_handler(params)

        self.assertIn("Unsupported operation", str(context.exception))

    @pytest.mark.asyncio
    async def test_missing_operation(self):
        """Test missing operation raises an error."""
        params = {"a": 10, "b": 5}

        with self.assertRaises(ValueError) as context:
            await calculator_handler(params)

        self.assertIn("Operation parameter is required", str(context.exception))

    @pytest.mark.asyncio
    async def test_missing_operand_a(self):
        """Test missing operand a raises an error."""
        params = {"operation": "add", "b": 5}

        with self.assertRaises(ValueError) as context:
            await calculator_handler(params)

        self.assertIn("Parameter 'a' is required", str(context.exception))

    @pytest.mark.asyncio
    async def test_missing_operand_b(self):
        """Test missing operand b raises an error."""
        params = {"operation": "add", "a": 10}

        with self.assertRaises(ValueError) as context:
            await calculator_handler(params)

        self.assertIn("Parameter 'b' is required", str(context.exception))

    @pytest.mark.asyncio
    async def test_float_operands(self):
        """Test operations with float operands."""
        params = {"operation": "add", "a": 3.5, "b": 2.5}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], 6.0)

        # Test division with floats
        params = {"operation": "divide", "a": 7.5, "b": 2.5}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], 3.0)

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        """Test operations with negative numbers."""
        params = {"operation": "add", "a": -5, "b": 3}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], -2)

        # Test multiplication with negatives
        params = {"operation": "multiply", "a": -4, "b": -3}
        result = await calculator_handler(params)

        self.assertEqual(result["result"], 12)


class TestCalculatorTool(unittest.TestCase):
    """Tests for the CalculatorTool class."""

    def test_create(self):
        """Test the create method returns a properly configured Tool instance."""
        tool = CalculatorTool.create()

        # Verify it's a Tool instance
        self.assertIsInstance(tool, Tool)

        # Verify tool properties
        self.assertEqual(tool.name, "calculator")
        self.assertIn("arithmetic operations", tool.description)

        # Verify parameters
        self.assertEqual(len(tool.parameters), 3)

        # Check operation parameter
        operation_param = next(p for p in tool.parameters if p.name == "operation")
        self.assertEqual(operation_param.type, "string")
        self.assertTrue(operation_param.required)
        self.assertEqual(operation_param.enum, ["add", "subtract", "multiply", "divide"])

        # Check a parameter
        a_param = next(p for p in tool.parameters if p.name == "a")
        self.assertEqual(a_param.type, "number")
        self.assertTrue(a_param.required)

        # Check b parameter
        b_param = next(p for p in tool.parameters if p.name == "b")
        self.assertEqual(b_param.type, "number")
        self.assertTrue(b_param.required)

        # Verify the handler
        self.assertEqual(tool.handler, calculator_handler)

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test executing the tool directly."""
        tool = CalculatorTool.create()

        # Test addition
        result = await tool.execute({"operation": "add", "a": 5, "b": 3})
        self.assertEqual(result["result"], 8)

        # Test division
        result = await tool.execute({"operation": "divide", "a": 10, "b": 2})
        self.assertEqual(result["result"], 5)

        # Test error
        with self.assertRaises(ValueError):
            await tool.execute({"operation": "divide", "a": 10, "b": 0})

    def test_schema_generation(self):
        """Test that the tool generates the correct schema."""
        tool = CalculatorTool.create()
        schema = tool.schema

        # Check schema structure
        self.assertEqual(schema["name"], "calculator")
        self.assertIn("arithmetic operations", schema["description"])

        # Check parameters schema
        params_schema = schema["parameters"]
        self.assertEqual(params_schema["type"], "object")

        # Check properties
        properties = params_schema["properties"]
        self.assertIn("operation", properties)
        self.assertIn("a", properties)
        self.assertIn("b", properties)

        # Check operation property
        operation_schema = properties["operation"]
        self.assertEqual(operation_schema["type"], "string")
        self.assertEqual(operation_schema["enum"], ["add", "subtract", "multiply", "divide"])

        # Check required fields
        required = params_schema["required"]
        self.assertIn("operation", required)
        self.assertIn("a", required)
        self.assertIn("b", required)
