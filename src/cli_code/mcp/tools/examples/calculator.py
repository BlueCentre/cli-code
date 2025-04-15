"""
Calculator tool for MCP protocol.

This module provides a calculator tool for performing basic arithmetic operations.
"""
from typing import Any, Dict, List, Union

from src.cli_code.mcp.tools.models import Tool, ToolParameter


async def calculator_handler(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a basic arithmetic operation.

    Args:
        parameters: Dictionary containing:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: The first operand
            b: The second operand

    Returns:
        The result of the operation

    Raises:
        ValueError: If the operation is not supported or division by zero
    """
    # Extract parameters
    operation = parameters.get("operation")
    a = parameters.get("a")
    b = parameters.get("b")
    
    # Validate required parameters
    if not operation:
        raise ValueError("Operation parameter is required")
    if a is None:
        raise ValueError("Parameter 'a' is required")
    if b is None:
        raise ValueError("Parameter 'b' is required")
    
    # Perform the calculation
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        result = a / b
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    return {"operation": operation, "a": a, "b": b, "result": result}


class CalculatorTool:
    """Calculator tool for basic arithmetic operations."""

    @staticmethod
    def create() -> Tool:
        """
        Create a calculator tool.

        Returns:
            A Tool instance for the calculator
        """
        return Tool(
            name="calculator",
            description="Performs basic arithmetic operations (add, subtract, multiply, divide)",
            parameters=[
                ToolParameter(
                    name="operation",
                    description="The operation to perform (add, subtract, multiply, divide)",
                    type="string",
                    required=True,
                    enum=["add", "subtract", "multiply", "divide"],
                ),
                ToolParameter(name="a", description="The first operand", type="number", required=True),
                ToolParameter(name="b", description="The second operand", type="number", required=True),
            ],
            handler=calculator_handler,
        )
