"""
Calculator tool for MCP protocol.

This module provides a simple calculator tool for demonstration.
"""
from typing import Any, Dict, List

from cli_code.mcp.tools.models import Tool, ToolParameter


async def calculator_handler(operation: str, a: float, b: float) -> Dict[str, Any]:
    """
    Perform a basic arithmetic operation.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: The first operand
        b: The second operand
        
    Returns:
        The result of the operation
    
    Raises:
        ValueError: If the operation is not supported or division by zero
    """
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
    
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result
    }


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
                    enum=["add", "subtract", "multiply", "divide"]
                ),
                ToolParameter(
                    name="a",
                    description="The first operand",
                    type="number",
                    required=True
                ),
                ToolParameter(
                    name="b",
                    description="The second operand",
                    type="number",
                    required=True
                )
            ],
            handler=calculator_handler
        ) 