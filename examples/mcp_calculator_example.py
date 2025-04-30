#!/usr/bin/env python3
"""
Example script demonstrating the MCP calculator tool.

This script shows how to use the MCP protocol with the calculator tool.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from src.cli_code.mcp.client import MCPClient
from src.cli_code.mcp.integrations import MCPToolIntegration
from src.cli_code.mcp.tools.examples.calculator import CalculatorTool
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("mcp_calculator_example")


async def run_calculator_example(operation: str, a: float, b: float):
    """
    Run the calculator example.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: The first operand
        b: The second operand
    """
    console = Console()
    console.print(Panel("MCP Calculator Example", style="blue"))

    # Set up the tool registry
    registry = ToolRegistry()

    # Register the calculator tool
    calculator_tool = CalculatorTool.create()
    registry.register(calculator_tool)

    # Set up the tool service
    tool_service = ToolService(registry)

    # Create a stub client for local execution (no actual API calls)
    client = MCPClient(
        endpoint="http://localhost/mcp",  # Not used in this example
        api_key="dummy-key",  # Not used in this example
        model="default",  # Not used in this example
    )

    # Set up the integration
    integration = MCPToolIntegration(client, tool_service)

    console.print(f"[bold]Executing calculator tool with parameters:[/bold]")
    console.print(f"- Operation: {operation}")
    console.print(f"- a: {a}")
    console.print(f"- b: {b}")
    console.print()

    try:
        # Execute the calculator tool directly
        parameters = {"operation": operation, "a": a, "b": b}

        # Use the tool service for direct execution
        result = await tool_service.execute_tool("calculator", parameters)

        # Print the result
        console.print("[bold green]Result:[/bold green]")
        console.print(Panel(json.dumps(result, indent=2), style="green"))
    except Exception as e:
        logger.exception("Error executing calculator tool")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="MCP Calculator Example")
    parser.add_argument("operation", choices=["add", "subtract", "multiply", "divide"], help="The operation to perform")
    parser.add_argument("a", type=float, help="The first operand")
    parser.add_argument("b", type=float, help="The second operand")

    args = parser.parse_args()

    asyncio.run(run_calculator_example(args.operation, args.a, args.b))


if __name__ == "__main__":
    main()
