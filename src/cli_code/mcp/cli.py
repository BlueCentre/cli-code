"""
CLI interface for MCP tools.

This module provides a CLI interface for interacting with MCP tools.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import rich
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt

from cli_code.mcp.client import MCPClient
from cli_code.mcp.integrations import MCPToolIntegration
from cli_code.mcp.tools.examples.calculator import CalculatorTool
from cli_code.mcp.tools.examples.github import GitHubTool
from cli_code.mcp.tools.examples.weather import WeatherTool
from cli_code.mcp.tools.registry import ToolRegistry
from cli_code.mcp.tools.service import ToolService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("cli_code.mcp.cli")


async def run_interactive_session(integration: MCPToolIntegration, console: Console):
    """
    Run an interactive session with the MCP tools.
    
    Args:
        integration: The tool integration to use
        console: The console to use for I/O
    """
    conversation_history = []
    console.print(Panel("MCP Tool Console - Type 'exit' to quit", style="blue"))
    
    while True:
        # Get user input
        user_message = Prompt.ask("\n[bold blue]You[/bold blue]")
        
        if user_message.lower() in ("exit", "quit"):
            break
        
        try:
            # Process the message
            console.print("[bold yellow]Processing...[/bold yellow]")
            assistant_response, conversation_history = await integration.execute_conversation_turn(
                user_message,
                conversation_history
            )
            
            # Print the response
            console.print(f"\n[bold green]Assistant[/bold green]: {assistant_response}")
        except Exception as e:
            logger.exception("Error processing message")
            console.print(f"[bold red]Error[/bold red]: {str(e)}")


async def run_tool_directly(integration: MCPToolIntegration, tool_name: str, parameters: Dict[str, Any], console: Console):
    """
    Run a tool directly without going through the conversation interface.
    
    Args:
        integration: The tool integration to use
        tool_name: The name of the tool to run
        parameters: The parameters to pass to the tool
        console: The console to use for I/O
    """
    try:
        # Create a mock tool call
        tool_call_id = f"direct_call_{tool_name}"
        tool_call = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        # Process the tool call
        console.print(f"[bold yellow]Executing tool: {tool_name}[/bold yellow]")
        result = await integration.tool_service.execute_tool(tool_name, parameters)
        
        # Print the result
        console.print("\n[bold green]Result[/bold green]:")
        console.print(Panel(json.dumps(result, indent=2), style="green"))
    except Exception as e:
        logger.exception(f"Error executing tool: {tool_name}")
        console.print(f"[bold red]Error[/bold red]: {str(e)}")


async def list_available_tools(integration: MCPToolIntegration, console: Console):
    """
    List all available tools.
    
    Args:
        integration: The tool integration to use
        console: The console to use for I/O
    """
    try:
        # Get tool definitions
        tools = integration.get_tool_definitions()
        
        # Print the tools
        console.print("\n[bold green]Available Tools[/bold green]:")
        for tool in tools:
            function = tool.get("function", {})
            name = function.get("name", "Unknown")
            description = function.get("description", "No description")
            parameters = function.get("parameters", {})
            
            console.print(f"\n[bold blue]{name}[/bold blue]: {description}")
            if parameters:
                console.print("[bold]Parameters:[/bold]")
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])
                
                for param_name, param_info in properties.items():
                    is_required = "[bold red]*[/bold red]" if param_name in required else ""
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description")
                    console.print(f"  - {param_name}{is_required} ({param_type}): {param_desc}")
    except Exception as e:
        logger.exception("Error listing tools")
        console.print(f"[bold red]Error[/bold red]: {str(e)}")


def register_tools(registry: ToolRegistry):
    """
    Register all available tools with the registry.
    
    Args:
        registry: The tool registry to register tools with
    """
    # Register the calculator tool
    calculator_tool = CalculatorTool.create()
    registry.register(calculator_tool)
    
    # Register the weather tool
    weather_tool = WeatherTool.create()
    registry.register(weather_tool)
    
    # Register GitHub tools
    github_list_repos_tool = GitHubTool.create_list_repos_tool()
    github_search_repos_tool = GitHubTool.create_search_repos_tool()
    registry.register(github_list_repos_tool)
    registry.register(github_search_repos_tool)
    
    # Register other tools here...


async def main():
    """Entry point for the CLI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP Tool CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Interactive session command
    interactive_parser = subparsers.add_parser("interactive", help="Run an interactive session")
    
    # Run tool command
    run_parser = subparsers.add_parser("run", help="Run a tool directly")
    run_parser.add_argument("tool_name", help="Name of the tool to run")
    run_parser.add_argument("--parameters", type=str, help="JSON string of parameters to pass to the tool")
    run_parser.add_argument("--parameter-file", type=str, help="Path to a JSON file containing parameters to pass to the tool")
    
    # List tools command
    list_parser = subparsers.add_parser("list", help="List available tools")
    
    args = parser.parse_args()
    
    # Set up console
    console = Console()
    
    # Set up tool registry
    registry = ToolRegistry()
    
    # Register tools with the registry
    register_tools(registry)
    
    # Set up tool service
    tool_service = ToolService(registry)
    
    # Set up MCP client
    # Get API key from environment variable
    api_key = os.environ.get("MCP_API_KEY")
    if not api_key:
        logger.warning("No MCP_API_KEY environment variable found. Some features may not work.")
    
    endpoint = os.environ.get("MCP_ENDPOINT", "https://api.example.com/mcp")
    client = MCPClient(
        endpoint=endpoint,
        api_key=api_key,
        model=os.environ.get("MCP_MODEL", "default")
    )
    
    # Set up integration
    integration = MCPToolIntegration(client, tool_service)
    
    # Handle command
    if args.command == "interactive":
        await run_interactive_session(integration, console)
    elif args.command == "run":
        # Parse parameters
        parameters = {}
        if args.parameters:
            try:
                parameters = json.loads(args.parameters)
            except json.JSONDecodeError:
                console.print("[bold red]Error[/bold red]: Invalid JSON in parameters")
                return
        elif args.parameter_file:
            try:
                with open(args.parameter_file, "r") as f:
                    parameters = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                console.print(f"[bold red]Error[/bold red]: {str(e)}")
                return
        
        await run_tool_directly(integration, args.tool_name, parameters, console)
    elif args.command == "list":
        await list_available_tools(integration, console)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 