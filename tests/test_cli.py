"""
Tests for the MCP CLI module.
"""

import argparse
import asyncio
import json
import os
import unittest
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from rich.console import Console

# Adjust import path based on your project structure
from src.cli_code.mcp import cli
from src.cli_code.mcp.client import MCPClient
from src.cli_code.mcp.integrations import MCPToolIntegration
from src.cli_code.mcp.tools.examples.calculator import CalculatorTool
from src.cli_code.mcp.tools.examples.github import GitHubTool
from src.cli_code.mcp.tools.examples.weather import WeatherTool
from src.cli_code.mcp.tools.models import ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService


@pytest.mark.asyncio
@patch("cli_code.mcp.cli.AVAILABLE_TOOLS", new_callable=MagicMock)
class TestMCPCLI(unittest.TestCase):
    """Tests for the MCP CLI functions."""

    def setUp(self):
        """Set up mocks for common dependencies."""
        self.mock_console = MagicMock(spec=Console)
        self.mock_registry = MagicMock(spec=ToolRegistry)
        self.mock_tool_service = MagicMock(spec=ToolService)
        self.mock_client = MagicMock(spec=MCPClient)
        self.mock_integration = MagicMock(spec=MCPToolIntegration)
        self.mock_integration.tool_service = self.mock_tool_service  # Link mock service

        # Patch dependencies used in the CLI module setup and functions
        self.patches = [
            patch("src.cli_code.mcp.cli.Console", return_value=self.mock_console),
            patch("src.cli_code.mcp.cli.ToolRegistry", return_value=self.mock_registry),
            patch("src.cli_code.mcp.cli.ToolService", return_value=self.mock_tool_service),
            patch("src.cli_code.mcp.cli.MCPClient", return_value=self.mock_client),
            patch("src.cli_code.mcp.cli.MCPToolIntegration", return_value=self.mock_integration),
            patch("src.cli_code.mcp.cli.register_tools"),  # Mock tool registration
            patch("src.cli_code.mcp.cli.run_interactive_session", new_callable=AsyncMock),
            patch("src.cli_code.mcp.cli.run_tool_directly", new_callable=AsyncMock),
            patch("src.cli_code.mcp.cli.list_available_tools", new_callable=AsyncMock),
            patch.dict(os.environ, {"MCP_API_KEY": "test-key"}, clear=True),  # Ensure API key is set for tests
            patch("src.cli_code.mcp.cli.logger"),  # Mock logger to check warnings
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

        self.mock_logger = cli.logger  # Get the mocked logger

    # --- Test list_available_tools ---

    async def test_list_available_tools_success(self):
        """Test successfully listing available tools."""
        # Unpatch list_available_tools for this specific test
        for p in self.patches:
            if p.target == cli.list_available_tools:
                p.stop()
                break

        mock_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "Desc 1",
                    "parameters": {"properties": {"p1": {"type": "string"}}, "required": ["p1"]},
                },
            },
            {"type": "function", "function": {"name": "tool2", "description": "Desc 2", "parameters": {}}},
        ]
        self.mock_integration.get_tool_definitions.return_value = mock_tools

        await cli.list_available_tools(self.mock_integration, self.mock_console)

        self.mock_integration.get_tool_definitions.assert_called_once()
        self.mock_console.print.assert_has_calls(
            [
                call("\n[bold green]Available Tools[/bold green]:"),
                call("\n[bold blue]tool1[/bold blue]: Desc 1"),
                call("[bold]Parameters:[/bold]"),
                call(
                    "  - p1[bold red]*[/bold red] (string): No description"
                ),  # Assuming default description if not present
                call("\n[bold blue]tool2[/bold blue]: Desc 2"),
                # No parameters printed for tool2
            ]
        )
        self.mock_logger.exception.assert_not_called()

        # Re-patch list_available_tools
        for p in self.patches:
            if p.target == cli.list_available_tools:
                p.start()
                break

    async def test_list_available_tools_error(self):
        """Test error handling when listing tools."""
        # Unpatch list_available_tools for this specific test
        for p in self.patches:
            if p.target == cli.list_available_tools:
                p.stop()
                break

        error_message = "Failed to fetch tools"
        self.mock_integration.get_tool_definitions.side_effect = Exception(error_message)

        await cli.list_available_tools(self.mock_integration, self.mock_console)

        self.mock_integration.get_tool_definitions.assert_called_once()
        self.mock_logger.exception.assert_called_once_with("Error listing tools")
        self.mock_console.print.assert_called_with(f"[bold red]Error[/bold red]: {error_message}")

        # Re-patch list_available_tools
        for p in self.patches:
            if p.target == cli.list_available_tools:
                p.start()
                break

    # --- Test run_tool_directly ---

    async def test_run_tool_directly_success(self):
        """Test successfully running a tool directly."""
        # Unpatch run_tool_directly for this specific test
        for p in self.patches:
            if p.target == cli.run_tool_directly:
                p.stop()
                break

        tool_name = "test_tool"
        params = {"arg1": "value1"}
        mock_result = ToolResult(tool_name=tool_name, parameters=params, result={"output": "success"}, success=True)

        # Mock the tool_service on the actual integration object used
        self.mock_integration.tool_service.execute_tool = AsyncMock(return_value=mock_result)

        await cli.run_tool_directly(self.mock_integration, tool_name, params, self.mock_console)

        self.mock_integration.tool_service.execute_tool.assert_awaited_once_with(tool_name, params)
        self.mock_console.print.assert_has_calls(
            [
                call(f"[bold yellow]Executing tool: {tool_name}[/bold yellow]"),
                call("\n[bold green]Result[/bold green]:"),
                call(ANY),  # Check the Panel content separately if needed
            ]
        )
        # Check panel content
        panel_call = self.mock_console.print.call_args_list[2]
        panel_arg = panel_call.args[0]
        self.assertEqual(panel_arg.renderable, json.dumps(mock_result, indent=2))
        self.mock_logger.exception.assert_not_called()

        # Re-patch run_tool_directly
        for p in self.patches:
            if p.target == cli.run_tool_directly:
                p.start()
                break

    async def test_run_tool_directly_error(self):
        """Test error handling when running a tool directly."""
        # Unpatch run_tool_directly for this specific test
        for p in self.patches:
            if p.target == cli.run_tool_directly:
                p.stop()
                break

        tool_name = "error_tool"
        params = {}
        error_message = "Tool execution failed"
        self.mock_integration.tool_service.execute_tool = AsyncMock(side_effect=Exception(error_message))

        await cli.run_tool_directly(self.mock_integration, tool_name, params, self.mock_console)

        self.mock_integration.tool_service.execute_tool.assert_awaited_once_with(tool_name, params)
        self.mock_logger.exception.assert_called_once_with(f"Error executing tool: {tool_name}")
        self.mock_console.print.assert_has_calls(
            [
                call(f"[bold yellow]Executing tool: {tool_name}[/bold yellow]"),  # Still prints executing message
                call(f"[bold red]Error[/bold red]: {error_message}"),
            ]
        )

        # Re-patch run_tool_directly
        for p in self.patches:
            if p.target == cli.run_tool_directly:
                p.start()
                break

    # --- Test main / Argument Parsing ---

    @patch("sys.argv", ["cli_script.py", "list"])
    async def test_main_list_command(self):
        """Test the main function with the 'list' command."""
        await cli.main()
        cli.list_available_tools.assert_awaited_once_with(ANY, ANY)  # Check it was called
        cli.run_tool_directly.assert_not_awaited()
        cli.run_interactive_session.assert_not_awaited()
        cli.register_tools.assert_called_once()  # Ensure tools are registered

    @patch("sys.argv", ["cli_script.py", "run", "my_tool", "--parameters", '{"key": "val"}'])
    async def test_main_run_command_with_params(self):
        """Test the main function with the 'run' command and JSON params."""
        await cli.main()
        cli.run_tool_directly.assert_awaited_once_with(ANY, "my_tool", {"key": "val"}, ANY)
        cli.list_available_tools.assert_not_awaited()
        cli.run_interactive_session.assert_not_awaited()
        cli.register_tools.assert_called_once()

    @patch("sys.argv", ["cli_script.py", "run", "my_tool", "--parameter-file", "params.json"])
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"file_key": "file_val"}')
    async def test_main_run_command_with_file(self, mock_open):
        """Test the main function with the 'run' command and a parameter file."""
        await cli.main()
        mock_open.assert_called_once_with("params.json", "r")
        cli.run_tool_directly.assert_awaited_once_with(ANY, "my_tool", {"file_key": "file_val"}, ANY)
        cli.list_available_tools.assert_not_awaited()
        cli.run_interactive_session.assert_not_awaited()

    @patch("sys.argv", ["cli_script.py", "run", "my_tool", "--parameters", "{invalid json"])
    async def test_main_run_command_invalid_json_params(self):
        """Test 'run' command with invalid inline JSON parameters."""
        await cli.main()
        cli.run_tool_directly.assert_not_awaited()  # Should not be called
        self.mock_console.print.assert_called_with("[bold red]Error[/bold red]: Invalid JSON in parameters")

    @patch("sys.argv", ["cli_script.py", "run", "my_tool", "--parameter-file", "nonexistent.json"])
    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    async def test_main_run_command_file_not_found(self, mock_open):
        """Test 'run' command with a non-existent parameter file."""
        await cli.main()
        mock_open.assert_called_once_with("nonexistent.json", "r")
        cli.run_tool_directly.assert_not_awaited()
        self.mock_console.print.assert_called_with("[bold red]Error[/bold red]: File not found")

    @patch("sys.argv", ["cli_script.py", "interactive"])
    async def test_main_interactive_command(self):
        """Test the main function with the 'interactive' command."""
        await cli.main()
        cli.run_interactive_session.assert_awaited_once_with(ANY, ANY)
        cli.list_available_tools.assert_not_awaited()
        cli.run_tool_directly.assert_not_awaited()

    @patch("sys.argv", ["cli_script.py"])  # No command
    @patch("argparse.ArgumentParser.print_help")  # Mock print_help
    async def test_main_no_command(self, mock_print_help):
        """Test the main function with no command (should print help)."""
        await cli.main()
        mock_print_help.assert_called_once()
        cli.run_interactive_session.assert_not_awaited()
        cli.list_available_tools.assert_not_awaited()
        cli.run_tool_directly.assert_not_awaited()

    @patch.dict(os.environ, {}, clear=True)  # Clear environment variables
    @patch("sys.argv", ["cli_script.py", "list"])
    async def test_main_no_api_key_warning(self):
        """Test that a warning is logged if MCP_API_KEY is missing."""
        # We need to ensure the logger patch from setUp is active
        mock_logger_instance = self.mock_logger

        await cli.main()

        # Check if the warning was logged
        mock_logger_instance.warning.assert_called_with(
            "No MCP_API_KEY environment variable found. Some features may not work."
        )
        # Check that list was still called despite the warning
        cli.list_available_tools.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
