"""
Tests for the tools executor module.

This module provides comprehensive tests for the ToolExecutor class.
"""

import asyncio
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

import jsonschema

from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.models import Tool
from src.cli_code.mcp.tools.registry import ToolRegistry


class TestToolExecutor(unittest.TestCase):
    """Test case for the ToolExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock registry
        self.registry = MagicMock(spec=ToolRegistry)

        # Create a logger for testing
        self.logger = MagicMock(spec=logging.Logger)

        # Create an executor for testing
        self.executor = ToolExecutor(self.registry)

        # Patch the logger in the executor module
        self.logger_patcher = patch("src.cli_code.mcp.tools.executor.logger", self.logger)
        self.logger_patcher.start()

        # Create a mock tool
        self.tool = MagicMock(spec=Tool)
        self.tool.name = "test_tool"
        self.tool.description = "A test tool for testing"
        self.tool.schema = {
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            }
        }

        # Configure the registry to return the mock tool
        self.registry.get_tool.return_value = self.tool

    def tearDown(self):
        """Clean up after tests."""
        # Stop the logger patcher
        self.logger_patcher.stop()

    def test_init(self):
        """Test initializing a ToolExecutor."""
        # Check that the attributes were set correctly
        self.assertEqual(self.executor.registry, self.registry)

    @patch("logging.getLogger")
    def test_init_default_logger(self, mock_get_logger):
        """Test initializing a ToolExecutor with a default logger."""
        # Create an executor (which will use the default logger)
        executor = ToolExecutor(self.registry)

        # Check that the attributes were set correctly
        self.assertEqual(executor.registry, self.registry)

    def test_validate_parameters_valid(self):
        """Test validating parameters that are valid."""
        # Define valid parameters
        params = {"name": "John", "age": 30}

        # Validate the parameters
        result = self.executor.validate_parameters("test_tool", params)

        # Check the result
        self.assertTrue(result)
        self.registry.get_tool.assert_called_once_with("test_tool")

    def test_validate_parameters_missing_required(self):
        """Test validating parameters that are missing a required field."""
        # Define invalid parameters (missing required 'name' field)
        params = {"age": 30}

        # Validate the parameters
        result = self.executor.validate_parameters("test_tool", params)

        # Check the result
        self.assertFalse(result)
        self.registry.get_tool.assert_called_once_with("test_tool")
        self.logger.error.assert_called_once()  # Some error was logged

    def test_validate_parameters_wrong_type(self):
        """Test validating parameters that have the wrong type."""
        # Define invalid parameters (wrong type for 'age')
        params = {"name": "John", "age": "thirty"}

        # Validate the parameters
        result = self.executor.validate_parameters("test_tool", params)

        # Check the result
        self.assertFalse(result)
        self.registry.get_tool.assert_called_once_with("test_tool")
        self.logger.error.assert_called_once()  # Some error was logged

    def test_validate_parameters_unknown_tool(self):
        """Test validating parameters for an unknown tool."""
        # Configure the registry to return None for an unknown tool
        self.registry.get_tool.return_value = None

        # Define parameters
        params = {"name": "John"}

        # Validate the parameters
        result = self.executor.validate_parameters("unknown_tool", params)

        # Check the result
        self.assertFalse(result)
        self.registry.get_tool.assert_called_once_with("unknown_tool")
        self.logger.error.assert_called_once_with("Tool 'unknown_tool' not found")

    def test_validate_parameters_with_tool_object(self):
        """Test validating parameters by passing the Tool object directly."""
        # Define valid parameters
        params = {"name": "John", "age": 30}

        # Validate the parameters using the tool object
        result = self.executor.validate_parameters(self.tool, params)

        # Check the result
        self.assertTrue(result)
        # registry.get_tool should NOT be called in this case
        self.registry.get_tool.assert_not_called()

    @patch("asyncio.run")
    def test_execute_success(self, mock_asyncio_run):
        """Test executing a tool successfully."""

        # Configure the tool to execute successfully
        async def mock_execute(params):
            return {"result": "Success"}

        self.tool.execute = mock_execute

        # Define parameters
        params = {"name": "John", "age": 30}

        # Execute the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.executor.execute("test_tool", params))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, params)
        self.assertEqual(result.result, {"result": "Success"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

        # Verify the interactions
        self.registry.get_tool.assert_called_once_with("test_tool")

        # Verify logging
        self.logger.info.assert_any_call(f"Attempting to execute tool: test_tool")
        self.logger.info.assert_any_call(f"Executing tool: test_tool")
        self.logger.info.assert_any_call(f"Tool execution completed: test_tool")

    @patch("asyncio.run")
    def test_execute_unknown_tool(self, mock_asyncio_run):
        """Test executing an unknown tool."""
        # Configure the registry to return None for an unknown tool
        self.registry.get_tool.return_value = None

        # Define parameters
        params = {"name": "John"}

        # Execute the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.executor.execute("unknown_tool", params))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(result.tool_name, "unknown_tool")
        self.assertEqual(result.parameters, params)
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool not found: unknown_tool")

        # Check that the registry was called
        self.registry.get_tool.assert_called_once_with("unknown_tool")

        # Verify logging
        self.logger.warning.assert_called_once_with(f"Tool not found: unknown_tool")

    @patch("asyncio.run")
    def test_execute_validation_error(self, mock_asyncio_run):
        """Test executing a tool with invalid parameters."""
        # Define invalid parameters (missing required 'name' field)
        params = {"age": 30}

        # Execute the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.executor.execute("test_tool", params))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, params)
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Parameter validation failed")

        # Check that the registry was called but the tool was not executed
        self.registry.get_tool.assert_called_once_with("test_tool")
        self.tool.execute.assert_not_called()

        # Verify logging
        self.logger.info.assert_called_with(f"Attempting to execute tool: test_tool")
        self.logger.warning.assert_called_with(f"Parameter validation failed for tool: test_tool")

    @patch("asyncio.run")
    def test_execute_tool_error(self, mock_asyncio_run):
        """Test executing a tool that raises an error."""
        # Configure the tool to raise an exception
        error_message = "Tool execution failed"

        async def mock_execute_error(params):
            raise Exception(error_message)

        self.tool.execute = mock_execute_error

        # Define parameters
        params = {"name": "John", "age": 30}

        # Execute the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.executor.execute("test_tool", params))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, params)
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, error_message)

        # Verify the interactions
        self.registry.get_tool.assert_called_once_with("test_tool")

        # Verify logging
        self.logger.info.assert_any_call(f"Attempting to execute tool: test_tool")
        self.logger.error.assert_called_once_with(f"Error executing tool test_tool: {error_message}")

    @patch("asyncio.run")
    def test_execute_registry_error(self, mock_asyncio_run):
        """Test executing when the registry raises an error finding the tool."""
        # Configure the registry to raise an error
        error_message = "Registry internal error"
        self.registry.get_tool.side_effect = ValueError(error_message)

        # Define parameters
        params = {"name": "John"}

        # Execute the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.executor.execute("error_tool", params))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        # Check the result
        self.assertEqual(result.tool_name, "error_tool")
        self.assertEqual(result.parameters, params)
        self.assertIsNone(result.result)
        self.assertFalse(result.success)
        self.assertEqual(result.error, error_message)  # Error should be propagated

        # Check that the registry was called
        self.registry.get_tool.assert_called_once_with("error_tool")

        # Verify logging
        self.logger.error.assert_called_once_with(f"Error executing tool error_tool: {error_message}")


if __name__ == "__main__":
    unittest.main()
