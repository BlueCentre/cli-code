"""
Tests for the ToolExecutor class.
"""

import unittest
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from jsonschema.exceptions import ValidationError

from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.models import Tool, ToolParameter, ToolResult
from src.cli_code.mcp.tools.registry import ToolRegistry


class TestToolExecutor(unittest.TestCase):
    """Tests for the ToolExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock tool
        self.mock_tool = MagicMock(spec=Tool)
        self.mock_tool.name = "test_tool"
        self.mock_tool.parameters = [
            ToolParameter(
                name="param1",
                description="A required parameter",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="param2",
                description="An optional parameter",
                type="number",
                required=False,
            ),
        ]
        
        # Set up the schema for validation
        self.mock_tool.schema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "A required parameter"},
                "param2": {"type": "number", "description": "An optional parameter"},
            },
            "required": ["param1"],
        }

        # Create a mock registry
        self.mock_registry = MagicMock(spec=ToolRegistry)
        self.mock_registry.get_tool.return_value = self.mock_tool
        
        # Create an executor
        self.executor = ToolExecutor(self.mock_registry)
    
    def test_init(self):
        """Test initialization of the executor."""
        self.assertEqual(self.executor.registry, self.mock_registry)
    
    def test_validate_parameters_success(self):
        """Test successful parameter validation."""
        # Test with a required parameter
        result = self.executor.validate_parameters("test_tool", {"param1": "value"})
        self.assertTrue(result)
        
        # Test with both required and optional parameters
        result = self.executor.validate_parameters("test_tool", {"param1": "value", "param2": 42})
        self.assertTrue(result)
    
    def test_validate_parameters_missing_required(self):
        """Test validation with missing required parameter."""
        result = self.executor.validate_parameters("test_tool", {})
        self.assertFalse(result)
        
        result = self.executor.validate_parameters("test_tool", {"param2": 42})
        self.assertFalse(result)
    
    def test_validate_parameters_wrong_type(self):
        """Test validation with incorrect parameter type."""
        result = self.executor.validate_parameters("test_tool", {"param1": 123})  # Should be string
        self.assertFalse(result)
        
        result = self.executor.validate_parameters("test_tool", {"param1": "value", "param2": "not_a_number"})
        self.assertFalse(result)
    
    def test_validate_parameters_unknown_tool(self):
        """Test validation with unknown tool."""
        self.mock_registry.get_tool.return_value = None
        result = self.executor.validate_parameters("unknown_tool", {"param1": "value"})
        self.assertFalse(result)
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        result = await self.executor.execute("test_tool", {"param1": "value"})
        
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value"})
        self.assertEqual(result.result, {"result": "success"})
        self.assertIsNone(result.error)
        
        # Verify the handler was called with the right parameters
        self.mock_tool.handler.assert_called_once_with(param1="value")
    
    @pytest.mark.asyncio
    async def test_execute_validation_error(self):
        """Test execution with validation errors."""
        # Missing required parameter
        result = await self.executor.execute("test_tool", {})
        
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {})
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("validation failed", result.error.lower())
        
        # Handler should not be called
        self.mock_tool.handler.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test execution with unknown tool."""
        self.mock_registry.get_tool.return_value = None
        result = await self.executor.execute("unknown_tool", {"param1": "value"})
        
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "unknown_tool")
        self.assertEqual(result.parameters, {"param1": "value"})
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error.lower())
    
    @pytest.mark.asyncio
    async def test_execute_handler_exception(self):
        """Test execution when handler raises an exception."""
        self.mock_tool.handler.side_effect = ValueError("Test error")
        
        result = await self.executor.execute("test_tool", {"param1": "value"})
        
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.parameters, {"param1": "value"})
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("test error", result.error.lower())
        
        # Verify the handler was called
        self.mock_tool.handler.assert_called_once_with(param1="value") 