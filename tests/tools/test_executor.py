"""
Test the tool executor.

These tests verify the functionality of the ToolExecutor class.
"""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import jsonschema
import pytest

from src.cli_code.mcp.tools.executor import ToolExecutor
from src.cli_code.mcp.tools.models import Tool, ToolParameter, ToolResult


@pytest.fixture
def mock_registry():
    """Create a mock tool registry."""
    registry = MagicMock()
    return registry


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool = MagicMock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "Test tool"
    tool.parameters = [
        ToolParameter(name="param1", description="Test parameter", type="string", required=True),
    ]
    tool.schema = {
        "name": "test_tool",
        "description": "Test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
            "required": ["param1"],
        },
    }
    tool.execute = AsyncMock(return_value={"result": "success"})
    return tool


@pytest.fixture
def executor(mock_registry, mock_tool):
    """Create a tool executor with mocked registry."""
    mock_registry.get_tool.return_value = mock_tool
    return ToolExecutor(mock_registry)


def test_init(mock_registry):
    """Test initializing the executor."""
    executor = ToolExecutor(mock_registry)
    assert executor.registry == mock_registry


def test_validate_parameters_success(executor, mock_tool):
    """Test successful parameter validation."""
    result = executor.validate_parameters(mock_tool, {"param1": "value"})
    assert result is True


def test_validate_parameters_missing_required(executor, mock_tool):
    """Test validation with a missing required parameter."""
    result = executor.validate_parameters(mock_tool, {})
    assert result is False


def test_validate_parameters_wrong_type(executor, mock_tool):
    """Test validation with a parameter of the wrong type."""
    # For jsonschema validation, use a parameter with a different schema
    mock_tool.schema = {
        "name": "test_tool",
        "description": "Test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "number"},
            },
            "required": ["param1"],
        },
    }

    result = executor.validate_parameters(mock_tool, {"param1": "value", "param2": "not_a_number"})
    assert result is False


def test_validate_parameters_unknown_tool(executor, mock_registry):
    """Test validation with unknown tool."""
    mock_registry.get_tool.return_value = None
    result = executor.validate_parameters("unknown_tool", {"param1": "value"})
    assert result is False


@pytest.mark.asyncio
async def test_execute_success(executor, mock_tool):
    """Test successful tool execution."""
    result = await executor.execute("test_tool", {"param1": "value"})

    assert isinstance(result, ToolResult)
    assert result.tool_name == "test_tool"
    assert result.parameters == {"param1": "value"}
    assert result.result == {"result": "success"}
    assert result.success is True
    assert result.error is None

    # Verify the execute method was called with the right parameters
    mock_tool.execute.assert_called_once_with({"param1": "value"})


@pytest.mark.asyncio
async def test_execute_validation_error(executor, mock_tool):
    """Test execution with validation errors."""
    # Missing required parameter
    result = await executor.execute("test_tool", {})

    assert isinstance(result, ToolResult)
    assert result.tool_name == "test_tool"
    assert result.parameters == {}
    assert result.result is None
    assert result.success is False
    assert result.error is not None
    assert "validation failed" in result.error.lower()

    # Execute should not be called
    mock_tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_not_found(executor, mock_registry):
    """Test execution with unknown tool."""
    mock_registry.get_tool.return_value = None
    result = await executor.execute("unknown_tool", {"param1": "value"})

    assert isinstance(result, ToolResult)
    assert result.tool_name == "unknown_tool"
    assert result.parameters == {"param1": "value"}
    assert result.result is None
    assert result.success is False
    assert result.error is not None
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_execute_handler_exception(executor, mock_tool):
    """Test execution when handler raises an exception."""
    # Set up the mock to raise an exception
    mock_tool.execute.side_effect = ValueError("Test error")

    result = await executor.execute("test_tool", {"param1": "value"})

    assert isinstance(result, ToolResult)
    assert result.tool_name == "test_tool"
    assert result.parameters == {"param1": "value"}
    assert result.result is None
    assert result.success is False
    assert result.error is not None
    assert "test error" in result.error.lower()

    # Verify the execute method was called
    mock_tool.execute.assert_called_once_with({"param1": "value"})
