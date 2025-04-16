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


@pytest.mark.asyncio
async def test_execute_registry_exception(executor, mock_registry):
    """Test execution when registry raises an exception."""
    # Set up the mock to raise an exception
    mock_registry.get_tool.side_effect = Exception("Registry error")

    result = await executor.execute("test_tool", {"param1": "value"})

    assert isinstance(result, ToolResult)
    assert result.tool_name == "test_tool"
    assert result.parameters == {"param1": "value"}
    assert result.result is None
    assert result.success is False
    assert result.error is not None
    assert "registry error" in result.error.lower()


def test_validate_parameters_with_schema_without_parameters_key(executor, mock_tool):
    """Test validation with a schema that doesn't have a 'parameters' key."""
    # Create a schema without the 'parameters' key
    mock_tool.schema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
        },
        "required": ["param1"],
    }

    # This should still work, as the executor falls back to the schema itself
    result = executor.validate_parameters(mock_tool, {"param1": "value"})
    assert result is True


def test_validate_parameters_with_validation_error_details(executor, mock_tool):
    """Test validation error with more complex schema validation issues."""
    # Create a more complex schema
    mock_tool.schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 18},
                "email": {"type": "string", "format": "email"},
                "preferences": {
                    "type": "object",
                    "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
                },
            },
            "required": ["age", "email"],
        }
    }

    # Test with multiple validation issues
    result = executor.validate_parameters(
        mock_tool,
        {
            "age": 15,  # Below minimum
            "email": "not-an-email",  # Invalid format
            "preferences": {
                "color": "yellow"  # Not in enum
            },
        },
    )
    assert result is False


@pytest.mark.asyncio
async def test_execute_with_complex_result(executor, mock_tool):
    """Test execution with a more complex result structure."""
    # Set up the mock to return a complex result
    complex_result = {
        "data": {
            "items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}],
            "metadata": {"totalCount": 2, "processingTime": "10ms"},
        },
        "status": "success",
    }
    mock_tool.execute.return_value = complex_result

    result = await executor.execute("test_tool", {"param1": "value"})

    assert isinstance(result, ToolResult)
    assert result.result == complex_result
    assert result.success is True


@pytest.mark.asyncio
async def test_execute_with_none_result(executor, mock_tool):
    """Test execution with None as the result."""
    # Set up the mock to return None
    mock_tool.execute.return_value = None

    result = await executor.execute("test_tool", {"param1": "value"})

    assert isinstance(result, ToolResult)
    assert result.result is None
    assert result.success is True


def test_logging():
    """Test that appropriate logging happens during tool execution."""
    # Create a mock Logger
    mock_logger = MagicMock()

    # Create a mock tool and registry
    mock_tool = MagicMock(spec=Tool)
    mock_tool.name = "test_tool"
    mock_tool.schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
            "required": ["param1"],
        },
    }

    mock_registry = MagicMock()
    mock_registry.get_tool.return_value = mock_tool

    # Create executor with our mock registry
    executor = ToolExecutor(mock_registry)

    # Replace the executor's logger with our mock
    executor.logger = mock_logger

    # Test with valid parameters
    executor.validate_parameters(mock_tool, {"param1": "value"})

    # Test with invalid parameters
    executor.validate_parameters(mock_tool, {})

    # Check that warning was called with a message about validation
    mock_logger.error.assert_not_called()
    mock_logger.warning.assert_called()

    # Get the warning args
    warning_args = mock_logger.warning.call_args[0][0]
    assert "validation failed" in warning_args.lower()
