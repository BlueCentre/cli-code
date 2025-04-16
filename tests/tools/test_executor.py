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
from src.cli_code.mcp.tools.registry import ToolRegistry


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
    assert result.tool_name == "test_tool"
    assert result.parameters == {"param1": "value"}
    assert result.result == complex_result
    assert result.success is True
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_with_none_result(executor, mock_tool):
    """Test execution with a None result."""
    # Set up the mock to return None
    mock_tool.execute.return_value = None

    result = await executor.execute("test_tool", {"param1": "value"})

    assert isinstance(result, ToolResult)
    assert result.tool_name == "test_tool"
    assert result.parameters == {"param1": "value"}
    assert result.result is None
    assert result.success is True  # Still successful even with None result
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_with_timeout_simulation(executor, mock_tool):
    """Test execution with timeout simulation."""
    import asyncio

    # Create a slow handler that would time out
    async def slow_handler(params):
        await asyncio.sleep(0.1)  # Sleep briefly to simulate delay
        return {"result": "slow"}

    # Replace the mock handler with our slow handler
    mock_tool.execute = slow_handler

    # Mock asyncio.wait_for to raise TimeoutError
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await executor.execute("test_tool", {"param1": "value"})

        # Should return a failure result with timeout error
        assert result.success is False
        assert result.result is None
        assert "timeout" in result.error.lower()


@patch("logging.Logger")
def test_logging(mock_logger, executor, mock_tool):
    """Test that logging is used appropriately."""
    # Replace the logger in the executor
    with patch("logging.getLogger", return_value=mock_logger):
        # Create a new executor to trigger the getLogger call
        new_executor = ToolExecutor(mock_registry())

        # Test validation logging
        new_executor.validate_parameters("unknown_tool", {})
        mock_logger.error.assert_called()

        # Reset the mock
        mock_logger.reset_mock()

        # Test successful validation logging
        new_executor.validate_parameters(mock_tool, {"param1": "value"})
        # Success doesn't log anything
        mock_logger.error.assert_not_called()


# Test validation with different schema types
def test_validate_parameters_with_array_schema(executor, mock_tool):
    """Test validation with an array schema."""
    mock_tool.schema = {
        "parameters": {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
    }

    # Valid array
    assert executor.validate_parameters(mock_tool, {"items": ["a", "b", "c"]}) is True

    # Invalid array (contains a non-string)
    assert executor.validate_parameters(mock_tool, {"items": ["a", 1, "c"]}) is False

    # Missing required array
    assert executor.validate_parameters(mock_tool, {}) is False


def test_validate_parameters_with_nested_object_schema(executor, mock_tool):
    """Test validation with a nested object schema."""
    mock_tool.schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "address": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                    "required": ["name", "address"],
                }
            },
            "required": ["user"],
        }
    }

    # Valid nested object
    assert (
        executor.validate_parameters(
            mock_tool, {"user": {"name": "John", "age": 30, "address": {"city": "New York", "zip": "10001"}}}
        )
        is True
    )

    # Missing required nested field
    assert (
        executor.validate_parameters(
            mock_tool,
            {
                "user": {
                    "name": "John",
                    "address": {
                        "zip": "10001"
                        # Missing required city
                    },
                }
            },
        )
        is False
    )


@pytest.mark.asyncio
async def test_execute_with_different_parameter_types(executor, mock_tool):
    """Test execution with different parameter types."""
    mock_tool.schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "string_param": {"type": "string"},
                "number_param": {"type": "number"},
                "boolean_param": {"type": "boolean"},
                "null_param": {"type": "null"},
                "array_param": {"type": "array", "items": {"type": "string"}},
                "object_param": {"type": "object"},
            },
            "required": ["string_param"],
        }
    }

    params = {
        "string_param": "string value",
        "number_param": 42.5,
        "boolean_param": True,
        "null_param": None,
        "array_param": ["item1", "item2"],
        "object_param": {"key": "value"},
    }

    await executor.execute("test_tool", params)

    # Verify that the execute method was called with all parameters
    mock_tool.execute.assert_called_once_with(params)


if __name__ == "__main__":
    pytest.main()
