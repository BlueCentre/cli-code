import pytest

from mcp_code.mcp_client.messages.tools.tool_result import ToolResult

# Test cases for ToolResult


def test_tool_result_instantiation_success():
    """Test successful instantiation with required fields."""
    content = [{"type": "text", "text": "Operation successful."}]
    result = ToolResult(content=content, isError=False)
    assert result.content == content
    assert not result.isError
    assert result.model_dump() == {"content": content, "isError": False}


def test_tool_result_instantiation_error():
    """Test successful instantiation indicating an error."""
    content = [{"type": "error", "message": "Failed to execute"}]
    result = ToolResult(content=content, isError=True)
    assert result.content == content
    assert result.isError
    assert result.model_dump() == {"content": content, "isError": True}


def test_tool_result_default_is_error():
    """Test that isError defaults to False."""
    content = [{"status": "ok"}]
    result = ToolResult(content=content)
    assert result.content == content
    assert not result.isError  # Default should be False
    assert result.model_dump() == {"content": content, "isError": False}


def test_tool_result_empty_content():
    """Test instantiation with empty content list."""
    content = []
    result = ToolResult(content=content)
    assert result.content == content
    assert not result.isError
    assert result.model_dump() == {"content": [], "isError": False}


def test_tool_result_complex_content():
    """Test instantiation with more complex content."""
    content = [{"type": "text", "text": "Result value: 42"}, {"type": "json", "data": {"a": 1, "b": [True, None]}}]
    result = ToolResult(content=content, isError=False)
    assert result.content == content
    assert not result.isError
    assert result.model_dump() == {"content": content, "isError": False}
