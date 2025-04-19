import pytest

from mcp_code.mcp_client.messages.tools.tool import Tool
from mcp_code.mcp_client.messages.tools.tool_input_schema import ToolInputSchema

# Test cases for Tool


def test_tool_instantiation():
    """Test successful instantiation of the Tool model."""
    input_schema_dict = {
        "type": "object",
        "properties": {"param1": {"type": "string", "description": "First parameter"}, "param2": {"type": "integer"}},
        "required": ["param1"],
    }
    input_schema = ToolInputSchema(**input_schema_dict)

    tool = Tool(name="my_tool", description="A test tool.", inputSchema=input_schema)

    assert tool.name == "my_tool"
    assert tool.description == "A test tool."
    assert tool.inputSchema == input_schema
    assert tool.inputSchema.type == "object"
    assert "param1" in tool.inputSchema.properties

    # Test model dump
    expected_dump = {
        "name": "my_tool",
        "description": "A test tool.",
        "inputSchema": input_schema.model_dump(),  # Use model_dump of the nested schema
    }
    assert tool.model_dump() == expected_dump


def test_tool_instantiation_minimal_schema():
    """Test instantiation with a minimal input schema."""
    input_schema_dict = {"type": "object", "properties": {}}
    input_schema = ToolInputSchema(**input_schema_dict)

    tool = Tool(name="simple_tool", description="Another tool.", inputSchema=input_schema)

    assert tool.name == "simple_tool"
    assert tool.inputSchema.type == "object"
    assert tool.inputSchema.properties == {}

    expected_dump = {"name": "simple_tool", "description": "Another tool.", "inputSchema": input_schema.model_dump()}
    assert tool.model_dump() == expected_dump


# Add more tests for edge cases or validation if necessary
