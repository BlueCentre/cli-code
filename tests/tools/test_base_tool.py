"""
Tests for the BaseTool class.
"""

import pytest
from google.generativeai.types import FunctionDeclaration

from src.cli_code.tools.base import BaseTool


class ConcreteTool(BaseTool):
    """Concrete implementation of BaseTool for testing."""

    name = "test_tool"
    description = "Test tool for testing"

    def execute(self, arg1: str, arg2: int = 42, arg3: bool = False):
        """Execute the test tool.

        Args:
            arg1: Required string argument
            arg2: Optional integer argument
            arg3: Optional boolean argument
        """
        return f"Executed with arg1={arg1}, arg2={arg2}, arg3={arg3}"


class MissingNameTool(BaseTool):
    """Tool without a name for testing."""

    description = "Tool without a name"

    def execute(self):
        """Execute the nameless tool."""
        return "Executed nameless tool"


def test_execute_method():
    """Test the execute method of a concrete tool implementation."""
    tool = ConcreteTool()
    result = tool.execute("test")
    assert result == "Executed with arg1=test, arg2=42, arg3=False"

    # Test with custom values
    result = tool.execute("test", 100, True)
    assert result == "Executed with arg1=test, arg2=100, arg3=True"


def test_get_function_declaration():
    """Test generating function declaration from a tool."""
    declaration = ConcreteTool.get_function_declaration()

    # Verify the declaration is of the correct type
    assert isinstance(declaration, FunctionDeclaration)

    # Verify basic properties
    assert declaration.name == "test_tool"
    assert declaration.description == "Test tool for testing"

    # Verify parameters exist
    assert declaration.parameters is not None

    # Since the structure varies between versions, we'll just verify that key parameters exist
    # FunctionDeclaration represents a JSON schema, but its Python representation varies
    params_str = str(declaration.parameters)

    # Verify parameter names appear in the string representation
    assert "arg1" in params_str
    assert "arg2" in params_str
    assert "arg3" in params_str

    # Verify types appear in the string representation
    assert "STRING" in params_str or "string" in params_str
    assert "INTEGER" in params_str or "integer" in params_str
    assert "BOOLEAN" in params_str or "boolean" in params_str

    # Verify required parameter
    assert "required" in params_str.lower()
    assert "arg1" in params_str


def test_get_function_declaration_empty_params():
    """Test generating function declaration for a tool with no parameters."""

    # Define a simple tool class inline
    class NoParamsTool(BaseTool):
        name = "no_params"
        description = "Tool with no parameters"

        def execute(self):
            return "Executed"

    declaration = NoParamsTool.get_function_declaration()

    # Verify the declaration is of the correct type
    assert isinstance(declaration, FunctionDeclaration)

    # Verify properties
    assert declaration.name == "no_params"
    assert declaration.description == "Tool with no parameters"

    # The parameters field exists but should be minimal
    # We'll just verify it doesn't have our test parameters
    if declaration.parameters is not None:
        params_str = str(declaration.parameters)
        assert "arg1" not in params_str
        assert "arg2" not in params_str
        assert "arg3" not in params_str


def test_get_function_declaration_missing_name():
    """Test generating function declaration for a tool without a name."""
    # This should log a warning and return None
    declaration = MissingNameTool.get_function_declaration()

    # Verify result is None
    assert declaration is None


def test_get_function_declaration_error(mocker):
    """Test error handling during function declaration generation."""
    # Mock inspect.signature to raise an exception
    mocker.patch("inspect.signature", side_effect=ValueError("Test error"))

    # Attempt to generate declaration
    declaration = ConcreteTool.get_function_declaration()

    # Verify result is None
    assert declaration is None
