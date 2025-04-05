"""
Tests for the BaseTool base class.
"""
import pytest
from unittest.mock import patch, MagicMock

from cli_code.tools.base import BaseTool


class TestTool(BaseTool):
    """A concrete implementation of BaseTool for testing."""
    
    name = "test_tool"
    description = "Test tool for testing purposes"
    
    def execute(self, param1: str, param2: int = 0, param3: bool = False):
        """Execute the test tool.
        
        Args:
            param1: A string parameter
            param2: An integer parameter with default
            param3: A boolean parameter with default
            
        Returns:
            A string response
        """
        return f"Executed with {param1}, {param2}, {param3}"


def test_tool_execute():
    """Test the execute method of the concrete implementation."""
    tool = TestTool()
    result = tool.execute("test", 42, True)
    
    assert result == "Executed with test, 42, True"
    
    # Test with default values
    result = tool.execute("test")
    assert result == "Executed with test, 0, False"


def test_get_function_declaration():
    """Test the get_function_declaration method."""
    # Create a simple test that works without mocking
    declaration = TestTool.get_function_declaration()
    
    # Basic assertions about the declaration that don't depend on implementation details
    assert declaration is not None
    assert declaration.name == "test_tool"
    assert declaration.description == "Test tool for testing purposes"
    
    # Create a simple representation of the parameters to test
    # This avoids depending on the exact Schema implementation
    param_repr = str(declaration.parameters)
    
    # Check if key parameters are mentioned in the string representation
    assert "param1" in param_repr
    assert "param2" in param_repr
    assert "param3" in param_repr
    assert "STRING" in param_repr  # Uppercase in the string representation
    assert "INTEGER" in param_repr  # Uppercase in the string representation
    assert "BOOLEAN" in param_repr  # Uppercase in the string representation
    assert "required" in param_repr


def test_get_function_declaration_no_name():
    """Test get_function_declaration when name is missing."""
    class NoNameTool(BaseTool):
        name = None
        description = "Tool with no name"
        
        def execute(self, param: str):
            return f"Executed with {param}"
    
    with patch("cli_code.tools.base.log") as mock_log:
        declaration = NoNameTool.get_function_declaration()
        assert declaration is None
        mock_log.warning.assert_called_once()


def test_abstract_class_methods():
    """Test that BaseTool cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTool() 