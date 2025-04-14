"""
Tests specifically for the tools module initialization to improve code coverage.
This file focuses on testing the __init__.py module functions and branch coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import pytest
import logging

# Check if running in CI
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Direct import for coverage tracking
import src.cli_code.tools

# Handle imports
try:
    from src.cli_code.tools import get_tool, AVAILABLE_TOOLS
    from src.cli_code.tools.base import BaseTool
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    get_tool = MagicMock
    AVAILABLE_TOOLS = {}
    BaseTool = MagicMock

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestToolsInitModule:
    """Test suite for tools module initialization and tool retrieval."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock logging to prevent actual log outputs
        self.logging_patch = patch('src.cli_code.tools.logging')
        self.mock_logging = self.logging_patch.start()
        
        # Store original AVAILABLE_TOOLS for restoration later
        self.original_tools = AVAILABLE_TOOLS.copy()
        
    def teardown_method(self):
        """Tear down test fixtures."""
        self.logging_patch.stop()
        
        # Restore original AVAILABLE_TOOLS
        global AVAILABLE_TOOLS
        AVAILABLE_TOOLS.clear()
        AVAILABLE_TOOLS.update(self.original_tools)
    
    def test_get_tool_valid(self):
        """Test retrieving a valid tool."""
        # Most tools should be available
        assert 'ls' in AVAILABLE_TOOLS, "Basic 'ls' tool should be available"
        
        # Get a tool instance
        ls_tool = get_tool('ls')
        
        # Verify instance creation
        assert ls_tool is not None
        assert hasattr(ls_tool, 'execute'), "Tool should have execute method"
    
    def test_get_tool_missing(self):
        """Test retrieving a non-existent tool."""
        # Try to get a non-existent tool
        non_existent_tool = get_tool('non_existent_tool')
        
        # Verify error handling
        assert non_existent_tool is None
        self.mock_logging.warning.assert_called_with(
            "Tool 'non_existent_tool' not found in AVAILABLE_TOOLS."
        )
    
    def test_get_tool_summarize_code(self):
        """Test handling of the special summarize_code tool case."""
        # Temporarily add a mock summarize_code tool to AVAILABLE_TOOLS
        mock_summarize_tool = MagicMock()
        global AVAILABLE_TOOLS
        AVAILABLE_TOOLS['summarize_code'] = mock_summarize_tool
        
        # Try to get the tool
        result = get_tool('summarize_code')
        
        # Verify special case handling
        assert result is None
        self.mock_logging.error.assert_called_with(
            "get_tool() called for 'summarize_code', which requires special instantiation with model instance."
        )
    
    def test_get_tool_instantiation_error(self):
        """Test handling of tool instantiation errors."""
        # Create a mock tool class that raises an exception when instantiated
        mock_error_tool = MagicMock()
        mock_error_tool.side_effect = Exception("Instantiation error")
        
        # Add the error-raising tool to AVAILABLE_TOOLS
        global AVAILABLE_TOOLS
        AVAILABLE_TOOLS['error_tool'] = mock_error_tool
        
        # Try to get the tool
        result = get_tool('error_tool')
        
        # Verify error handling
        assert result is None
        self.mock_logging.error.assert_called()  # Should log the error
    
    def test_all_standard_tools_available(self):
        """Test that all standard tools are registered correctly."""
        # Define the core tools that should always be available
        core_tools = ['view', 'edit', 'ls', 'grep', 'glob', 'tree']
        
        # Check each core tool
        for tool_name in core_tools:
            assert tool_name in AVAILABLE_TOOLS, f"Core tool '{tool_name}' should be available"
            
            # Also check that the tool can be instantiated
            tool_instance = get_tool(tool_name)
            assert tool_instance is not None, f"Tool '{tool_name}' should be instantiable"
            assert isinstance(tool_instance, BaseTool), f"Tool '{tool_name}' should be a BaseTool subclass"
    
    @patch('src.cli_code.tools.AVAILABLE_TOOLS', {})
    def test_empty_tools_dict(self):
        """Test behavior when AVAILABLE_TOOLS is empty."""
        # Try to get a tool from an empty dict
        result = get_tool('ls')
        
        # Verify error handling
        assert result is None
        self.mock_logging.warning.assert_called_with(
            "Tool 'ls' not found in AVAILABLE_TOOLS."
        )
    
    def test_optional_tools_registration(self):
        """Test that optional tools are conditionally registered."""
        # Check a few optional tools that should be registered if imports succeeded
        optional_tools = ['bash', 'task_complete', 'create_directory', 'linter_checker', 'formatter', 'test_runner']
        
        for tool_name in optional_tools:
            if tool_name in AVAILABLE_TOOLS:
                # Tool is available, test instantiation
                tool_instance = get_tool(tool_name)
                assert tool_instance is not None, f"Optional tool '{tool_name}' should be instantiable if available"
                assert isinstance(tool_instance, BaseTool), f"Tool '{tool_name}' should be a BaseTool subclass" 