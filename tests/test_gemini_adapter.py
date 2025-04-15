"""
Tests for the Gemini model adapter.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

# Mock the tools module for testing
import sys
from unittest.mock import MagicMock

# Create module mocks
mock_tools = MagicMock()
mock_get_tool = MagicMock()

# Add them to sys.modules to avoid import errors
sys.modules['src.cli_code.tools'] = mock_tools
sys.modules['src.cli_code.tools.base'] = MagicMock()
mock_tools.get_tool = mock_get_tool

from src.cli_code.mcp.adapters.gemini_adapter import GeminiModelAdapter
from src.cli_code.mcp.client import MCPClient, MCPMessage, MCPToolCall


class TestGeminiModelAdapter(unittest.TestCase):
    """Tests for the GeminiModelAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_agent = MagicMock()
        self.mcp_client = MagicMock(spec=MCPClient)
        self.console = MagicMock(spec=Console)
        self.adapter = GeminiModelAdapter(self.model_agent, self.mcp_client, self.console)
        # Reset the mock_get_tool before each test
        mock_get_tool.reset_mock()
    
    def tearDown(self):
        """Clean up after each test."""
        # Reset mocks to ensure they don't affect other tests
        mock_get_tool.reset_mock()
    
    def test_map_gemini_role_to_mcp(self):
        """Test mapping Gemini roles to MCP roles."""
        self.assertEqual(self.adapter._map_gemini_role_to_mcp("user"), "user")
        self.assertEqual(self.adapter._map_gemini_role_to_mcp("model"), "assistant")
        self.assertEqual(self.adapter._map_gemini_role_to_mcp("system"), "system")
        self.assertEqual(self.adapter._map_gemini_role_to_mcp("function"), "tool")
        self.assertEqual(self.adapter._map_gemini_role_to_mcp("unknown"), "user")  # Default
    
    def test_map_mcp_role_to_gemini(self):
        """Test mapping MCP roles to Gemini roles."""
        self.assertEqual(self.adapter._map_mcp_role_to_gemini("user"), "user")
        self.assertEqual(self.adapter._map_mcp_role_to_gemini("assistant"), "model")
        self.assertEqual(self.adapter._map_mcp_role_to_gemini("system"), "system")
        self.assertEqual(self.adapter._map_mcp_role_to_gemini("tool"), "function")
        self.assertEqual(self.adapter._map_mcp_role_to_gemini("unknown"), "user")  # Default
    
    def test_format_for_mcp_text_content(self):
        """Test formatting history with text content for MCP."""
        history = [
            {"role": "user", "parts": ["Hello, how are you?"]},
            {"role": "model", "parts": ["I'm doing well, thank you!"]}
        ]
        
        mcp_messages = self.adapter.format_for_mcp("test prompt", history)
        
        self.assertEqual(len(mcp_messages), 2)
        
        self.assertEqual(mcp_messages[0].role, "user")
        self.assertEqual(mcp_messages[0].content, "Hello, how are you?")
        
        self.assertEqual(mcp_messages[1].role, "assistant")
        self.assertEqual(mcp_messages[1].content, "I'm doing well, thank you!")
    
    def test_format_for_mcp_with_function_call(self):
        """Test formatting history with function calls for MCP."""
        history = [
            {"role": "user", "parts": ["What's the weather in New York?"]},
            {
                "role": "model", 
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"location": "New York"}
                        }
                    }
                ]
            }
        ]
        
        mcp_messages = self.adapter.format_for_mcp("test prompt", history)
        
        self.assertEqual(len(mcp_messages), 2)
        
        self.assertEqual(mcp_messages[0].role, "user")
        self.assertEqual(mcp_messages[0].content, "What's the weather in New York?")
        
        self.assertEqual(mcp_messages[1].role, "assistant")
        self.assertIsNone(mcp_messages[1].content)
        self.assertIsNotNone(mcp_messages[1].tool_calls)
        self.assertEqual(mcp_messages[1].tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(mcp_messages[1].tool_calls[0]["function"]["arguments"], {"location": "New York"})
    
    def test_parse_from_mcp_text_content(self):
        """Test parsing MCP message with text content."""
        mcp_message = MCPMessage(role="assistant", content="Hello, how can I help?")
        
        gemini_message = self.adapter.parse_from_mcp(mcp_message)
        
        self.assertEqual(gemini_message["role"], "model")
        self.assertEqual(len(gemini_message["parts"]), 1)
        self.assertEqual(gemini_message["parts"][0], "Hello, how can I help?")
    
    def test_parse_from_mcp_with_tool_calls(self):
        """Test parsing MCP message with tool calls."""
        tool_calls = [
            {
                "id": "call_test_123",
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": {"param1": "value1"}
                }
            }
        ]
        mcp_message = MCPMessage(role="assistant", content=None, tool_calls=tool_calls)
        
        gemini_message = self.adapter.parse_from_mcp(mcp_message)
        
        self.assertEqual(gemini_message["role"], "model")
        self.assertEqual(len(gemini_message["parts"]), 1)
        self.assertIn("function_call", gemini_message["parts"][0])
        self.assertEqual(gemini_message["parts"][0]["function_call"]["name"], "test_function")
        self.assertEqual(gemini_message["parts"][0]["function_call"]["args"], {"param1": "value1"})
    
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        # Set up the mock
        tool_impl = MagicMock(return_value="Tool result")
        mock_get_tool.return_value = tool_impl
        
        # Create a tool call
        tool_call = MCPToolCall(
            id="call_test_123",
            type="function",
            function={"name": "test_tool", "arguments": {"param1": "value1"}}
        )
        
        # Execute the tool
        result = self.adapter.execute_tool(tool_call)
        
        # Verify result
        self.assertEqual(result, {"result": "Tool result"})
        mock_get_tool.assert_called_once_with("test_tool")
        tool_impl.assert_called_once_with({"param1": "value1"}, self.console)
    
    def test_execute_tool_not_found(self):
        """Test tool execution when tool is not found."""
        # Set up the mock
        mock_get_tool.return_value = None
        
        # Create a tool call
        tool_call = MCPToolCall(
            id="call_test_123",
            type="function",
            function={"name": "nonexistent_tool", "arguments": {}}
        )
        
        # Execute the tool
        result = self.adapter.execute_tool(tool_call)
        
        # Verify result
        self.assertEqual(result, {"error": "Tool not found: nonexistent_tool"})
    
    def test_execute_tool_error(self):
        """Test tool execution when tool raises an error."""
        # Set up the mock
        tool_impl = MagicMock(side_effect=Exception("Tool error"))
        mock_get_tool.return_value = tool_impl
        
        # Create a tool call
        tool_call = MCPToolCall(
            id="call_test_123",
            type="function",
            function={"name": "error_tool", "arguments": {}}
        )
        
        # Execute the tool
        result = self.adapter.execute_tool(tool_call)
        
        # Verify result
        self.assertEqual(result, {"error": "Error executing tool error_tool: Tool error"})
    
    def test_send_request(self):
        """Test sending a request through the adapter."""
        # Set up the model agent mock
        self.model_agent.history = [
            {"role": "user", "parts": ["Hello"]}
        ]
        self.model_agent.generate.return_value = "Response from model"
        
        # Send a request
        result = self.adapter.send_request("test prompt")
        
        # Verify result
        self.assertEqual(result, "Response from model")
        self.model_agent.generate.assert_called_once_with("test prompt")


if __name__ == "__main__":
    unittest.main() 