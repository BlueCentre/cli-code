"""
Tests for the MCP client.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from src.cli_code.mcp.client import MCPClient, MCPConfig, MCPMessage, MCPToolCall


class TestMCPMessage(unittest.TestCase):
    """Tests for the MCPMessage class."""

    def test_message_to_dict_minimal(self):
        """Test converting a minimal message to a dict."""
        message = MCPMessage(role="user", content="Hello")
        expected = {"role": "user", "content": "Hello"}
        self.assertEqual(message.to_dict(), expected)

    def test_message_to_dict_complete(self):
        """Test converting a complete message to a dict."""
        tool_calls = [{"id": "call1", "type": "function", "function": {"name": "test"}}]
        message = MCPMessage(
            role="assistant",
            content="Calling tool",
            tool_calls=tool_calls,
            tool_call_id=None,
            name=None
        )
        expected = {
            "role": "assistant",
            "content": "Calling tool",
            "tool_calls": tool_calls
        }
        self.assertEqual(message.to_dict(), expected)

    def test_message_from_dict(self):
        """Test creating a message from a dict."""
        data = {
            "role": "tool",
            "content": "Tool result",
            "tool_call_id": "call1",
            "name": "test_tool"
        }
        message = MCPMessage.from_dict(data)
        self.assertEqual(message.role, "tool")
        self.assertEqual(message.content, "Tool result")
        self.assertEqual(message.tool_call_id, "call1")
        self.assertEqual(message.name, "test_tool")
        self.assertIsNone(message.tool_calls)


class TestMCPToolCall(unittest.TestCase):
    """Tests for the MCPToolCall class."""

    def test_tool_call_to_dict(self):
        """Test converting a tool call to a dict."""
        function = {"name": "test_function", "arguments": '{"arg1": "value1"}'}
        tool_call = MCPToolCall(id="call1", type="function", function=function)
        expected = {
            "id": "call1",
            "type": "function",
            "function": function
        }
        self.assertEqual(tool_call.to_dict(), expected)

    def test_tool_call_from_dict(self):
        """Test creating a tool call from a dict."""
        function = {"name": "test_function", "arguments": '{"arg1": "value1"}'}
        data = {
            "id": "call1",
            "type": "function",
            "function": function
        }
        tool_call = MCPToolCall.from_dict(data)
        self.assertEqual(tool_call.id, "call1")
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function, function)


class TestMCPClient(unittest.TestCase):
    """Tests for the MCPClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MCPConfig(
            server_url="http://test-server",
            api_key="test-key",
            server_config={"timeout": 30},
            model_config={"model": "test-model"},
            tool_config={"tools": []}
        )
        self.client = MCPClient(self.config)
    
    def test_process_response_mcp_format(self):
        """Test processing response in MCP format."""
        response = {
            "message": {
                "role": "assistant",
                "content": "Test response"
            }
        }
        message = self.client.process_response(response)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Test response")
    
    def test_process_response_openai_format(self):
        """Test processing response in OpenAI format."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Test response"
                    }
                }
            ]
        }
        message = self.client.process_response(response)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Test response")
    
    def test_process_response_unexpected_format(self):
        """Test processing response with unexpected format."""
        response = {"unexpected": "format"}
        message = self.client.process_response(response)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Error processing response")
    
    @patch('src.cli_code.mcp.client.MCPClient.send_message')
    def test_handle_tool_call_not_implemented(self, mock_send):
        """Test that handle_tool_call raises NotImplementedError."""
        function = {"name": "test_function", "arguments": '{"arg1": "value1"}'}
        tool_call = MCPToolCall(id="call1", type="function", function=function)
        
        with self.assertRaises(NotImplementedError) as context:
            self.client.handle_tool_call(tool_call)
        
        self.assertIn("Tool execution not implemented", str(context.exception))
        self.assertIn("test_function", str(context.exception))


if __name__ == '__main__':
    unittest.main() 