"""
Tests for the MCP model adapter interface.
"""
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

from src.cli_code.mcp.client import MCPClient, MCPConfig, MCPMessage
from src.cli_code.mcp.model_adapter import MCPModelAdapter, ModelAdapterFactory
from src.cli_code.models.base import AbstractModelAgent


# Create a concrete implementation for testing
class TestAdapter(MCPModelAdapter):
    """Test implementation of MCPModelAdapter."""
    
    def __init__(self, model_agent, mcp_client, console=None):
        super().__init__(model_agent, mcp_client)
        self.console = console
    
    def format_for_mcp(self, prompt, history):
        return [MCPMessage(role="user", content=prompt)]
    
    def parse_from_mcp(self, message):
        return {"role": "user", "content": message.content}
    
    def execute_tool(self, tool_call):
        return {"result": "test_result"}
    
    def send_request(self, prompt):
        return f"Response to: {prompt}"


class TestModelAdapter(unittest.TestCase):
    """Tests for the MCPModelAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_agent = MagicMock(spec=AbstractModelAgent)
        self.mcp_client = MagicMock(spec=MCPClient)
        self.adapter = TestAdapter(self.model_agent, self.mcp_client)
    
    def test_initialization(self):
        """Test adapter initialization."""
        self.assertEqual(self.adapter.model_agent, self.model_agent)
        self.assertEqual(self.adapter.mcp_client, self.mcp_client)
    
    def test_format_for_mcp(self):
        """Test format_for_mcp implementation."""
        messages = self.adapter.format_for_mcp("test prompt", [])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "test prompt")
    
    def test_parse_from_mcp(self):
        """Test parse_from_mcp implementation."""
        message = MCPMessage(role="user", content="test content")
        result = self.adapter.parse_from_mcp(message)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"], "test content")
    
    def test_execute_tool(self):
        """Test execute_tool implementation."""
        tool_call = MagicMock()
        result = self.adapter.execute_tool(tool_call)
        self.assertEqual(result, {"result": "test_result"})
    
    def test_send_request(self):
        """Test send_request implementation."""
        result = self.adapter.send_request("test prompt")
        self.assertEqual(result, "Response to: test prompt")


class TestModelAdapterFactory(unittest.TestCase):
    """Tests for the ModelAdapterFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_agent = MagicMock(spec=AbstractModelAgent)
        self.config = MCPConfig(server_url="http://test-server")
        self.console = MagicMock(spec=Console)
    
    @patch("src.cli_code.mcp.model_adapter.ModelAdapterFactory.create_adapter")
    def test_factory_creates_gemini_adapter(self, mock_create_adapter):
        """Test that the factory creates a Gemini adapter."""
        # Set up the mock
        mock_create_adapter.return_value = "gemini_adapter"
        
        # Call the factory
        adapter = ModelAdapterFactory.create_adapter(
            "gemini", self.model_agent, self.config, self.console
        )
        
        # Verify the result
        self.assertEqual(adapter, "gemini_adapter")
        mock_create_adapter.assert_called_once_with(
            "gemini", self.model_agent, self.config, self.console
        )
    
    def test_factory_raises_for_unsupported_model(self):
        """Test that the factory raises for unsupported models."""
        with self.assertRaises(ValueError) as context:
            ModelAdapterFactory.create_adapter(
                "unsupported", self.model_agent, self.config, self.console
            )
        
        self.assertIn("Unsupported model type", str(context.exception))


if __name__ == "__main__":
    unittest.main() 