"""
Tests for the MCP model adapter interface.
"""

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from rich.console import Console

# Import the expected concrete adapter (adjust path if necessary)
from src.cli_code.mcp.adapters.gemini_adapter import GeminiModelAdapter
from src.cli_code.mcp.client import MCPClient, MCPConfig, MCPMessage, MCPToolCall
from src.cli_code.mcp.model_adapter import MCPModelAdapter, ModelAdapterFactory
from src.cli_code.models.base import AbstractModelAgent


# Define a concrete mock adapter for testing the abstract base class interactions
class MockAdapter(MCPModelAdapter):
    def format_for_mcp(self, prompt: str, history: List[Dict[str, Any]]) -> List[MCPMessage]:
        # Simple mock implementation returning a list containing one message
        return [MCPMessage(role="user", content=prompt)]

    def parse_from_mcp(self, message: MCPMessage) -> Dict[str, Any]:
        # Simple mock implementation returning a dict
        return {"role": message.role, "parts": [message.content]}

    def execute_tool(self, tool_call: MCPToolCall) -> Dict[str, Any]:
        # Simple mock implementation returning a dict
        return {"result": "test_result"}

    def send_request(self, prompt: str) -> Optional[str]:
        # Simple mock implementation returning a string
        return f"Response to: {prompt}"


class TestModelAdapter(unittest.TestCase):
    """Tests for the MCPModelAdapter abstract base class setup and basic structure."""

    def setUp(self):
        """Set up mocks for model agent and MCP client."""
        self.mock_model_agent = MagicMock(spec=AbstractModelAgent)
        self.mock_mcp_client = MagicMock(spec=MCPClient)
        # Instantiate the concrete MockAdapter instead of MagicMock(spec=MCPModelAdapter)
        self.adapter = MockAdapter(self.mock_model_agent, self.mock_mcp_client)

    def test_init(self):
        """Test adapter initialization."""
        self.assertEqual(self.adapter.model_agent, self.mock_model_agent)
        self.assertEqual(self.adapter.mcp_client, self.mock_mcp_client)

    def test_format_for_mcp(self):
        """Test format_for_mcp implementation."""
        messages = self.adapter.format_for_mcp("test prompt", [])
        # Assert based on the MockAdapter implementation
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], MCPMessage)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "test prompt")

    def test_parse_from_mcp(self):
        """Test parse_from_mcp implementation."""
        message = MCPMessage(role="user", content="test content")
        result = self.adapter.parse_from_mcp(message)
        # Assert based on the MockAdapter implementation
        self.assertIsInstance(result, dict)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["parts"], ["test content"])

    def test_execute_tool(self):
        """Test execute_tool implementation."""
        tool_call = MagicMock(spec=MCPToolCall)
        result = self.adapter.execute_tool(tool_call)
        # Assert based on the MockAdapter implementation
        self.assertEqual(result, {"result": "test_result"})

    def test_send_request(self):
        """Test send_request implementation."""
        result = self.adapter.send_request("test prompt")
        # Assert based on the MockAdapter implementation
        self.assertEqual(result, "Response to: test prompt")


class TestModelAdapterFactory(unittest.TestCase):
    """Tests for the ModelAdapterFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_agent = MagicMock(spec=AbstractModelAgent)
        self.config = MCPConfig(server_url="http://test-server")
        self.console = MagicMock(spec=Console)
        self.mock_console = MagicMock(spec=Console)
        self.mock_mcp_config = MagicMock(spec=MCPConfig)

    # Patch the target where GeminiModelAdapter is imported/used within the factory method
    @patch("src.cli_code.mcp.adapters.gemini_adapter.GeminiModelAdapter")
    @patch("src.cli_code.mcp.model_adapter.MCPClient")  # Also patch MCPClient used internally
    def test_factory_creates_gemini_adapter(self, mock_mcp_client_cls, mock_gemini_adapter_cls):
        """Test that the factory creates a Gemini adapter."""
        # Configure mocks
        mock_mcp_instance = MagicMock(spec=MCPClient)
        mock_mcp_client_cls.return_value = mock_mcp_instance
        mock_adapter_instance = MagicMock()
        mock_gemini_adapter_cls.return_value = mock_adapter_instance

        adapter = ModelAdapterFactory.create_adapter(
            model_type="gemini",
            model_agent=self.model_agent,
            mcp_config=self.mock_mcp_config,
            console=self.mock_console,
        )

        # Check that MCPClient was instantiated with the config
        mock_mcp_client_cls.assert_called_once_with(self.mock_mcp_config)
        # Check that the Gemini adapter was instantiated correctly
        mock_gemini_adapter_cls.assert_called_once_with(self.model_agent, mock_mcp_instance, self.mock_console)
        # Check that the returned adapter is the instance created by the mock
        self.assertEqual(adapter, mock_adapter_instance)

    def test_factory_raises_not_implemented_for_openai(self):
        """Test factory raises NotImplementedError for OpenAI."""
        with self.assertRaises(NotImplementedError) as context:
            ModelAdapterFactory.create_adapter("openai", self.model_agent, self.config, self.console)
        self.assertIn("OpenAI adapter not yet implemented", str(context.exception))

    def test_factory_raises_not_implemented_for_ollama(self):
        """Test factory raises NotImplementedError for Ollama."""
        with self.assertRaises(NotImplementedError) as context:
            ModelAdapterFactory.create_adapter("ollama", self.model_agent, self.config, self.console)
        self.assertIn("Ollama adapter not yet implemented", str(context.exception))

    def test_factory_raises_value_error_for_unsupported(self):
        """Test factory raises ValueError for unknown models."""
        with self.assertRaises(ValueError) as context:
            ModelAdapterFactory.create_adapter("unknown_model", self.model_agent, self.config, self.console)
        self.assertIn("Unsupported model type: unknown_model", str(context.exception))


# Keep the minimal test for the ABC just to ensure __init__ works
# It doesn't contribute much to coverage but confirms basic structure
class TestMCPModelAdapterABC(unittest.TestCase):
    def test_initialization(self):
        model_agent = MagicMock(spec=AbstractModelAgent)
        mcp_client = MagicMock(spec=MCPClient)
        # Instantiate with a minimal concrete class
        adapter = MockAdapter(model_agent, mcp_client)
        self.assertIs(adapter.model_agent, model_agent)
        self.assertIs(adapter.mcp_client, mcp_client)


if __name__ == "__main__":
    unittest.main()
