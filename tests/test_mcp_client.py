"""
Tests for the MCP client.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from src.cli_code.mcp.client import MCPClient, MCPMessage, MCPToolCall


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
            role="assistant", content="Calling tool", tool_calls=tool_calls, tool_call_id=None, name=None
        )
        expected = {"role": "assistant", "content": "Calling tool", "tool_calls": tool_calls}
        self.assertEqual(message.to_dict(), expected)

    def test_message_from_dict(self):
        """Test creating a message from a dict."""
        data = {"role": "tool", "content": "Tool result", "tool_call_id": "call1", "name": "test_tool"}
        message = MCPMessage.from_dict(data)
        self.assertEqual(message.role, "tool")
        self.assertEqual(message.content, "Tool result")
        self.assertEqual(message.tool_call_id, "call1")
        self.assertEqual(message.name, "test_tool")
        self.assertIsNone(message.tool_calls)

    def test_message_to_dict_with_tool_call_id(self):
        """Test converting a message with tool_call_id to a dict."""
        message = MCPMessage(role="tool", content="Tool result", tool_call_id="call1", name="test_tool")
        expected = {"role": "tool", "content": "Tool result", "tool_call_id": "call1", "name": "test_tool"}
        self.assertEqual(message.to_dict(), expected)

    def test_message_to_dict_without_content(self):
        """Test converting a message without content to a dict."""
        tool_calls = [{"id": "call1", "type": "function", "function": {"name": "test"}}]
        message = MCPMessage(role="assistant", content=None, tool_calls=tool_calls)
        expected = {"role": "assistant", "tool_calls": tool_calls}
        self.assertEqual(message.to_dict(), expected)


class TestMCPToolCall(unittest.TestCase):
    """Tests for the MCPToolCall class."""

    def test_tool_call_to_dict(self):
        """Test converting a tool call to a dict."""
        function = {"name": "test_function", "arguments": '{"arg1": "value1"}'}
        tool_call = MCPToolCall(id="call1", type="function", function=function)
        expected = {"id": "call1", "type": "function", "function": function}
        self.assertEqual(tool_call.to_dict(), expected)

    def test_tool_call_from_dict(self):
        """Test creating a tool call from a dict."""
        function = {"name": "test_function", "arguments": '{"arg1": "value1"}'}
        data = {"id": "call1", "type": "function", "function": function}
        tool_call = MCPToolCall.from_dict(data)
        self.assertEqual(tool_call.id, "call1")
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function, function)


class TestMCPClient(unittest.TestCase):
    """Tests for the MCPClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.endpoint = "http://test-server/v1/chat/completions"
        self.api_key = "test-key"
        self.model = "test-model"
        self.client = MCPClient(endpoint=self.endpoint, api_key=self.api_key, model=self.model)

    def test_init(self):
        """Test client initialization."""
        self.assertEqual(self.client.endpoint, self.endpoint)
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertEqual(self.client.model, self.model)

        # Test without API key
        client = MCPClient(endpoint=self.endpoint)
        self.assertIsNone(client.api_key)
        self.assertEqual(client.model, "default")

    def test_process_response_mcp_format(self):
        """Test processing response in MCP format."""
        response = {"message": {"role": "assistant", "content": "Test response"}}
        message = self.client.process_response(response)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Test response")

    def test_process_response_openai_format(self):
        """Test processing response in OpenAI format."""
        response = {"choices": [{"message": {"role": "assistant", "content": "Test response"}}]}
        message = self.client.process_response(response)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Test response")

    def test_process_response_unexpected_format(self):
        """Test processing response with unexpected format."""
        response = {"unexpected": "format"}
        message = self.client.process_response(response)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Error processing response")

    def test_handle_tool_call_not_implemented(self):
        """Test that handle_tool_call raises NotImplementedError."""
        function = {"name": "test_function", "arguments": '{"arg1": "value1"}'}
        tool_call = MCPToolCall(id="call1", type="function", function=function)

        with self.assertRaises(NotImplementedError) as context:
            self.client.handle_tool_call(tool_call)

        self.assertIn("Tool execution not implemented", str(context.exception))
        self.assertIn("test_function", str(context.exception))

    @pytest.mark.asyncio
    async def test_send_request_success(self):
        """Test successful request sending."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": {"role": "assistant", "content": "Response"}})

        # Mock the session
        mock_session = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock ClientSession
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await self.client.send_request(
                messages=[{"role": "user", "content": "Hello"}],
                tools=[{"name": "test_tool"}],
                temperature=0.5,
                max_tokens=100,
            )

            # Check the result
            self.assertEqual(result, {"message": {"role": "assistant", "content": "Response"}})

            # Check that post was called with the right arguments
            mock_session.post.assert_called_once()
            args, kwargs = mock_session.post.call_args
            self.assertEqual(args[0], self.endpoint)

            # Check headers
            self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.api_key}")
            self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

            # Check payload
            self.assertEqual(kwargs["json"]["model"], self.model)
            self.assertEqual(kwargs["json"]["messages"], [{"role": "user", "content": "Hello"}])
            self.assertEqual(kwargs["json"]["tools"], [{"name": "test_tool"}])
            self.assertEqual(kwargs["json"]["temperature"], 0.5)
            self.assertEqual(kwargs["json"]["max_tokens"], 100)

    @pytest.mark.asyncio
    async def test_send_request_without_api_key(self):
        """Test request sending without API key."""
        # Create client without API key
        client = MCPClient(endpoint=self.endpoint)

        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": {"role": "assistant", "content": "Response"}})

        # Mock the session
        mock_session = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock ClientSession
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.send_request(messages=[{"role": "user", "content": "Hello"}])

            # Check the headers do not contain Authorization
            args, kwargs = mock_session.post.call_args
            self.assertNotIn("Authorization", kwargs["headers"])

    @pytest.mark.asyncio
    async def test_send_request_http_error(self):
        """Test request sending with HTTP error."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad request")

        # Mock the session
        mock_session = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock ClientSession
        with patch("aiohttp.ClientSession", return_value=mock_session):
            with self.assertRaises(Exception) as context:
                await self.client.send_request(messages=[{"role": "user", "content": "Hello"}])

            self.assertIn("Error from MCP server", str(context.exception))
            self.assertIn("400", str(context.exception))
            self.assertIn("Bad request", str(context.exception))

    @pytest.mark.asyncio
    async def test_send_request_client_error(self):
        """Test request sending with client error."""
        # Mock ClientSession to raise an error
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.side_effect = aiohttp.ClientError("Connection error")

            with self.assertRaises(Exception) as context:
                await self.client.send_request(messages=[{"role": "user", "content": "Hello"}])

            self.assertIn("Error sending request", str(context.exception))
            self.assertIn("Connection error", str(context.exception))


if __name__ == "__main__":
    unittest.main()
