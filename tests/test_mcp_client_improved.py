"""
Improved tests for the MCP client.

This module provides comprehensive tests for the MCPClient class focusing
on areas with low test coverage.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.cli_code.mcp.client import MCPClient, MCPMessage, MCPToolCall


class TestMCPMessageImproved(unittest.TestCase):
    """Improved tests for the MCPMessage class."""

    def test_init_basic(self):
        """Test basic initialization of MCPMessage."""
        message = MCPMessage("Hello, world!", "user")
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.role, "user")
        self.assertIsNone(message.tool_calls)
        self.assertIsNone(message.tool_responses)

    def test_init_with_tool_calls(self):
        """Test initialization with tool calls."""
        tool_calls = [MCPToolCall("test_tool", {"param": "value"})]
        message = MCPMessage("Hello, world!", "assistant", tool_calls=tool_calls)
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.role, "assistant")
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].function_name, "test_tool")
        self.assertEqual(message.tool_calls[0].function_args, {"param": "value"})

    def test_init_with_tool_responses(self):
        """Test initialization with tool responses."""
        tool_responses = [{"name": "test_tool", "result": {"success": True}}]
        message = MCPMessage("Result received", "tool", tool_responses=tool_responses)
        self.assertEqual(message.content, "Result received")
        self.assertEqual(message.role, "tool")
        self.assertEqual(len(message.tool_responses), 1)
        self.assertEqual(message.tool_responses[0]["name"], "test_tool")
        self.assertEqual(message.tool_responses[0]["result"]["success"], True)

    def test_to_dict_basic(self):
        """Test converting a basic message to dict."""
        message = MCPMessage("Hello, world!", "user")
        message_dict = message.to_dict()
        self.assertEqual(message_dict["content"], "Hello, world!")
        self.assertEqual(message_dict["role"], "user")
        self.assertNotIn("tool_calls", message_dict)
        self.assertNotIn("tool_responses", message_dict)

    def test_to_dict_with_tool_calls(self):
        """Test converting a message with tool calls to dict."""
        tool_calls = [MCPToolCall("test_tool", {"param": "value"})]
        message = MCPMessage("Hello, world!", "assistant", tool_calls=tool_calls)
        message_dict = message.to_dict()
        self.assertEqual(message_dict["content"], "Hello, world!")
        self.assertEqual(message_dict["role"], "assistant")
        self.assertIn("tool_calls", message_dict)
        self.assertEqual(len(message_dict["tool_calls"]), 1)
        self.assertEqual(message_dict["tool_calls"][0]["function"]["name"], "test_tool")
        self.assertEqual(json.loads(message_dict["tool_calls"][0]["function"]["arguments"]), {"param": "value"})

    def test_to_dict_with_tool_responses(self):
        """Test converting a message with tool responses to dict."""
        tool_responses = [{"name": "test_tool", "result": {"success": True}}]
        message = MCPMessage("Result received", "tool", tool_responses=tool_responses)
        message_dict = message.to_dict()
        self.assertEqual(message_dict["content"], "Result received")
        self.assertEqual(message_dict["role"], "tool")
        self.assertIn("tool_responses", message_dict)
        self.assertEqual(len(message_dict["tool_responses"]), 1)
        self.assertEqual(message_dict["tool_responses"][0]["name"], "test_tool")
        self.assertEqual(message_dict["tool_responses"][0]["result"]["success"], True)

    def test_from_dict_basic(self):
        """Test creating a message from a basic dict."""
        message_dict = {"content": "Hello, world!", "role": "user"}
        message = MCPMessage.from_dict(message_dict)
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.role, "user")
        self.assertIsNone(message.tool_calls)
        self.assertIsNone(message.tool_responses)

    def test_from_dict_with_tool_calls(self):
        """Test creating a message from a dict with tool calls."""
        message_dict = {
            "content": "Hello, world!",
            "role": "assistant",
            "tool_calls": [
                {"id": "call1", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}
            ],
        }
        message = MCPMessage.from_dict(message_dict)
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.role, "assistant")
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].function_name, "test_tool")
        self.assertEqual(message.tool_calls[0].function_args, {"param": "value"})

    def test_from_dict_with_tool_responses(self):
        """Test creating a message from a dict with tool responses."""
        message_dict = {
            "content": "Result received",
            "role": "tool",
            "tool_responses": [{"name": "test_tool", "result": {"success": True}}],
        }
        message = MCPMessage.from_dict(message_dict)
        self.assertEqual(message.content, "Result received")
        self.assertEqual(message.role, "tool")
        self.assertEqual(len(message.tool_responses), 1)
        self.assertEqual(message.tool_responses[0]["name"], "test_tool")
        self.assertEqual(message.tool_responses[0]["result"]["success"], True)


class TestMCPToolCallImproved(unittest.TestCase):
    """Improved tests for the MCPToolCall class."""

    def test_init_basic(self):
        """Test basic initialization of MCPToolCall."""
        tool_call = MCPToolCall("test_tool", {"param": "value"})
        self.assertEqual(tool_call.function_name, "test_tool")
        self.assertEqual(tool_call.function_args, {"param": "value"})
        self.assertIsNotNone(tool_call.id)  # ID should be generated

    def test_init_with_id(self):
        """Test initialization with a specific ID."""
        tool_call = MCPToolCall("test_tool", {"param": "value"}, "custom_id")
        self.assertEqual(tool_call.function_name, "test_tool")
        self.assertEqual(tool_call.function_args, {"param": "value"})
        self.assertEqual(tool_call.id, "custom_id")

    def test_to_dict(self):
        """Test converting a tool call to dict."""
        tool_call = MCPToolCall("test_tool", {"param": "value"}, "custom_id")
        tool_call_dict = tool_call.to_dict()
        self.assertEqual(tool_call_dict["id"], "custom_id")
        self.assertEqual(tool_call_dict["function"]["name"], "test_tool")
        self.assertEqual(json.loads(tool_call_dict["function"]["arguments"]), {"param": "value"})

    def test_from_dict(self):
        """Test creating a tool call from a dict."""
        tool_call_dict = {
            "id": "custom_id",
            "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})},
        }
        tool_call = MCPToolCall.from_dict(tool_call_dict)
        self.assertEqual(tool_call.id, "custom_id")
        self.assertEqual(tool_call.function_name, "test_tool")
        self.assertEqual(tool_call.function_args, {"param": "value"})

    def test_from_dict_with_non_json_args(self):
        """Test creating a tool call from a dict with non-JSON args."""
        tool_call_dict = {"id": "custom_id", "function": {"name": "test_tool", "arguments": "non-json-string"}}
        tool_call = MCPToolCall.from_dict(tool_call_dict)
        self.assertEqual(tool_call.id, "custom_id")
        self.assertEqual(tool_call.function_name, "test_tool")
        self.assertEqual(tool_call.function_args, "non-json-string")


class TestMCPClientImproved(unittest.TestCase):
    """Improved tests for the MCPClient class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock ClientSession
        self.session = MagicMock(spec=aiohttp.ClientSession)

        # Create a mock response for the session
        self.mock_response = MagicMock()
        self.mock_response.status = 200
        self.mock_response.json = AsyncMock(return_value={"status": "success", "response": "Hello, world!"})

        # Set up the session post method to return the mock response
        self.session.post = AsyncMock(return_value=self.mock_response)

        # Create the client with the mock session
        self.client = MCPClient("https://example.com", session=self.session)

    @pytest.mark.asyncio
    async def test_send_message_basic(self):
        """Test sending a basic message."""
        # Send a basic message
        response = await self.client.send_message("Hello, world!")

        # Check that the session post method was called with the right parameters
        self.session.post.assert_called_once()
        call_args = self.session.post.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/v1/chat/completions")

        # Check the response
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["response"], "Hello, world!")

    @pytest.mark.asyncio
    async def test_send_message_with_conversation_id(self):
        """Test sending a message with a conversation ID."""
        # Send a message with a conversation ID
        response = await self.client.send_message("Hello, world!", "conv123")

        # Check that the session post method was called with the right parameters
        self.session.post.assert_called_once()
        call_args = self.session.post.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/v1/chat/completions")

        # Check the JSON payload
        call_kwargs = self.session.post.call_args[1]
        json_payload = call_kwargs["json"]
        self.assertIn("conversation_id", json_payload)
        self.assertEqual(json_payload["conversation_id"], "conv123")

        # Check the response
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["response"], "Hello, world!")

    @pytest.mark.asyncio
    async def test_send_message_with_history(self):
        """Test sending a message with conversation history."""
        # Create conversation history
        history = [MCPMessage("Hello", "user"), MCPMessage("Hi there!", "assistant")]

        # Send a message with history
        response = await self.client.send_message("How are you?", conversation_history=history)

        # Check that the session post method was called with the right parameters
        self.session.post.assert_called_once()
        call_args = self.session.post.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/v1/chat/completions")

        # Check the JSON payload
        call_kwargs = self.session.post.call_args[1]
        json_payload = call_kwargs["json"]
        self.assertIn("conversation_history", json_payload)
        self.assertEqual(len(json_payload["conversation_history"]), 2)
        self.assertEqual(json_payload["conversation_history"][0]["content"], "Hello")
        self.assertEqual(json_payload["conversation_history"][0]["role"], "user")
        self.assertEqual(json_payload["conversation_history"][1]["content"], "Hi there!")
        self.assertEqual(json_payload["conversation_history"][1]["role"], "assistant")

        # Check the response
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["response"], "Hello, world!")

    @pytest.mark.asyncio
    async def test_send_message_with_model_config(self):
        """Test sending a message with model configuration."""
        # Create model config
        model_config = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 100}

        # Send a message with model config
        response = await self.client.send_message("Hello, world!", model_config=model_config)

        # Check that the session post method was called with the right parameters
        self.session.post.assert_called_once()
        call_args = self.session.post.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/v1/chat/completions")

        # Check the JSON payload
        call_kwargs = self.session.post.call_args[1]
        json_payload = call_kwargs["json"]
        self.assertIn("model_config", json_payload)
        self.assertEqual(json_payload["model_config"]["temperature"], 0.7)
        self.assertEqual(json_payload["model_config"]["top_p"], 0.9)
        self.assertEqual(json_payload["model_config"]["max_tokens"], 100)

        # Check the response
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["response"], "Hello, world!")

    @pytest.mark.asyncio
    async def test_send_message_with_response_format(self):
        """Test sending a message with a response format."""
        # Send a message with a response format
        response = await self.client.send_message("Hello, world!", response_format={"type": "json"})

        # Check that the session post method was called with the right parameters
        self.session.post.assert_called_once()
        call_args = self.session.post.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/v1/chat/completions")

        # Check the JSON payload
        call_kwargs = self.session.post.call_args[1]
        json_payload = call_kwargs["json"]
        self.assertIn("response_format", json_payload)
        self.assertEqual(json_payload["response_format"]["type"], "json")

        # Check the response
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["response"], "Hello, world!")

    @pytest.mark.asyncio
    async def test_send_message_with_all_options(self):
        """Test sending a message with all options."""
        # Create conversation history
        history = [MCPMessage("Hello", "user"), MCPMessage("Hi there!", "assistant")]

        # Create model config
        model_config = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 100}

        # Send a message with all options
        response = await self.client.send_message("How are you?", "conv123", history, model_config, {"type": "json"})

        # Check that the session post method was called with the right parameters
        self.session.post.assert_called_once()
        call_args = self.session.post.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/v1/chat/completions")

        # Check the JSON payload
        call_kwargs = self.session.post.call_args[1]
        json_payload = call_kwargs["json"]
        self.assertEqual(json_payload["message"], "How are you?")
        self.assertEqual(json_payload["conversation_id"], "conv123")
        self.assertEqual(len(json_payload["conversation_history"]), 2)
        self.assertEqual(json_payload["model_config"]["temperature"], 0.7)
        self.assertEqual(json_payload["response_format"]["type"], "json")

        # Check the response
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["response"], "Hello, world!")

    @pytest.mark.asyncio
    async def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        # Mock the response for tool definitions
        mock_tools_response = MagicMock()
        mock_tools_response.status = 200
        mock_tools_response.json = AsyncMock(
            return_value={
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"param1": {"type": "string"}},
                            "required": ["param1"],
                        },
                    }
                ]
            }
        )

        # Set up the session get method to return the mock response
        self.session.get = AsyncMock(return_value=mock_tools_response)

        # Get tool definitions
        tools = await self.client.get_tool_definitions()

        # Check that the session get method was called with the right parameters
        self.session.get.assert_called_once()
        call_args = self.session.get.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/v1/tools")

        # Check the tools
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test_tool")
        self.assertEqual(tools[0]["description"], "A test tool")
        self.assertIn("parameters", tools[0])

    @pytest.mark.asyncio
    async def test_error_response(self):
        """Test handling an error response."""
        # Mock an error response
        mock_error_response = MagicMock()
        mock_error_response.status = 400
        mock_error_response.json = AsyncMock(return_value={"status": "error", "error": "Invalid request"})

        # Set up the session post method to return the error response
        self.session.post = AsyncMock(return_value=mock_error_response)

        # Send a message
        response = await self.client.send_message("Hello, world!")

        # Check the response
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["error"], "Invalid request")

    @pytest.mark.asyncio
    async def test_http_error(self):
        """Test handling an HTTP error."""
        # Mock a session that raises an exception
        error_session = MagicMock(spec=aiohttp.ClientSession)
        error_session.post = AsyncMock(side_effect=aiohttp.ClientError("Connection error"))

        # Create a client with the error session
        error_client = MCPClient("https://example.com", session=error_session)

        # Send a message
        response = await error_client.send_message("Hello, world!")

        # Check the response
        self.assertEqual(response["status"], "error")
        self.assertIn("error", response)
        self.assertIn("Connection error", response["error"])

    @pytest.mark.asyncio
    async def test_json_decode_error(self):
        """Test handling a JSON decode error."""
        # Mock a response that raises a JSON decode error
        mock_json_error_response = MagicMock()
        mock_json_error_response.status = 200
        mock_json_error_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))

        # Set up the session post method to return the error response
        self.session.post = AsyncMock(return_value=mock_json_error_response)

        # Send a message
        response = await self.client.send_message("Hello, world!")

        # Check the response
        self.assertEqual(response["status"], "error")
        self.assertIn("error", response)
        self.assertIn("Invalid JSON", response["error"])

    @pytest.mark.asyncio
    async def test_check_tool_service_available(self):
        """Test checking if the tool service is available."""
        # Mock the response for the health check
        mock_health_response = MagicMock()
        mock_health_response.status = 200

        # Set up the session get method to return the mock response
        self.session.get = AsyncMock(return_value=mock_health_response)

        # Check if the tool service is available
        is_available = await self.client.check_tool_service_available()

        # Check that the session get method was called with the right parameters
        self.session.get.assert_called_once()
        call_args = self.session.get.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/health")

        # Check the result
        self.assertTrue(is_available)

    @pytest.mark.asyncio
    async def test_check_tool_service_unavailable(self):
        """Test checking if the tool service is unavailable."""
        # Mock the response for the health check
        mock_health_response = MagicMock()
        mock_health_response.status = 500

        # Set up the session get method to return the mock response
        self.session.get = AsyncMock(return_value=mock_health_response)

        # Check if the tool service is available
        is_available = await self.client.check_tool_service_available()

        # Check that the session get method was called with the right parameters
        self.session.get.assert_called_once()
        call_args = self.session.get.call_args[0]
        self.assertEqual(call_args[0], "https://example.com/health")

        # Check the result
        self.assertFalse(is_available)

    @pytest.mark.asyncio
    async def test_check_tool_service_error(self):
        """Test checking if the tool service is available with an error."""
        # Mock a session that raises an exception
        error_session = MagicMock(spec=aiohttp.ClientSession)
        error_session.get = AsyncMock(side_effect=aiohttp.ClientError("Connection error"))

        # Create a client with the error session
        error_client = MCPClient("https://example.com", session=error_session)

        # Check if the tool service is available
        is_available = await error_client.check_tool_service_available()

        # Check the result
        self.assertFalse(is_available)

    @pytest.mark.asyncio
    async def test_process_response_with_tool_calls(self):
        """Test processing a response with tool calls."""
        # Create a response with tool calls
        response = {
            "response": "I'll help with that",
            "tool_calls": [
                {"id": "call1", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}
            ],
        }

        # Process the response
        processed_response = self.client._process_response(response)

        # Check the processed response
        self.assertEqual(processed_response["response"], "I'll help with that")
        self.assertIn("tool_calls", processed_response)
        self.assertEqual(len(processed_response["tool_calls"]), 1)
        self.assertEqual(processed_response["tool_calls"][0]["id"], "call1")
        self.assertEqual(processed_response["tool_calls"][0]["function"]["name"], "test_tool")

    @pytest.mark.asyncio
    async def test_process_response_error(self):
        """Test processing an error response."""
        # Create an error response
        response = {"status": "error", "error": "Invalid request"}

        # Process the response
        processed_response = self.client._process_response(response)

        # Check the processed response
        self.assertEqual(processed_response["status"], "error")
        self.assertEqual(processed_response["error"], "Invalid request")
