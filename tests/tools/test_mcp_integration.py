"""
Tests for the MCPToolIntegration class.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli_code.mcp.client import MCPClient, MCPMessage, MCPToolCall
from src.cli_code.mcp.integrations import MCPToolIntegration
from src.cli_code.mcp.tools.models import Tool, ToolParameter
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService


class TestMCPToolIntegration(unittest.TestCase):
    """Tests for the MCPToolIntegration class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock tool service
        self.registry = ToolRegistry()
        self.tool_service = ToolService(self.registry)

        # Mock the execute_tool method
        self.tool_service.execute_tool = AsyncMock(
            return_value={
                "name": "test_tool",
                "parameters": {"param1": "test_value"},
                "result": {"key": "value"},
                "success": True,
            }
        )

        # Create a mock MCP client
        self.client = MagicMock(spec=MCPClient)
        self.client.send_request = AsyncMock()
        self.client.process_response = MagicMock()

        # Create the integration
        self.integration = MCPToolIntegration(self.client, self.tool_service)

        # Create a test tool
        self.test_tool = Tool(
            name="test_tool",
            description="Test tool",
            parameters=[ToolParameter(name="param1", description="Test parameter", type="string", required=True)],
            handler=AsyncMock(return_value={"key": "value"}),
        )
        self.registry.register(self.test_tool)

        # Mock the tool service get_available_tools method
        self.tool_service.get_available_tools = MagicMock(
            return_value={
                "test_tool": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"param1": {"type": "string", "description": "Test parameter"}},
                        "required": ["param1"],
                    },
                }
            }
        )

    @pytest.mark.asyncio
    async def test_handle_tool_call(self):
        """Test handling a tool call."""
        # Create a tool call
        tool_call = MCPToolCall(
            id="test_id",
            type="function",
            function={"name": "test_tool", "arguments": json.dumps({"param1": "test_value"})},
        )

        # Handle the tool call
        result = await self.integration.handle_tool_call(tool_call)

        # Verify the result
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "test_id")
        self.assertEqual(result["name"], "test_tool")
        self.assertEqual(json.loads(result["content"]), {"key": "value"})

        # Verify the tool service was called
        self.tool_service.execute_tool.assert_called_once_with("test_tool", {"param1": "test_value"})

    @pytest.mark.asyncio
    async def test_handle_tool_call_with_object_arguments(self):
        """Test handling a tool call with object arguments."""
        # Create a tool call with object arguments
        tool_call = MCPToolCall(
            id="test_id", type="function", function={"name": "test_tool", "arguments": {"param1": "test_value"}}
        )

        # Handle the tool call
        result = await self.integration.handle_tool_call(tool_call)

        # Verify the result
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "test_id")
        self.assertEqual(result["name"], "test_tool")
        self.assertEqual(json.loads(result["content"]), {"key": "value"})

        # Verify the tool service was called
        self.tool_service.execute_tool.assert_called_once_with("test_tool", {"param1": "test_value"})

    @pytest.mark.asyncio
    async def test_handle_tool_call_with_invalid_json(self):
        """Test handling a tool call with invalid JSON arguments."""
        # Create a tool call with invalid JSON
        tool_call = MCPToolCall(
            id="test_id", type="function", function={"name": "test_tool", "arguments": "invalid json"}
        )

        # Handle the tool call
        result = await self.integration.handle_tool_call(tool_call)

        # Verify the result
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "test_id")
        self.assertEqual(result["name"], "test_tool")

        # Verify the tool service was called with empty parameters
        self.tool_service.execute_tool.assert_called_once_with("test_tool", {})

    @pytest.mark.asyncio
    async def test_process_assistant_message_with_tool_calls(self):
        """Test processing an assistant message with tool calls."""
        # Create a message with tool calls
        message = MCPMessage(
            role="assistant",
            content="Using tools",
            tool_calls=[
                {
                    "id": "test_id_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": json.dumps({"param1": "test_value_1"})},
                },
                {
                    "id": "test_id_2",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": json.dumps({"param1": "test_value_2"})},
                },
            ],
        )

        # Process the message
        tool_responses = await self.integration.process_assistant_message(message)

        # Verify the tool responses
        self.assertEqual(len(tool_responses), 2)
        self.assertEqual(tool_responses[0]["role"], "tool")
        self.assertEqual(tool_responses[0]["tool_call_id"], "test_id_1")
        self.assertEqual(tool_responses[0]["name"], "test_tool")
        self.assertEqual(tool_responses[1]["role"], "tool")
        self.assertEqual(tool_responses[1]["tool_call_id"], "test_id_2")
        self.assertEqual(tool_responses[1]["name"], "test_tool")

        # Verify the tool service was called twice
        self.assertEqual(self.tool_service.execute_tool.call_count, 2)

    @pytest.mark.asyncio
    async def test_process_assistant_message_without_tool_calls(self):
        """Test processing an assistant message without tool calls."""
        # Create a message without tool calls
        message = MCPMessage(role="assistant", content="No tools used")

        # Process the message
        tool_responses = await self.integration.process_assistant_message(message)

        # Verify no tool responses
        self.assertEqual(len(tool_responses), 0)

        # Verify the tool service was not called
        self.tool_service.execute_tool.assert_not_called()

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        # Get tool definitions
        tool_definitions = self.integration.get_tool_definitions()

        # Verify the tool definitions
        self.assertEqual(len(tool_definitions), 1)
        self.assertEqual(tool_definitions[0]["type"], "function")
        self.assertEqual(tool_definitions[0]["function"]["name"], "test_tool")
        self.assertEqual(tool_definitions[0]["function"]["description"], "Test tool")

        # Verify the tool service was called
        self.tool_service.get_available_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_conversation_turn_without_tool_calls(self):
        """Test executing a conversation turn without tool calls."""
        # Set up the client to return a response without tool calls
        assistant_message = MCPMessage(role="assistant", content="Response without tools")
        self.client.process_response.return_value = assistant_message

        # Execute a conversation turn
        response, history = await self.integration.execute_conversation_turn("User message", [])

        # Verify the response
        self.assertEqual(response, "Response without tools")

        # Verify the history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "User message")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Response without tools")

        # Verify the client was called
        self.client.send_request.assert_called_once()
        self.client.process_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_conversation_turn_with_tool_calls(self):
        """Test executing a conversation turn with tool calls."""
        # Set up the client to return a response with tool calls
        first_message = MCPMessage(
            role="assistant",
            content="Using tools",
            tool_calls=[
                {
                    "id": "test_id",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": json.dumps({"param1": "test_value"})},
                }
            ],
        )
        follow_up_message = MCPMessage(role="assistant", content="Response after tool calls")
        self.client.process_response.side_effect = [first_message, follow_up_message]

        # Execute a conversation turn
        response, history = await self.integration.execute_conversation_turn("User message", [])

        # Verify the response
        self.assertEqual(response, "Response after tool calls")

        # Verify the history
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "User message")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[2]["role"], "tool")
        self.assertEqual(history[3]["role"], "assistant")
        self.assertEqual(history[3]["content"], "Response after tool calls")

        # Verify the client was called twice
        self.assertEqual(self.client.send_request.call_count, 2)
        self.assertEqual(self.client.process_response.call_count, 2)

        # Verify the tool service was called
        self.tool_service.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_conversation_turn_with_history(self):
        """Test executing a conversation turn with existing history."""
        # Set up the client to return a response without tool calls
        assistant_message = MCPMessage(role="assistant", content="Response without tools")
        self.client.process_response.return_value = assistant_message

        # Create history
        history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]

        # Execute a conversation turn
        response, new_history = await self.integration.execute_conversation_turn("User message", history)

        # Verify the response
        self.assertEqual(response, "Response without tools")

        # Verify the history
        self.assertEqual(len(new_history), 4)
        self.assertEqual(new_history[0]["role"], "user")
        self.assertEqual(new_history[0]["content"], "Previous message")
        self.assertEqual(new_history[1]["role"], "assistant")
        self.assertEqual(new_history[1]["content"], "Previous response")
        self.assertEqual(new_history[2]["role"], "user")
        self.assertEqual(new_history[2]["content"], "User message")
        self.assertEqual(new_history[3]["role"], "assistant")
        self.assertEqual(new_history[3]["content"], "Response without tools")

        # Verify the client was called
        self.client.send_request.assert_called_once()
        self.client.process_response.assert_called_once()
