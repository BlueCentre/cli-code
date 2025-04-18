"""
Integration utilities for MCP protocol.

This module provides utilities for integrating various components with the MCP protocol.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.cli_code.mcp.client import MCPClient, MCPMessage, MCPToolCall
from src.cli_code.mcp.tools.service import ToolService

logger = logging.getLogger(__name__)


class MCPToolIntegration:
    """Integration for MCP tools."""

    def __init__(self, client: MCPClient, tool_service: ToolService):
        """
        Initialize the MCP tool integration.

        Args:
            client: The MCP client to use
            tool_service: The tool service to use
        """
        self.client = client
        self.tool_service = tool_service

    async def handle_tool_call(self, tool_call: MCPToolCall) -> Dict[str, Any]:
        """
        Handle a tool call from the MCP protocol.

        Args:
            tool_call: The tool call to handle

        Returns:
            The result of the tool execution
        """
        # Extract tool name and parameters
        function_name = tool_call.function.get("name", "")
        function_args = tool_call.function.get("arguments", "{}")

        # Parse arguments if they're a string
        if isinstance(function_args, str):
            try:
                parameters = json.loads(function_args)
            except json.JSONDecodeError:
                logger.error(f"Invalid arguments JSON: {function_args}")
                parameters = {}
        else:
            parameters = function_args

        # Execute the tool
        logger.info(f"Executing tool: {function_name} with args: {parameters}")
        result = await self.tool_service.execute_tool(function_name, parameters)

        # Create a tool message with the result
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(result.get("result", {})),
        }

    async def process_assistant_message(self, message: MCPMessage) -> List[Dict[str, Any]]:
        """
        Process an assistant message and handle any tool calls.

        Args:
            message: The assistant message to process

        Returns:
            A list of tool response messages
        """
        if not message.tool_calls:
            return []

        tool_responses = []

        for tool_call_data in message.tool_calls:
            tool_call = MCPToolCall.from_dict(tool_call_data)
            tool_response = await self.handle_tool_call(tool_call)
            tool_responses.append(tool_response)

        return tool_responses

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for the MCP protocol.

        Returns:
            A list of tool definitions
        """
        schemas = self.tool_service.get_available_tools()

        # Convert schemas to the format expected by the MCP protocol
        tools = []
        for name, schema in schemas.items():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {}),
                    },
                }
            )

        return tools

    async def execute_conversation_turn(
        self, user_message: str, conversation_history: List[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute a full conversation turn with the MCP protocol.

        Args:
            user_message: The user's message
            conversation_history: The conversation history

        Returns:
            A tuple of (assistant_response, updated_conversation_history)
        """
        if conversation_history is None:
            conversation_history = []

        # Add the user message to the conversation history
        conversation_history.append({"role": "user", "content": user_message})

        # Get tool definitions
        tools = self.get_tool_definitions()

        # Send the request to the MCP server
        response = await self.client.send_request(messages=conversation_history, tools=tools)

        # Process the response
        assistant_message = self.client.process_response(response)

        # Add the assistant message to the conversation history
        conversation_history.append(assistant_message.to_dict())

        # Handle tool calls if present
        if assistant_message.tool_calls:
            tool_responses = await self.process_assistant_message(assistant_message)

            # Add tool responses to the conversation history
            conversation_history.extend(tool_responses)

            # Send a follow-up request to get the assistant's response to the tool results
            follow_up_response = await self.client.send_request(messages=conversation_history, tools=tools)

            # Process the follow-up response
            follow_up_message = self.client.process_response(follow_up_response)

            # Add the follow-up message to the conversation history
            conversation_history.append(follow_up_message.to_dict())

            # Return the follow-up response
            return follow_up_message.content, conversation_history

        # Return the assistant's response
        return assistant_message.content, conversation_history
