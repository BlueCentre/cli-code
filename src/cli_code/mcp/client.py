"""
MCP (Model Context Provider) client.

This module provides a client for interacting with MCP servers.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
import requests

logger = logging.getLogger(__name__)


class MCPMessage:
    """Represents a message in the MCP protocol."""

    def __init__(
        self,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a message.

        Args:
            role: Role of the message sender (user/assistant/system/tool)
            content: Text content of the message
            tool_calls: List of tool calls made by the assistant
            tool_call_id: ID of the tool call this message responds to
            name: Name of the tool for tool messages
        """
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API requests."""
        message_dict = {"role": self.role}

        if self.content is not None:
            message_dict["content"] = self.content

        if self.tool_calls:
            message_dict["tool_calls"] = self.tool_calls

        if self.tool_call_id:
            message_dict["tool_call_id"] = self.tool_call_id

        if self.name:
            message_dict["name"] = self.name

        return message_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create a message from dictionary data."""
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


class MCPToolCall:
    """Represents a tool call in the MCP protocol."""

    def __init__(
        self,
        id: str,
        type: str,
        function: Dict[str, Any],
    ):
        """
        Initialize a tool call.

        Args:
            id: Unique identifier for the tool call
            type: Type of the tool call (usually "function")
            function: Function details including name and arguments
        """
        self.id = id
        self.type = type
        self.function = function

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool call to dictionary format."""
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPToolCall":
        """Create a tool call from dictionary data."""
        return cls(
            id=data["id"],
            type=data["type"],
            function=data["function"],
        )


class MCPClient:
    """Client for interacting with Model Context Provider servers."""

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        model: str = "default",
    ):
        """
        Initialize the client.

        Args:
            endpoint: URL of the MCP server
            api_key: API key for authentication
            model: Model to use for requests
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)

    def process_response(self, response: Dict[str, Any]) -> MCPMessage:
        """
        Process the response from the MCP server.

        Args:
            response: The response from the MCP server

        Returns:
            Processed message from the response
        """
        # Extract the message from the response
        if "message" in response:
            return MCPMessage.from_dict(response["message"])
        elif "choices" in response and response["choices"]:
            # Handle OpenAI-style responses
            message_data = response["choices"][0]["message"]
            return MCPMessage.from_dict(message_data)
        else:
            # Handle unexpected response format
            self.logger.error(f"Unexpected response format: {response}")
            return MCPMessage(role="assistant", content="Error processing response")

    async def send_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a request to the MCP server.

        Args:
            messages: List of messages in the conversation
            tools: List of available tools
            model: Model to use (overrides the client's default)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            The server's response
        """
        # Prepare the request payload
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Send the request
        self.logger.info(f"Sending request to {self.endpoint}")
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Error from MCP server: {error_text}")
                        raise Exception(f"Error from MCP server: {response.status} {error_text}")
                    
                    result = await response.json()
                    return result
        except aiohttp.ClientError as e:
            self.logger.error(f"Error sending request: {e}")
            raise Exception(f"Error sending request: {e}") from e

    def handle_tool_call(self, tool_call: MCPToolCall) -> Dict[str, Any]:
        """
        Handle a tool call from the assistant.

        Args:
            tool_call: The tool call to handle

        Returns:
            The result of the tool execution

        Raises:
            NotImplementedError: This is a placeholder for tool execution
        """
        # This is where tool execution would be implemented
        # For now, we'll raise an error as this is just a placeholder
        function_name = tool_call.function.get("name", "unknown")
        function_args = tool_call.function.get("arguments", "{}")

        raise NotImplementedError(f"Tool execution not implemented: {function_name}({function_args})")
