"""
Gemini model adapter for MCP protocol.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from rich.console import Console

from ...tools import get_tool
from ..client import MCPClient, MCPMessage, MCPToolCall
from ..model_adapter import MCPModelAdapter


class GeminiModelAdapter(MCPModelAdapter):
    """Adapter for Gemini models to use MCP protocol."""

    def __init__(self, model_agent, mcp_client: MCPClient, console: Console):
        """
        Initialize the Gemini model adapter.

        Args:
            model_agent: The Gemini model agent
            mcp_client: The MCP protocol client
            console: The console for output
        """
        super().__init__(model_agent, mcp_client)
        self.console = console
        self.logger = logging.getLogger("gemini_adapter")

    def format_for_mcp(self, prompt: str, history: List[Dict[str, Any]]) -> List[MCPMessage]:
        """
        Format Gemini-specific history for the MCP protocol.

        Args:
            prompt: The user's input prompt
            history: The conversation history in Gemini format

        Returns:
            A list of MCPMessage objects formatted for the MCP protocol
        """
        mcp_messages = []

        # Convert Gemini history format to MCP format
        for entry in history:
            role = entry.get("role", "")
            parts = entry.get("parts", [])

            # Map Gemini roles to MCP roles
            mcp_role = self._map_gemini_role_to_mcp(role)

            # Handle content from parts
            content = None
            tool_calls = None
            tool_call_id = None  # Added for tool response
            name = None  # Added for tool response

            if parts:
                part = parts[0]  # Process the first part mainly

                if isinstance(part, str):
                    content = part
                elif isinstance(part, dict):
                    if "text" in part:
                        content = part["text"]
                    elif "function_call" in part:
                        function_call = part["function_call"]
                        tool_calls = [
                            {
                                "id": f"call_{function_call['name']}_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {"name": function_call["name"], "arguments": function_call.get("args", {})},
                            }
                        ]
                    elif "function_response" in part:
                        # Handle function response parts correctly
                        function_response = part["function_response"]
                        name = function_response.get("name")
                        # Extract content from the response part of function_response
                        response_content = function_response.get("response", {}).get("content")
                        if response_content:
                            content = response_content  # Keep it as string (usually JSON)

            # Create MCP message
            mcp_message = MCPMessage(
                role=mcp_role, content=content, tool_calls=tool_calls, name=name, tool_call_id=tool_call_id
            )
            mcp_messages.append(mcp_message)

        return mcp_messages

    def parse_from_mcp(self, message: MCPMessage) -> Dict[str, Any]:
        """
        Parse an MCP message to Gemini format.

        Args:
            message: The MCP message to parse

        Returns:
            The parsed message in Gemini format
        """
        gemini_role = self._map_mcp_role_to_gemini(message.role)

        # Initialize parts
        parts = []

        # Add content as text part if present
        if message.content:
            parts.append(message.content)

        # Add tool calls as function_call parts if present
        if message.tool_calls:
            for tool_call_dict in message.tool_calls:  # Iterate over the list of dicts
                if tool_call_dict.get("type") == "function":
                    func_data = tool_call_dict.get("function", {})
                    args_str = func_data.get("arguments", "{}")
                    args_dict = {}
                    try:
                        args_dict = json.loads(args_str)  # Parse JSON string to dict
                    except json.JSONDecodeError:
                        self.logger.warning(f"Could not parse JSON arguments for tool call: {args_str}")
                        # Keep args as string or handle error as needed? Gemini might expect dict.
                        # For now, let's pass the potentially problematic string if parsing fails,
                        # but log it. A better approach might be needed.
                        args_dict = args_str  # Fallback to raw string? Risky. Let's try empty dict?

                    function_call = {
                        "name": func_data.get("name"),
                        "args": args_dict,  # Assign the parsed dict
                    }
                    parts.append({"function_call": function_call})

        # Handle tool response (role='tool')
        if message.role == "tool" and message.name:
            # Gemini expects a function_response part for tool role messages
            response_content_dict = {}
            if message.content:
                try:
                    response_content_dict = json.loads(message.content)
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse tool response content as JSON: {message.content}")
                    response_content_dict = {"error": "Invalid JSON content received", "raw_content": message.content}

            function_response_part = {
                "function_response": {
                    "name": message.name,
                    "response": response_content_dict,  # Gemini expects the parsed dict here
                }
            }
            # Replace existing content part with function_response part for tool role
            parts = [function_response_part]

        # Create Gemini message
        gemini_message = {"role": gemini_role, "parts": parts}

        return gemini_message

    def execute_tool(self, tool_call: MCPToolCall) -> Dict[str, Any]:
        """
        Execute a tool using the Gemini model's tool execution mechanism.

        Args:
            tool_call: The tool call to execute

        Returns:
            The result of the tool execution
        """
        function_name = tool_call.function.get("name", "")
        function_args = tool_call.function.get("arguments", "{}")

        # Parse arguments
        if isinstance(function_args, str):
            try:
                args = json.loads(function_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = function_args

        self.logger.info(f"Executing tool: {function_name} with args: {args}")

        # Get the tool implementation
        tool_impl = get_tool(function_name)
        if not tool_impl:
            error_msg = f"Tool not found: {function_name}"
            self.logger.error(error_msg)
            return {"error": error_msg}

        # Execute the tool
        try:
            result = tool_impl(args, self.console)
            self.logger.info(f"Tool execution result: {result}")
            return {"result": result}
        except Exception as e:
            error_msg = f"Error executing tool {function_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

    def send_request(self, prompt: str) -> Optional[str]:
        """
        Send a request to the Gemini model through the MCP protocol.

        Args:
            prompt: The user's input prompt

        Returns:
            The generated text response, or None if an error occurs
        """
        # Get history from the model agent
        history = getattr(self.model_agent, "history", [])

        # Format the messages for MCP
        mcp_messages = self.format_for_mcp(prompt, history)

        try:
            # This would normally send the message to an MCP server
            # For now, we'll use the model agent directly
            response = self.model_agent.generate(prompt)

            # In a real implementation, we would handle tool calls here
            # by parsing the response and executing tools when needed

            return response
        except Exception as e:
            self.logger.error(f"Error sending request: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    def _map_gemini_role_to_mcp(self, role: str) -> str:
        """Map Gemini role to MCP role."""
        role_map = {"user": "user", "model": "assistant", "system": "system", "function": "tool"}
        return role_map.get(role.lower(), "user")

    def _map_mcp_role_to_gemini(self, role: str) -> str:
        """Map MCP role to Gemini role."""
        role_map = {"user": "user", "assistant": "model", "system": "system", "tool": "function"}
        return role_map.get(role.lower(), "user")
