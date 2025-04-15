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
            if parts and isinstance(parts[0], str):
                content = parts[0]
            elif parts and isinstance(parts[0], dict) and "text" in parts[0]:
                content = parts[0]["text"]
            
            # Check for function calls in the parts
            tool_calls = None
            if parts and isinstance(parts[0], dict) and "function_call" in parts[0]:
                function_call = parts[0]["function_call"]
                tool_calls = [{
                    "id": f"call_{function_call['name']}_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": function_call.get("args", {})
                    }
                }]
            
            # Create MCP message
            mcp_message = MCPMessage(
                role=mcp_role,
                content=content,
                tool_calls=tool_calls
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
            for tool_call in message.tool_calls:
                if tool_call.get("type") == "function":
                    function_call = {
                        "name": tool_call["function"]["name"],
                        "args": tool_call["function"].get("arguments", {})
                    }
                    parts.append({"function_call": function_call})
        
        # Create Gemini message
        gemini_message = {
            "role": gemini_role,
            "parts": parts
        }
        
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
        role_map = {
            "user": "user",
            "model": "assistant",
            "system": "system",
            "function": "tool"
        }
        return role_map.get(role.lower(), "user")
    
    def _map_mcp_role_to_gemini(self, role: str) -> str:
        """Map MCP role to Gemini role."""
        role_map = {
            "user": "user",
            "assistant": "model",
            "system": "system",
            "tool": "function"
        }
        return role_map.get(role.lower(), "user") 