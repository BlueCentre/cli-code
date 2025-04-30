"""
Tests for the MCP integration utilities.
"""

import json
import logging
import subprocess
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

from src.cli_code.mcp.client import MCPClient, MCPMessage, MCPToolCall
from src.cli_code.mcp.integrations import MCPToolIntegration
from src.cli_code.mcp.tools.models import ToolResult
from src.cli_code.mcp.tools.service import ToolService


# Helper to create MCPToolCall instance easily
def create_mcp_tool_call(name="test_tool", args=None, call_id="call_123"):
    if args is None:
        args = {}
    return MCPToolCall(
        id=call_id,
        type="function",
        function={"name": name, "arguments": json.dumps(args) if isinstance(args, dict) else args},
    )


@pytest.mark.asyncio
class TestMCPToolIntegration(unittest.TestCase):
    """Tests for the MCPToolIntegration class."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MagicMock(spec=MCPClient)
        self.tool_service = MagicMock(spec=ToolService)

        self.logger_patcher = patch("src.cli_code.mcp.integrations.logger")
        self.mock_logger = self.logger_patcher.start()

        self.integration = MCPToolIntegration(self.client, self.tool_service)

    def tearDown(self):
        self.logger_patcher.stop()

    async def test_handle_tool_call_success_dict_args(self):
        tool_call = create_mcp_tool_call(name="tool1", args={"p": 1}, call_id="t1")
        mock_tool_result = ToolResult(tool_name="tool1", parameters={"p": 1}, result={"data": "abc"}, success=True)
        self.tool_service.execute_tool = AsyncMock(return_value=mock_tool_result)

        response_msg = await self.integration.handle_tool_call(tool_call)

        self.tool_service.execute_tool.assert_awaited_once_with("tool1", {"p": 1})
        self.assertEqual(response_msg["role"], "tool")
        self.assertEqual(response_msg["tool_call_id"], "t1")
        self.assertEqual(response_msg["name"], "tool1")
        self.assertEqual(response_msg["content"], json.dumps({"data": "abc"}))
        self.mock_logger.info.assert_called_once_with("Executing tool: tool1 with args: {'p': 1}")

    async def test_handle_tool_call_success_str_args(self):
        tool_call = create_mcp_tool_call(name="tool_str", args="string arg", call_id="t_str")
        mock_tool_result = ToolResult(tool_name="tool_str", parameters="string arg", result=123, success=True)
        self.tool_service.execute_tool = AsyncMock(return_value=mock_tool_result)

        response_msg = await self.integration.handle_tool_call(tool_call)

        self.tool_service.execute_tool.assert_awaited_once_with("tool_str", "string arg")
        self.assertEqual(response_msg["role"], "tool")
        self.assertEqual(response_msg["tool_call_id"], "t_str")
        self.assertEqual(response_msg["name"], "tool_str")
        self.assertEqual(response_msg["content"], json.dumps(123))
        self.mock_logger.info.assert_called_once_with("Executing tool: tool_str with args: string arg")

    async def test_handle_tool_call_invalid_json_args(self):
        invalid_json_str = "{invalid json"
        tool_call = MCPToolCall(
            id="t_bad_json", type="function", function={"name": "bad_json", "arguments": invalid_json_str}
        )
        mock_tool_result = ToolResult(tool_name="bad_json", parameters={}, result="fallback", success=True)
        self.tool_service.execute_tool = AsyncMock(return_value=mock_tool_result)

        response_msg = await self.integration.handle_tool_call(tool_call)

        self.tool_service.execute_tool.assert_awaited_once_with("bad_json", {})
        self.assertEqual(response_msg["content"], json.dumps("fallback"))
        self.mock_logger.error.assert_called_once_with(f"Invalid arguments JSON: {invalid_json_str}")
        self.mock_logger.info.assert_called_once_with("Executing tool: bad_json with args: {}")

    async def test_handle_tool_call_execution_failure(self):
        tool_call = create_mcp_tool_call(name="fail_tool", args={"a": 1}, call_id="t_fail")
        mock_tool_result = ToolResult(
            tool_name="fail_tool", parameters={"a": 1}, result=None, success=False, error="Execution failed"
        )
        self.tool_service.execute_tool = AsyncMock(return_value=mock_tool_result)

        response_msg = await self.integration.handle_tool_call(tool_call)

        self.tool_service.execute_tool.assert_awaited_once_with("fail_tool", {"a": 1})
        self.assertEqual(response_msg["role"], "tool")
        self.assertEqual(response_msg["tool_call_id"], "t_fail")
        self.assertEqual(response_msg["name"], "fail_tool")
        self.assertEqual(response_msg["content"], json.dumps(None))
        self.mock_logger.info.assert_called_once_with("Executing tool: fail_tool with args: {'a': 1}")

    async def test_process_assistant_message_no_calls(self):
        message = MCPMessage(role="assistant", content="No tools needed.", tool_calls=None)
        responses = await self.integration.process_assistant_message(message)
        self.assertEqual(responses, [])

    async def test_process_assistant_message_single_call(self):
        tool_call_dict = {"id": "c1", "type": "function", "function": {"name": "t1", "arguments": '{"p": 1}'}}
        message = MCPMessage(role="assistant", content=None, tool_calls=[tool_call_dict])

        mock_tool_result = ToolResult(tool_name="t1", parameters={"p": 1}, result="Res1", success=True)
        self.tool_service.execute_tool = AsyncMock(return_value=mock_tool_result)

        responses = await self.integration.process_assistant_message(message)

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["role"], "tool")
        self.assertEqual(responses[0]["tool_call_id"], "c1")
        self.assertEqual(responses[0]["name"], "t1")
        self.assertEqual(responses[0]["content"], json.dumps("Res1"))
        self.tool_service.execute_tool.assert_awaited_once_with("t1", {"p": 1})

    async def test_process_assistant_message_multiple_calls(self):
        tool_call_dict1 = {"id": "c1", "type": "function", "function": {"name": "t1", "arguments": '{"p": 1}'}}
        tool_call_dict2 = {"id": "c2", "type": "function", "function": {"name": "t2", "arguments": "{}"}}
        message = MCPMessage(role="assistant", content=None, tool_calls=[tool_call_dict1, tool_call_dict2])

        results_map = {
            ("t1", '{"p": 1}'): ToolResult(tool_name="t1", parameters={"p": 1}, result="Res1", success=True),
            ("t2", "{}"): ToolResult(tool_name="t2", parameters={}, result="Res2", success=True),
        }

        async def mock_execute(name, params):
            # Use json.dumps on params before dictionary lookup
            key = (name, json.dumps(params))
            return results_map[key]

        self.tool_service.execute_tool = AsyncMock(side_effect=mock_execute)

        responses = await self.integration.process_assistant_message(message)

        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]["tool_call_id"], "c1")
        self.assertEqual(responses[0]["content"], json.dumps("Res1"))
        self.assertEqual(responses[1]["tool_call_id"], "c2")
        self.assertEqual(responses[1]["content"], json.dumps("Res2"))

        self.assertEqual(self.tool_service.execute_tool.await_count, 2)
        self.tool_service.execute_tool.assert_any_await("t1", {"p": 1})
        self.tool_service.execute_tool.assert_any_await("t2", {})

    def test_get_tool_definitions(self):
        mock_schemas = {
            "tool1": {
                "description": "Desc 1",
                "parameters": {"type": "object", "properties": {"p1": {"type": "string"}}},
            },
            "tool2": {
                "description": "Desc 2",
                "parameters": {"type": "object", "properties": {"p2": {"type": "integer"}}},
            },
        }
        self.tool_service.get_available_tools = MagicMock(return_value=mock_schemas)

        definitions = self.integration.get_tool_definitions()

        self.assertEqual(len(definitions), 2)
        expected_defs = [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "Desc 1",
                    "parameters": {"type": "object", "properties": {"p1": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool2",
                    "description": "Desc 2",
                    "parameters": {"type": "object", "properties": {"p2": {"type": "integer"}}},
                },
            },
        ]
        self.assertCountEqual(definitions, expected_defs)
        self.tool_service.get_available_tools.assert_called_once()

    def test_get_tool_definitions_empty(self):
        self.tool_service.get_available_tools = MagicMock(return_value={})
        definitions = self.integration.get_tool_definitions()
        self.assertEqual(definitions, [])

    async def test_execute_turn_no_tools(self):
        self.tool_service.get_available_tools = MagicMock(return_value={})

        mock_assistant_response = {"role": "assistant", "content": "Final answer"}
        self.client.send_request = AsyncMock(return_value={"some_raw_response": True})
        self.client.process_response = MagicMock(return_value=MCPMessage.from_dict(mock_assistant_response))

        user_message = "User question?"
        history = []

        response_content, final_history = await self.integration.execute_conversation_turn(user_message, history)

        self.assertEqual(response_content, "Final answer")
        self.assertEqual(len(final_history), 2)
        self.assertEqual(final_history[0], {"role": "user", "content": "User question?"})
        self.assertEqual(final_history[1], mock_assistant_response)
        self.client.send_request.assert_awaited_once_with(messages=final_history[0:1], tools=[])
        self.client.process_response.assert_called_once_with({"some_raw_response": True})

    async def test_execute_turn_with_tool_call(self):
        mock_schemas = {"calc": {"description": "Calculator", "parameters": {}}}
        self.tool_service.get_available_tools = MagicMock(return_value=mock_schemas)
        formatted_tools = self.integration.get_tool_definitions()

        tool_call_dict = {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": '{"op": "add"}'}}
        first_assistant_msg = MCPMessage(role="assistant", content=None, tool_calls=[tool_call_dict])
        final_assistant_msg = MCPMessage(role="assistant", content="The answer is 3")

        # process_response needs to return MCPMessage instances
        self.client.process_response.side_effect = [first_assistant_msg, final_assistant_msg]

        mock_tool_result = ToolResult(tool_name="calc", parameters={"op": "add"}, result=3, success=True)
        self.tool_service.execute_tool = AsyncMock(return_value=mock_tool_result)

        self.client.send_request = AsyncMock()

        user_message = "What is 1+2?"
        history = []

        response_content, final_history = await self.integration.execute_conversation_turn(user_message, history)

        self.assertEqual(response_content, "The answer is 3")
        self.assertEqual(len(final_history), 4)
        self.assertEqual(final_history[0], {"role": "user", "content": "What is 1+2?"})
        self.assertEqual(final_history[1], first_assistant_msg.to_dict())
        self.assertEqual(final_history[2]["role"], "tool")
        self.assertEqual(final_history[2]["tool_call_id"], "c1")
        self.assertEqual(final_history[2]["name"], "calc")
        self.assertEqual(final_history[2]["content"], json.dumps(3))
        self.assertEqual(final_history[3], final_assistant_msg.to_dict())

        self.tool_service.execute_tool.assert_awaited_once_with("calc", {"op": "add"})

        self.assertEqual(self.client.send_request.await_count, 2)
        self.client.send_request.assert_any_await(messages=final_history[0:1], tools=formatted_tools)
        self.client.send_request.assert_any_await(messages=final_history[0:3], tools=formatted_tools)

    async def test_execute_turn_multiple_tool_calls(self):
        """Test a turn involving multiple sequential tool calls."""
        mock_schemas = {"tool_a": {}, "tool_b": {}}
        self.tool_service.get_available_tools = MagicMock(return_value=mock_schemas)
        formatted_tools = self.integration.get_tool_definitions()

        # 1. User asks question
        user_message = "Run tool A then B"
        history = []

        # 2. Assistant asks for tool_a
        tool_call_a = {"id": "ca", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}}
        assistant_msg_1 = MCPMessage(role="assistant", content=None, tool_calls=[tool_call_a])
        # 3. Tool A runs successfully
        tool_result_a = ToolResult(tool_name="tool_a", parameters={}, result="Result A", success=True)
        # 4. Assistant asks for tool_b based on Tool A result
        tool_call_b = {"id": "cb", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}}
        assistant_msg_2 = MCPMessage(role="assistant", content=None, tool_calls=[tool_call_b])
        # 5. Tool B runs successfully
        tool_result_b = ToolResult(tool_name="tool_b", parameters={}, result="Result B", success=True)
        # 6. Final assistant response
        final_assistant_msg = MCPMessage(role="assistant", content="Finished with Result B")

        # Mock client responses
        self.client.send_request = AsyncMock()
        self.client.process_response.side_effect = [assistant_msg_1, assistant_msg_2, final_assistant_msg]

        # Mock tool service responses
        self.tool_service.execute_tool = AsyncMock(side_effect=[tool_result_a, tool_result_b])

        # Execute turn
        response_content, final_history = await self.integration.execute_conversation_turn(user_message, history)

        # Assertions
        self.assertEqual(response_content, "Finished with Result B")
        self.assertEqual(len(final_history), 6)  # user, assist1+call, tool_a, assist2+call, tool_b, assist_final
        self.assertEqual(final_history[0]["role"], "user")
        self.assertEqual(final_history[1], assistant_msg_1.to_dict())
        self.assertEqual(final_history[2]["role"], "tool")
        self.assertEqual(final_history[2]["name"], "tool_a")
        self.assertEqual(final_history[2]["content"], json.dumps("Result A"))
        self.assertEqual(final_history[3], assistant_msg_2.to_dict())
        self.assertEqual(final_history[4]["role"], "tool")
        self.assertEqual(final_history[4]["name"], "tool_b")
        self.assertEqual(final_history[4]["content"], json.dumps("Result B"))
        self.assertEqual(final_history[5], final_assistant_msg.to_dict())

        self.assertEqual(self.client.send_request.await_count, 3)
        self.client.send_request.assert_any_await(messages=final_history[0:1], tools=formatted_tools)
        self.client.send_request.assert_any_await(messages=final_history[0:3], tools=formatted_tools)
        self.client.send_request.assert_any_await(messages=final_history[0:5], tools=formatted_tools)

        self.assertEqual(self.tool_service.execute_tool.await_count, 2)
        self.tool_service.execute_tool.assert_any_await("tool_a", {})
        self.tool_service.execute_tool.assert_any_await("tool_b", {})

    async def test_execute_turn_tool_call_fails(self):
        """Test a turn where the requested tool call fails."""
        mock_schemas = {"fail_tool": {}}
        self.tool_service.get_available_tools = MagicMock(return_value=mock_schemas)
        formatted_tools = self.integration.get_tool_definitions()

        # 1. User asks question
        user_message = "Run the failing tool"
        history = []

        # 2. Assistant asks for fail_tool
        tool_call_dict = {"id": "cf", "type": "function", "function": {"name": "fail_tool", "arguments": "{}"}}
        assistant_msg_1 = MCPMessage(role="assistant", content=None, tool_calls=[tool_call_dict])
        # 3. Tool execution fails
        tool_result_fail = ToolResult(
            tool_name="fail_tool", parameters={}, result=None, success=False, error="It broke"
        )
        # 4. Final assistant response acknowledging failure
        final_assistant_msg = MCPMessage(role="assistant", content="Sorry, the tool failed.")

        # Mock client responses
        self.client.send_request = AsyncMock()
        self.client.process_response.side_effect = [assistant_msg_1, final_assistant_msg]

        # Mock tool service response
        self.tool_service.execute_tool = AsyncMock(return_value=tool_result_fail)

        # Execute turn
        response_content, final_history = await self.integration.execute_conversation_turn(user_message, history)

        # Assertions
        self.assertEqual(response_content, "Sorry, the tool failed.")
        self.assertEqual(len(final_history), 4)  # user, assist1+call, tool_fail, assist_final
        self.assertEqual(final_history[0]["role"], "user")
        self.assertEqual(final_history[1], assistant_msg_1.to_dict())
        self.assertEqual(final_history[2]["role"], "tool")
        self.assertEqual(final_history[2]["name"], "fail_tool")
        self.assertEqual(final_history[2]["content"], json.dumps(None))  # Failed result is None
        self.assertEqual(final_history[3], final_assistant_msg.to_dict())

        self.assertEqual(self.client.send_request.await_count, 2)
        self.client.send_request.assert_any_await(messages=final_history[0:1], tools=formatted_tools)
        self.client.send_request.assert_any_await(
            messages=final_history[0:3], tools=formatted_tools
        )  # Sends failure back to model

        self.tool_service.execute_tool.assert_awaited_once_with("fail_tool", {})

    async def test_execute_turn_client_request_fails(self):
        """Test a turn where the initial client request fails."""
        self.tool_service.get_available_tools = MagicMock(return_value={})

        # Mock client send_request to raise an exception
        error_message = "Network Error"
        self.client.send_request = AsyncMock(side_effect=Exception(error_message))
        self.client.process_response = MagicMock()  # Should not be called

        user_message = "User question?"
        history = []

        # Execute turn and expect an exception
        with self.assertRaises(Exception) as cm:
            await self.integration.execute_conversation_turn(user_message, history)

        self.assertEqual(str(cm.exception), error_message)
        self.client.send_request.assert_awaited_once_with(
            messages=[{"role": "user", "content": user_message}], tools=[]
        )
        self.client.process_response.assert_not_called()
        self.tool_service.execute_tool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
