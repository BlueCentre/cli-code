"""
Tests specifically for the OllamaModel class targeting advanced scenarios and edge cases
to improve code coverage on complex methods like generate().
"""

import json
import os
import sys
from unittest.mock import ANY, MagicMock, call, mock_open, patch

import pytest

# Check if running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"

# Handle imports
try:
    from rich.console import Console

    from cli_code.models.ollama import MAX_OLLAMA_ITERATIONS, OllamaModel

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    OllamaModel = MagicMock
    Console = MagicMock
    MAX_OLLAMA_ITERATIONS = 5

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestOllamaModelAdvanced:
    """Test suite for OllamaModel class focusing on complex methods and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock OpenAI module
        self.openai_patch = patch("cli_code.models.ollama.OpenAI")
        self.mock_openai = self.openai_patch.start()

        # Mock the OpenAI client instance
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client

        # Mock console
        self.mock_console = MagicMock(spec=Console)

        # Mock tool-related components
        self.get_tool_patch = patch("cli_code.models.ollama.get_tool")
        self.mock_get_tool = self.get_tool_patch.start()

        # Default tool mock
        self.mock_tool = MagicMock()
        self.mock_tool.execute.return_value = "Tool execution result"
        self.mock_get_tool.return_value = self.mock_tool

        # Mock initial context method to avoid complexity
        self.get_initial_context_patch = patch.object(
            OllamaModel, "_get_initial_context", return_value="Initial context"
        )
        self.mock_get_initial_context = self.get_initial_context_patch.start()

        # Set up mock for JSON loads
        self.json_loads_patch = patch("json.loads")
        self.mock_json_loads = self.json_loads_patch.start()

        # Mock questionary for user confirmations
        self.questionary_patch = patch("questionary.confirm")
        self.mock_questionary = self.questionary_patch.start()
        self.mock_questionary_confirm = MagicMock()
        self.mock_questionary.return_value = self.mock_questionary_confirm
        self.mock_questionary_confirm.ask.return_value = True  # Default to confirmed

        # Create model instance
        self.model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

    def teardown_method(self):
        """Tear down test fixtures."""
        self.openai_patch.stop()
        self.get_tool_patch.stop()
        self.get_initial_context_patch.stop()
        self.json_loads_patch.stop()
        self.questionary_patch.stop()

    def test_generate_with_text_response(self):
        """Test generate method with a simple text response."""
        # Mock chat completions response with text
        mock_message = MagicMock()
        mock_message.content = "This is a simple text response."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        self.mock_client.chat.completions.create.return_value = mock_response

        # Call generate
        result = self.model.generate("Tell me something interesting")

        # Verify API was called correctly
        self.mock_client.chat.completions.create.assert_called_once()
        call_kwargs = self.mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama3"

        # Verify result
        assert result == "This is a simple text response."

    def test_generate_with_tool_call(self):
        """Test generate method with a tool call response."""
        # Mock a tool call in the response
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call123"
        mock_tool_call.function.name = "ls"
        mock_tool_call.function.arguments = '{"dir": "."}'

        # Parse the arguments as expected
        self.mock_json_loads.return_value = {"dir": "."}

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "ls", "arguments": '{"dir": "."}'}}],
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Set up initial response
        self.mock_client.chat.completions.create.return_value = mock_response

        # Create a second response for after tool execution
        mock_message2 = MagicMock()
        mock_message2.content = "Tool executed successfully."
        mock_message2.tool_calls = None

        mock_choice2 = MagicMock()
        mock_choice2.message = mock_message2

        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]

        # Set up successive responses
        self.mock_client.chat.completions.create.side_effect = [mock_response, mock_response2]

        # Call generate
        result = self.model.generate("List the files in this directory")

        # Verify tool was called
        self.mock_get_tool.assert_called_with("ls")
        self.mock_tool.execute.assert_called_once()

        assert result == "Tool executed successfully."
        # Example of a more specific assertion
        # assert "Tool executed successfully" in result and "ls" in result

    def test_generate_with_task_complete_tool(self):
        """Test generate method with task_complete tool."""
        # Mock a task_complete tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call123"
        mock_tool_call.function.name = "task_complete"
        mock_tool_call.function.arguments = '{"summary": "Task completed successfully!"}'

        # Parse the arguments as expected
        self.mock_json_loads.return_value = {"summary": "Task completed successfully!"}

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "task_complete", "arguments": '{"summary": "Task completed successfully!"}'},
                }
            ],
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        self.mock_client.chat.completions.create.return_value = mock_response

        # Call generate
        result = self.model.generate("Complete this task")

        # Verify result contains the summary
        assert result == "Task completed successfully!"

    def test_generate_with_sensitive_tool_approved(self):
        """Test generate method with sensitive tool that requires approval."""
        # Mock a sensitive tool call (edit)
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call123"
        mock_tool_call.function.name = "edit"
        mock_tool_call.function.arguments = '{"file_path": "file.txt", "content": "new content"}'

        # Parse the arguments as expected
        self.mock_json_loads.return_value = {"file_path": "file.txt", "content": "new content"}

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "edit", "arguments": '{"file_path": "file.txt", "content": "new content"}'},
                }
            ],
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Set up confirmation to be approved
        self.mock_questionary_confirm.ask.return_value = True

        # Set up initial response
        self.mock_client.chat.completions.create.return_value = mock_response

        # Create a second response for after tool execution
        mock_message2 = MagicMock()
        mock_message2.content = "Edit completed."
        mock_message2.tool_calls = None

        mock_choice2 = MagicMock()
        mock_choice2.message = mock_message2

        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]

        # Set up successive responses
        self.mock_client.chat.completions.create.side_effect = [mock_response, mock_response2]

        # Call generate
        result = self.model.generate("Edit this file")

        # Verify user was asked for confirmation
        self.mock_questionary_confirm.ask.assert_called_once()

        # Verify tool was called after approval
        self.mock_get_tool.assert_called_with("edit")
        self.mock_tool.execute.assert_called_once()

        # Verify result
        assert result == "Edit completed."

    def test_generate_with_sensitive_tool_rejected(self):
        """Test generate method with sensitive tool that is rejected."""
        # Mock a sensitive tool call (edit)
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call123"
        mock_tool_call.function.name = "edit"
        mock_tool_call.function.arguments = '{"file_path": "file.txt", "content": "new content"}'

        # Parse the arguments as expected
        self.mock_json_loads.return_value = {"file_path": "file.txt", "content": "new content"}

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "edit", "arguments": '{"file_path": "file.txt", "content": "new content"}'},
                }
            ],
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Set up confirmation to be rejected
        self.mock_questionary_confirm.ask.return_value = False

        # Set up initial response
        self.mock_client.chat.completions.create.return_value = mock_response

        # Create a second response for after rejection
        mock_message2 = MagicMock()
        mock_message2.content = "I'll find another approach."
        mock_message2.tool_calls = None

        mock_choice2 = MagicMock()
        mock_choice2.message = mock_message2

        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]

        # Set up successive responses
        self.mock_client.chat.completions.create.side_effect = [mock_response, mock_response2]

        # Call generate
        result = self.model.generate("Edit this file")

        # Verify user was asked for confirmation
        self.mock_questionary_confirm.ask.assert_called_once()

        # Verify tool was NOT called after rejection
        self.mock_tool.execute.assert_not_called()

        # Verify result
        assert result == "I'll find another approach."

    def test_generate_with_api_error(self):
        """Test generate method with API error."""
        # Mock API error
        exception_message = "API Connection Failed"
        self.mock_client.chat.completions.create.side_effect = Exception(exception_message)

        # Call generate
        result = self.model.generate("Generate something")

        # Verify error handling
        expected_error_start = "(Error interacting with Ollama:"
        assert result.startswith(expected_error_start), (
            f"Expected result to start with '{expected_error_start}', got '{result}'"
        )
        # Check message includes original exception and ends with ')'
        assert exception_message in result, (
            f"Expected exception message '{exception_message}' to be in result '{result}'"
        )
        assert result.endswith(")")

        # Print to console was called
        self.mock_console.print.assert_called_once()
        # Verify the printed message contains the error
        args, _ = self.mock_console.print.call_args
        # The console print uses different formatting
        assert "Error during Ollama interaction:" in args[0]
        assert exception_message in args[0]

    def test_generate_max_iterations(self):
        """Test generate method with maximum iterations reached."""
        # Mock a tool call that will keep being returned
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call123"
        mock_tool_call.function.name = "ls"
        mock_tool_call.function.arguments = '{"dir": "."}'

        # Parse the arguments as expected
        self.mock_json_loads.return_value = {"dir": "."}

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "ls", "arguments": '{"dir": "."}'}}],
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Always return the same response with a tool call to force iteration
        self.mock_client.chat.completions.create.return_value = mock_response

        # Call generate
        result = self.model.generate("List files recursively")

        # Verify max iterations were handled
        # The loop runs MAX_OLLAMA_ITERATIONS times
        assert self.mock_client.chat.completions.create.call_count == MAX_OLLAMA_ITERATIONS
        # Check the specific error message returned by the function
        expected_return_message = "(Agent reached maximum iterations)"
        assert result == expected_return_message, f"Expected '{expected_return_message}', got '{result}'"
        # Verify console output (No specific error print in this case, only a log warning)
        # self.mock_console.print.assert_called_with(...) # Remove this check

    def test_manage_ollama_context(self):
        """Test context window management for Ollama."""
        # Add many more messages to history to force truncation
        num_messages = 50  # Increase from 30
        for i in range(num_messages):
            self.model.add_to_history({"role": "user", "content": f"Message {i}"})
            self.model.add_to_history({"role": "assistant", "content": f"Response {i}"})

        # Record history length before management (System prompt + 2*num_messages)
        initial_length = 1 + (2 * num_messages)
        # Assert initial length is correct before explicit truncation
        assert len(self.model.history) == 5  # It gets truncated inside add_to_history

        # Call context management explicitly (should have no effect if already truncated)
        self.model._manage_ollama_context()

        # Verify history length is still truncated based on MAX_OLLAMA_ITERATIONS
        expected_length = 5  # Based on default MAX_OLLAMA_ITERATIONS = 5
        assert len(self.model.history) == expected_length

        # Check that the system prompt is still the first message
        assert self.model.history[0]["role"] == "system"
        assert "You are a helpful AI coding assistant" in self.model.history[0]["content"]

    def test_error_handling_for_tool_execution(self):
        """Test error handling during tool execution."""
        # Mock a tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call123"
        mock_tool_call.function.name = "ls"
        mock_tool_call.function.arguments = '{"dir": "."}'

        # Parse the arguments as expected
        self.mock_json_loads.return_value = {"dir": "."}

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "ls", "arguments": '{"dir": "."}'}}],
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Set up initial response
        self.mock_client.chat.completions.create.return_value = mock_response

        # Make tool execution fail
        error_message = "Tool execution failed"
        self.mock_tool.execute.side_effect = Exception(error_message)

        # Create a second response for after tool failure
        mock_message2 = MagicMock()
        mock_message2.content = "I encountered an error."
        mock_message2.tool_calls = None

        mock_choice2 = MagicMock()
        mock_choice2.message = mock_message2

        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]

        # Set up successive responses
        self.mock_client.chat.completions.create.side_effect = [mock_response, mock_response2]

        # Call generate
        result = self.model.generate("List the files")

        # Verify error was handled gracefully with specific assertions
        assert result == "I encountered an error."
        # Verify that error details were added to history
        error_found = False
        for message in self.model.history:
            if message.get("role") == "tool" and message.get("name") == "ls":
                assert "error" in message.get("content", "").lower()
                assert error_message in message.get("content", "")
                error_found = True
        assert error_found, "Error message not found in history"

    def test_generate_direct_response(self):
        """Test generate returning a direct text response without tool calls."""
