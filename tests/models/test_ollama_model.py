"""
Tests specifically for the OllamaModel class to improve code coverage.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

# Check if running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"

# Handle imports
try:
    from rich.console import Console

    from cli_code.models.ollama import OllamaModel

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    OllamaModel = MagicMock
    Console = MagicMock

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestOllamaModel:
    """Test suite for OllamaModel class, focusing on previously uncovered methods."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock OpenAI module before initialization
        self.openai_patch = patch("cli_code.models.ollama.OpenAI")
        self.mock_openai = self.openai_patch.start()

        # Mock the OpenAI client instance
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client

        # Mock console
        self.mock_console = MagicMock(spec=Console)

        # Mock os.path.isdir and os.path.isfile
        self.isdir_patch = patch("os.path.isdir")
        self.isfile_patch = patch("os.path.isfile")
        self.mock_isdir = self.isdir_patch.start()
        self.mock_isfile = self.isfile_patch.start()

        # Mock glob
        self.glob_patch = patch("glob.glob")
        self.mock_glob = self.glob_patch.start()

        # Mock open
        self.open_patch = patch("builtins.open", mock_open(read_data="# Test content"))
        self.mock_open = self.open_patch.start()

        # Mock get_tool
        self.get_tool_patch = patch("cli_code.models.ollama.get_tool")
        self.mock_get_tool = self.get_tool_patch.start()

        # Default tool mock
        self.mock_tool = MagicMock()
        self.mock_tool.execute.return_value = "ls output"
        self.mock_get_tool.return_value = self.mock_tool

    def teardown_method(self):
        """Tear down test fixtures."""
        self.openai_patch.stop()
        self.isdir_patch.stop()
        self.isfile_patch.stop()
        self.glob_patch.stop()
        self.open_patch.stop()
        self.get_tool_patch.stop()

    def test_init(self):
        """Test initialization of OllamaModel."""
        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

        # Check if OpenAI client was initialized correctly
        self.mock_openai.assert_called_once_with(base_url="http://localhost:11434", api_key="ollama")

        # Check model attributes
        assert model.api_url == "http://localhost:11434"
        assert model.model_name == "llama3"

        # Check history initialization
        assert len(model.history) == 1
        assert model.history[0]["role"] == "system"

    def test_get_initial_context_with_rules_dir(self):
        """Test getting initial context from .rules directory."""
        # Set up mocks
        self.mock_isdir.return_value = True
        self.mock_glob.return_value = [".rules/context.md", ".rules/tools.md"]

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        context = model._get_initial_context()

        # Verify directory check
        self.mock_isdir.assert_called_with(".rules")

        # Verify glob search
        self.mock_glob.assert_called_with(".rules/*.md")

        # Verify files were read
        assert self.mock_open.call_count == 2

        # Check result content
        assert "Project rules and guidelines:" in context
        assert "# Content from" in context

    def test_get_initial_context_with_readme(self):
        """Test getting initial context from README.md when no .rules directory."""
        # Set up mocks
        self.mock_isdir.return_value = False
        self.mock_isfile.return_value = True

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        context = model._get_initial_context()

        # Verify README check
        self.mock_isfile.assert_called_with("README.md")

        # Verify file reading
        self.mock_open.assert_called_once_with("README.md", "r", encoding="utf-8", errors="ignore")

        # Check result content
        assert "Project README:" in context

    def test_get_initial_context_with_ls_fallback(self):
        """Test getting initial context via ls when no .rules or README."""
        # Set up mocks
        self.mock_isdir.return_value = False
        self.mock_isfile.return_value = False

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        context = model._get_initial_context()

        # Verify tool was used
        self.mock_get_tool.assert_called_with("ls")
        self.mock_tool.execute.assert_called_once()

        # Check result content
        assert "Current directory contents" in context
        assert "ls output" in context

    def test_prepare_openai_tools(self):
        """Test preparation of tools in OpenAI function format."""
        # Create a mock for AVAILABLE_TOOLS at the correct import path
        # Also mock the get_function_declaration method for a sample tool
        mock_tool_instance = MagicMock()

        # Mock the FunctionDeclaration and its components
        mock_declaration = MagicMock()
        mock_declaration.name = "test_tool"
        mock_declaration.description = "A test tool"
        mock_declaration.parameters = MagicMock()
        mock_declaration.parameters._pb = MagicMock()  # Mock the underlying protobuf object

        # Mock MessageToDict conversion result
        mock_params_dict = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "A string parameter"},
                "param2": {"type": "integer", "description": "An integer parameter"},
            },
            "required": ["param1"],
        }

        mock_tool_instance.get_function_declaration.return_value = mock_declaration

        # Patch the correct AVAILABLE_TOOLS path used by the OllamaModel class
        # Patch MessageToDict as well
        with (
            patch("cli_code.models.ollama.AVAILABLE_TOOLS", {"test_tool": mock_tool_instance}),
            patch("cli_code.models.ollama.MessageToDict", return_value=mock_params_dict) as mock_msg_to_dict,
        ):
            model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
            tools = model._prepare_openai_tools()

            # Verify tools format
            assert len(tools) == 1
            expected_schema = {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": mock_params_dict,
                },
            }
            assert tools[0] == expected_schema
            mock_tool_instance.get_function_declaration.assert_called_once()
            mock_msg_to_dict.assert_called_once_with(mock_declaration.parameters._pb)

    def test_manage_ollama_context(self):
        """Test context management for Ollama models."""
        # Mock the count_tokens function (not strictly needed for this test now, but harmless)
        with patch("cli_code.models.ollama.count_tokens", return_value=10) as mock_count_tokens:
            # Set a low iteration limit for the test
            with patch("cli_code.models.ollama.MAX_OLLAMA_ITERATIONS", 5):
                model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

                # Store the system message
                system_message = model.history[0]

                # Prevent _manage_ollama_context from running inside add_to_history during setup
                # Apply the patch *before* the loop that calls add_to_history
                with patch.object(model, "_manage_ollama_context") as mock_manage_inside_add:
                    # Add messages to exceed the limit (MAX_OLLAMA_ITERATIONS = 5)
                    # 1 system + 10 messages = 11 total
                    messages_to_add = 10
                    for i in range(messages_to_add):
                        role = "user" if i % 2 == 0 else "assistant"
                        model.add_to_history({"role": role, "content": f"Message {i}"})

                    # Verify it wasn't called during the loop because add_to_history used the mocked version
                    mock_manage_inside_add.assert_called()  # It should have been called 10 times by add_to_history
                    assert mock_manage_inside_add.call_count == messages_to_add

                # At this point, the *real* _manage_ollama_context has NOT run yet.
                initial_length = len(model.history)  # Should be 11
                assert initial_length == 11

                # Now, call the *real* context management method explicitly using the original method
                # We need to access the original method before it was patched
                original_manage_context = model.__class__._manage_ollama_context
                original_manage_context(model)  # Call it on the instance

                final_length = len(model.history)

                # Verify truncation occurred based on MAX_OLLAMA_ITERATIONS
                # Expected length = 1 (system) + (MAX_OLLAMA_ITERATIONS - 1) = 1 + 4 = 5
                expected_length = 5
                assert final_length == expected_length
                # Verify system message is preserved
                assert model.history[0] == system_message
                # Verify the *correct* last messages are preserved
                # History: [S, M0, M1, M2, M3, M4, M5, M6, M7, M8, M9]
                # Keep index 0 (S) and last 4 (M6, M7, M8, M9)
                assert model.history[1]["content"] == "Message 6"  # Check second element
                assert model.history[-1]["content"] == "Message 9"  # Check last element

    def test_add_to_history(self):
        """Test adding messages to history."""
        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")

        # Clear existing history
        model.history = []

        # Add a message
        message = {"role": "user", "content": "Test message"}
        model.add_to_history(message)

        # Verify message was added
        assert len(model.history) == 1
        assert model.history[0] == message

    def test_clear_history(self):
        """Test clearing history."""
        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        system_message = model.history[0]  # Get the initial system message

        # Add some messages
        model.add_to_history({"role": "user", "content": "Test message"})

        # Clear history
        model.clear_history()

        # Verify history was cleared except for the system message
        assert len(model.history) == 1
        assert model.history[0] == system_message

    def test_list_models(self):
        """Test listing available models."""
        # Create MagicMock objects for model data
        mock_model1 = MagicMock()
        mock_model1.id = "llama3:latest"
        mock_model1.name = "llama3"
        mock_model1.modified_at = "2023-10-26T17:28:18.419424546Z"
        mock_model1.size = 4697386093
        mock_model1.details = {"format": "gguf"}

        mock_model2 = MagicMock()
        mock_model2.id = "mistral:latest"
        mock_model2.name = "mistral"
        mock_model2.modified_at = "2023-10-27T17:28:18.419424546Z"
        mock_model2.size = 4109865159
        mock_model2.details = {"format": "gguf"}

        mock_models_data = [mock_model1, mock_model2]

        # Mock the client's list response object
        mock_response = MagicMock()
        mock_response.data = mock_models_data

        # Set up client mock to return response
        self.mock_client.models.list.return_value = mock_response

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        result = model.list_models()

        # Verify client method called
        self.mock_client.models.list.assert_called_once()

        # Verify result format (should match processed structure)
        expected_result = [
            {
                "id": "llama3:latest",
                "name": "llama3",
                "modified_at": "2023-10-26T17:28:18.419424546Z",
                "size": 4697386093,
                "details": {"format": "gguf"},
            },
            {
                "id": "mistral:latest",
                "name": "mistral",
                "modified_at": "2023-10-27T17:28:18.419424546Z",
                "size": 4109865159,
                "details": {"format": "gguf"},
            },
        ]
        assert result == expected_result

    def test_list_models_no_models(self):
        """Test listing when no models are available."""

    def test_generate_with_function_calls(self):
        """Test generate method with function calls."""
        # 1. Initial API Call Response (Requesting Tool Call)
        mock_tool_call_message = MagicMock()
        mock_tool_call_message.content = None
        mock_tool_calls = [
            MagicMock(id="call_123", function=MagicMock(name="test_tool", arguments='{"param1": "value1"}'))
        ]
        mock_tool_call_message.tool_calls = mock_tool_calls
        # Ensure model_dump returns a dict compatible with history
        mock_tool_call_message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [  # Simulate structure expected in history
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in mock_tool_calls
            ],
        }
        mock_tool_call_response = MagicMock()
        mock_tool_call_response.choices = [MagicMock(message=mock_tool_call_message, finish_reason="tool_calls")]

        # 2. Second API Call Response (After Tool Result)
        mock_final_text_message = MagicMock()
        mock_final_text_message.content = "Final answer after tool use."
        mock_final_text_message.tool_calls = None
        # Ensure model_dump returns a simple dict for the text response
        mock_final_text_message.model_dump.return_value = {
            "role": "assistant",
            "content": mock_final_text_message.content,
        }
        mock_final_text_response = MagicMock()
        mock_final_text_response.choices = [MagicMock(message=mock_final_text_message, finish_reason="stop")]

        # Configure side effect for API calls
        self.mock_client.chat.completions.create.side_effect = [mock_tool_call_response, mock_final_text_response]

        # Mock get_tool to return a tool that executes successfully
        tool_mock = MagicMock()
        tool_mock.execute.return_value = "Tool execution result"
        self.mock_get_tool.return_value = tool_mock

        model = OllamaModel("http://localhost:11434", self.mock_console, "llama3")
        result = model.generate("Test prompt")

        # Verify client method called twice
        assert self.mock_client.chat.completions.create.call_count == 2

        # Verify tool execution was called ONCE
        tool_mock.execute.assert_called_once_with(param1="value1")

        # Verify the final result is the text from the second response
        assert result == "Final answer after tool use."

        # Check history contains the tool request and response
        assert any(msg.get("role") == "assistant" and msg.get("tool_calls") for msg in model.history)
        assert any(msg.get("role") == "tool" and msg.get("content") == "Tool execution result" for msg in model.history)

    # ... test_generate_without_function_calls ...
    # ... test_generate_with_api_error ...
