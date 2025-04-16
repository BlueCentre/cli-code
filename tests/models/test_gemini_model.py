"""
Tests specifically for the GeminiModel class to improve code coverage.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, mock_open, patch

import pytest

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"

# Handle imports
try:
    import google.generativeai as genai
    from google.generativeai.types import FunctionDeclaration, Tool
    from google.generativeai.types.content_types import Content, Part
    from google.generativeai.types.generation_types import Candidate, FinishReason
    from rich.console import Console

    from src.cli_code.mcp.tools.registry import ToolRegistry
    from src.cli_code.models.gemini import MAX_HISTORY_TURNS, GeminiModel
    from src.cli_code.schemas import ChatMessage, Role
    from src.cli_code.tools import AVAILABLE_TOOLS
    from src.cli_code.tools.base import BaseTool

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    GeminiModel = MagicMock
    Console = MagicMock
    genai = MagicMock

# Set up conditional skipping
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
class TestGeminiModel:
    """Test suite for GeminiModel class, focusing on previously uncovered methods."""

    # Mock Tool class for testing - MOVED HERE
    class MockToolClass:
        def __init__(self, name="mock_tool"):
            self.name = name

        def get_function_declaration(self):
            # Return a mock FunctionDeclaration or similar structure expected by GeminiModel
            # This needs to match what _create_tool_definitions expects
            from google.generativeai.types import FunctionDeclaration  # Added import here as workaround

            mock_declaration = MagicMock(spec=FunctionDeclaration)
            mock_declaration.name = self.name
            mock_declaration.description = f"Description for {self.name}"
            # Add mock parameters if needed by the system prompt generation
            mock_declaration.parameters = MagicMock()
            mock_declaration.parameters.properties = {"arg1": MagicMock(type="string", description="Arg 1")}
            mock_declaration.parameters.required = ["arg1"]
            return mock_declaration

        async def execute(self, *args, **kwargs):
            return {"status": "mock tool executed"}

    def setup_method(self):
        """Set up test fixtures."""
        # Mock genai module
        self.genai_configure_patch = patch("google.generativeai.configure")
        self.mock_genai_configure = self.genai_configure_patch.start()

        self.genai_model_patch = patch("google.generativeai.GenerativeModel")
        self.mock_genai_model_class = self.genai_model_patch.start()
        self.mock_model_instance = MagicMock()
        self.mock_genai_model_class.return_value = self.mock_model_instance

        self.genai_list_models_patch = patch("google.generativeai.list_models")
        self.mock_genai_list_models = self.genai_list_models_patch.start()

        # Mock console
        self.mock_console = MagicMock(spec=Console)

        # Keep get_tool patch here if needed by other tests, or move into tests
        self.get_tool_patch = patch("src.cli_code.models.gemini.get_tool")
        self.mock_get_tool = self.get_tool_patch.start()
        # Configure default mock tool behavior if needed by other tests
        self.mock_tool = MagicMock()
        self.mock_tool.execute.return_value = "Default tool output"
        self.mock_get_tool.return_value = self.mock_tool

    def teardown_method(self):
        """Tear down test fixtures."""
        self.genai_configure_patch.stop()
        self.genai_model_patch.stop()
        self.genai_list_models_patch.stop()
        # REMOVED stops for os/glob/open mocks
        self.get_tool_patch.stop()

    def test_initialization(self):
        """Test initialization of GeminiModel."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")

        # Check if genai was configured correctly
        self.mock_genai_configure.assert_called_once_with(api_key="fake-api-key")

        # Check if model instance was created correctly
        self.mock_genai_model_class.assert_called_once()
        assert model.api_key == "fake-api-key"
        assert model.current_model_name == "gemini-2.5-pro-exp-03-25"

        # Check history initialization
        assert len(model.history) == 2  # System prompt and initial model response

    def test_initialize_model_instance(self):
        """Test model instance initialization."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")

        # Call the method directly to test
        model._initialize_model_instance()

        # Verify model was created with correct parameters
        self.mock_genai_model_class.assert_called_with(
            model_name="gemini-2.5-pro-exp-03-25",
            generation_config=model.generation_config,
            safety_settings=model.safety_settings,
            system_instruction=model.system_instruction,
        )

    def test_list_models(self):
        """Test listing available models."""
        # Set up mock response
        mock_model1 = MagicMock()
        mock_model1.name = "models/gemini-pro"
        mock_model1.display_name = "Gemini Pro"
        mock_model1.description = "A powerful model"
        mock_model1.supported_generation_methods = ["generateContent"]

        mock_model2 = MagicMock()
        mock_model2.name = "models/gemini-2.5-pro-exp-03-25"
        mock_model2.display_name = "Gemini 2.5 Pro"
        mock_model2.description = "An experimental model"
        mock_model2.supported_generation_methods = ["generateContent"]

        self.mock_genai_list_models.return_value = [mock_model1, mock_model2]

        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        result = model.list_models()

        # Verify list_models was called
        self.mock_genai_list_models.assert_called_once()

        # Verify result format
        assert len(result) == 2
        assert result[0]["id"] == "models/gemini-pro"
        assert result[0]["name"] == "Gemini Pro"
        assert result[1]["id"] == "models/gemini-2.5-pro-exp-03-25"

    def test_get_initial_context_with_rules_dir(self, tmp_path):
        """Test getting initial context from .rules directory using tmp_path."""
        # Arrange: Create .rules dir and files
        rules_dir = tmp_path / ".rules"
        rules_dir.mkdir()
        (rules_dir / "context.md").write_text("# Rule context")
        (rules_dir / "tools.md").write_text("# Rule tools")

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        # Act
        # Create model instance within the test CWD context
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-pro")
        context = model._get_initial_context()

        # Teardown
        os.chdir(original_cwd)

        # Assert
        assert "Project rules and guidelines:" in context
        assert "# Content from context.md" in context
        assert "# Rule context" in context
        assert "# Content from tools.md" in context
        assert "# Rule tools" in context

    def test_get_initial_context_with_readme(self, tmp_path):
        """Test getting initial context from README.md using tmp_path."""
        # Arrange: Create README.md
        readme_content = "# Project Readme Content"
        (tmp_path / "README.md").write_text(readme_content)

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        # Act
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-pro")
        context = model._get_initial_context()

        # Teardown
        os.chdir(original_cwd)

        # Assert
        assert "Project README:" in context
        assert readme_content in context

    def test_get_initial_context_with_ls_fallback(self, tmp_path):
        """Test getting initial context via ls fallback using tmp_path."""
        # Arrange: tmp_path is empty
        (tmp_path / "dummy_for_ls.txt").touch()  # Add a file for ls to find

        mock_ls_tool = MagicMock()
        ls_output = "dummy_for_ls.txt\n"
        mock_ls_tool.execute.return_value = ls_output

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        # Act: Patch get_tool locally
        # Note: GeminiModel imports get_tool directly
        with patch("src.cli_code.models.gemini.get_tool") as mock_get_tool:
            mock_get_tool.return_value = mock_ls_tool
            model = GeminiModel("fake-api-key", self.mock_console, "gemini-pro")
            context = model._get_initial_context()

        # Teardown
        os.chdir(original_cwd)

        # Assert
        mock_get_tool.assert_called_once_with("ls")
        mock_ls_tool.execute.assert_called_once()
        assert "Current directory contents" in context
        assert ls_output in context

    def test_create_tool_definitions(self):
        """Test creation of tool definitions for Gemini."""
        # Create a mock for AVAILABLE_TOOLS
        with patch("src.cli_code.models.gemini.AVAILABLE_TOOLS", new={"test_tool": MagicMock()}):
            # Mock the tool instance that will be created
            mock_tool_instance = MagicMock()
            mock_tool_instance.get_function_declaration.return_value = {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "param1": {"type": "string", "description": "A string parameter"},
                    "param2": {"type": "integer", "description": "An integer parameter"},
                },
                "required": ["param1"],
            }

            # Mock the tool class to return our mock instance
            mock_tool_class = MagicMock(return_value=mock_tool_instance)

            # Update the mocked AVAILABLE_TOOLS
            with patch("src.cli_code.models.gemini.AVAILABLE_TOOLS", new={"test_tool": mock_tool_class}):
                model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
                tools = model._create_tool_definitions()

                # Verify tools format
                assert len(tools) == 1
                assert tools[0]["name"] == "test_tool"
                assert "description" in tools[0]
                assert "parameters" in tools[0]

    def test_create_system_prompt(self):
        """Test creation of system prompt."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        prompt = model._create_system_prompt()

        # Verify prompt contains expected content
        assert "System Prompt for CLI-Code" in prompt

    def test_manage_context_window(self):
        """Test context window management."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        # Import the constant for clarity
        from src.cli_code.models.constants import MAX_HISTORY_TURNS

        # Add many messages to force context truncation
        # Start with history length 2 (system prompt + initial response)
        for i in range(30):
            model.add_to_history({"role": "user", "parts": [f"Test message {i}"]})
            model.add_to_history({"role": "model", "parts": [f"Test response {i}"]})

        # Record initial length (should be 2 + 30*2 = 62)
        initial_length = len(model.history)
        assert initial_length == 62  # Verify setup

        # Call context management
        model._manage_context_window()

        # Verify history was truncated to expected size: 2 initial + MAX_HISTORY_TURNS*2
        expected_length = 2 + (MAX_HISTORY_TURNS * 2)
        assert len(model.history) == expected_length  # Check specific length
        # The original check might also pass now if the logic is fixed, but let's be precise
        assert len(model.history) < initial_length

    def test_find_last_model_text(self):
        """Test finding last model text in history."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        model.history = []  # Clear initial history for predictable test
        model.add_to_history({"role": "user", "parts": ["User message 1"]})
        model.add_to_history({"role": "model", "parts": ["Model response 1"]})
        model.add_to_history({"role": "user", "parts": ["User message 2"]})
        model.add_to_history({"role": "model", "parts": ["Model response 2"]})
        result = model._find_last_model_text(model.history)
        assert result == "Model response 2"

    def test_add_to_history(self):
        """Test adding messages to history."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")

        # Clear history
        model.history = []

        # Add a message
        entry = {"role": "user", "parts": ["Test message"]}
        model.add_to_history(entry)

        # Verify message was added
        assert len(model.history) == 1
        assert model.history[0] == entry

    def test_clear_history(self):
        """Test clearing history."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")

        # Add a message
        model.add_to_history({"role": "user", "parts": ["Test message"]})

        # Clear history
        model.clear_history()

        # Verify history was cleared (should contain only system prompt + initial response)
        assert len(model.history) == 2  # Updated assertion: clear resets to initial state

    def test_get_help_text(self):
        """Test getting help text."""
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")
        help_text = model._get_help_text()

        # Verify help text content
        assert "CLI-Code Assistant Help" in help_text
        assert "Commands" in help_text

    def test_generate_with_function_calls(self):
        """Test generate method with function calls."""
        # Set up mock response with function call using proper mock structure
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_content = MagicMock()

        # Simulate Part and FunctionCall objects
        mock_function_part = MagicMock()
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param1": "value1"}
        mock_function_part.function_call = mock_function_call
        mock_function_part.text = None  # Ensure text is None for function call part

        mock_content.parts = [mock_function_part]
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "TOOL_CALLS"  # Use the actual enum value if available, otherwise string
        mock_response.candidates = [mock_candidate]

        # Create a second response (e.g., text after tool execution)
        mock_response_2 = MagicMock()
        mock_candidate_2 = MagicMock()
        mock_content_2 = MagicMock()
        mock_text_part_2 = MagicMock()
        mock_text_part_2.text = "Tool execution result"
        mock_content_2.parts = [mock_text_part_2]
        mock_candidate_2.content = mock_content_2
        mock_candidate_2.finish_reason = "STOP"  # Or 1
        mock_response_2.candidates = [mock_candidate_2]

        # Set up model instance to return the mock responses in sequence
        self.mock_model_instance.generate_content.side_effect = [mock_response, mock_response_2]

        # Mock tool execution
        tool_mock = MagicMock()
        tool_mock.execute.return_value = "Tool execution result"  # Match the text in mock_response_2
        self.mock_get_tool.return_value = tool_mock

        # Create model
        model = GeminiModel("fake-api-key", self.mock_console, "gemini-2.5-pro-exp-03-25")

        # Call generate
        result = model.generate("Test prompt")

        # Verify model was called (twice now)
        assert self.mock_model_instance.generate_content.call_count == 2

        # Verify tool lookup and execution
        self.mock_get_tool.assert_called_with("test_tool")
        tool_mock.execute.assert_called_once_with(param1="value1")

        # Verify the final result is the text from the second response
        assert result == "Tool execution result"


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.model_name = "gemini-test-model"
    config.api_key = "test-api-key"
    config.max_output_tokens = 100
    config.temperature = 0.7
    config.top_p = 0.9
    config.top_k = 40
    config.max_iterations = 5
    config.verbose = False
    # Mock the safe_get method

    def safe_get(key, default=None):
        return getattr(config, key, default)

    config.safe_get = MagicMock(side_effect=safe_get)
    return config


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock(spec=ToolRegistry)
    # Instantiate the nested class correctly
    mock_tool = TestGeminiModel.MockToolClass()
    registry.get_all_tools.return_value = [mock_tool]
    # Mock get_schemas to return a structure compatible with _create_tool_definitions
    registry.get_schemas.return_value = {mock_tool.name: mock_tool.get_function_declaration()}
    # Mock execute_tool to call the tool's execute method

    async def mock_execute_tool(name, args):
        if name == mock_tool.name:
            return await mock_tool.execute(**args)
        raise ValueError(f"Tool {name} not found")

    registry.execute_tool = AsyncMock(side_effect=mock_execute_tool)
    return registry


@pytest.fixture
def mock_console():
    return MagicMock()


@pytest.fixture
# Patch the actual google.generativeai.GenerativeModel class
@patch("google.generativeai.GenerativeModel", new_callable=MagicMock)
@patch("google.generativeai.configure")  # Also patch configure
def gemini_model(mock_genai_configure, mock_gen_model_class, mock_config, mock_tool_registry, mock_console):
    # Configure the mock class to return a mock instance
    mock_model_instance = MagicMock()
    mock_gen_model_class.return_value = mock_model_instance

    # Instantiate the class under test - it will use the patched GenerativeModel
    model = GeminiModel(config=mock_config, tool_registry=mock_tool_registry, console=mock_console)
    # We can now attach the mock instance for assertions if needed,
    # or configure its methods like generate_content_async
    model.model = mock_model_instance  # Ensure the instance uses the mock
    return model  # Return the actual GeminiModel instance


# --- Test Cases ---


# Test Initialization
@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
def test_gemini_model_initialization(gemini_model, mock_config, mock_tool_registry, mock_console):
    assert gemini_model.config == mock_config
    assert gemini_model.tool_registry == mock_tool_registry
    assert gemini_model.console == mock_console
    assert gemini_model.model is not None  # Check that the internal model is set (should be the mock)
    assert isinstance(gemini_model.history, list)
    assert len(gemini_model.history) == 1  # Should contain only system prompt initially
    assert gemini_model.history[0]["role"] == "system"
    # Check that genai.configure was called
    # mock_genai_configure.assert_called_once_with(api_key=mock_config.api_key)
    # Check that GenerativeModel was instantiated
    # mock_gen_model_class.assert_called_once()
    # We need to access the mocks passed by @patch, which pytest doesn't do directly here.
    # Instead, we can verify calls on the 'model' attribute if needed in specific tests.
    assert gemini_model.model is not None


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
def test_create_system_prompt(gemini_model):
    """Test that the system prompt contains key expected phrases."""
    # Access the history directly as it's set during initialization
    system_prompt_content = gemini_model.history[0]["parts"][0]
    assert isinstance(system_prompt_content, str)  # Now check the actual content
    assert "You are a helpful AI assistant" in system_prompt_content
    assert "tool description: Description for mock_tool" in system_prompt_content
    assert "parameter name: arg1" in system_prompt_content
    gemini_model._manage_context_window()

    # Check length after truncation
    final_length = len(gemini_model.history)
    # Expected: System Prompt (1) + Max Turns * (user + model_fc + user_fr = 3) = 1 + 3*MAX_HISTORY_TURNS
    expected_length = 1 + 3 * MAX_HISTORY_TURNS
    assert final_length == expected_length
    # Check that the oldest entries were removed (user 0, model 0, user 0)
    assert "user 0" not in str(gemini_model.history)
    assert "tool_0" not in str(gemini_model.history)
    # Check that the most recent entries are still present
    assert f"user {MAX_HISTORY_TURNS + 4}" in str(gemini_model.history)


# Test Response Parsing
@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
def test_extract_final_text_with_text(gemini_model):
    """Test extracting text from a STOP response with text."""
    from google.generativeai.types.content_types import Content, Part  # Import locally

    # Correct mock response structure
    mock_part = MagicMock(spec=Part)
    mock_part.text = "Expected text output"
    mock_content = MagicMock(spec=Content)
    mock_content.parts = [mock_part]
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = FinishReason.STOP  # or FinishReason.MAX_TOKENS

    # Pass the candidate directly to _extract_final_text
    extracted_text = gemini_model._extract_final_text(mock_candidate)
    assert extracted_text == "Expected text output\n"  # Method adds newline


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
def test_extract_final_text_no_text_part(gemini_model):
    """Test extraction when no text part is present in a STOP response."""
    from google.generativeai.types.content_types import Content, Part  # Import locally

    mock_part_no_text = MagicMock(spec=Part)
    # Remove or don't set the text attribute
    if hasattr(mock_part_no_text, "text"):
        del mock_part_no_text.text
    mock_content = MagicMock(spec=Content)
    mock_content.parts = [mock_part_no_text]  # Part exists, but no .text
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = FinishReason.STOP

    extracted_text = gemini_model._extract_final_text(mock_candidate)
    assert extracted_text == ""  # Expect empty string, not None


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
def test_extract_final_text_empty_parts(gemini_model):
    """Test extraction with empty parts list in a STOP response."""
    from google.generativeai.types.content_types import Content  # Import locally

    mock_content = MagicMock(spec=Content)
    mock_content.parts = []  # Empty parts list
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = FinishReason.STOP

    extracted_text = gemini_model._extract_final_text(mock_candidate)
    assert extracted_text == ""  # Expect empty string, not None


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
def test_extract_final_text_no_content(gemini_model):
    """Test extraction when the candidate has no content object."""
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.content = None  # No content object
    mock_candidate.finish_reason = FinishReason.STOP

    extracted_text = gemini_model._extract_final_text(mock_candidate)
    assert extracted_text == ""  # Expect empty string, not None


# Test Generation Flows
@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_generate_simple_text(gemini_model):
    """Test generating a simple text response without tools."""
    # Setup mock response for simple text
    mock_part = MagicMock(spec=Part)
    mock_part.text = "Simple answer"
    mock_content = MagicMock(spec=Content)
    mock_content.parts = [mock_part]
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = FinishReason.STOP
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    gemini_model.model.generate_content_async.return_value = mock_response

    # Call generate
    user_message = ChatMessage(role=Role.USER, content="Hello")
    response_message = await gemini_model.generate(user_message)

    # Assertions
    gemini_model.model.generate_content_async.assert_awaited_once()
    assert response_message.role == Role.MODEL
    assert response_message.content == "Simple answer"
    assert len(gemini_model.history) == 4  # System, Init Ack, User Msg, Model Resp


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_generate_with_function_calls(gemini_model):
    """Test the agent loop involving a function call and then a final text response."""
    # Mock 1: Function Call Response
    mock_func_call = MagicMock()
    mock_func_call.name = "mock_tool"
    mock_func_call.args = {"arg1": "value1"}
    mock_part_fc = MagicMock(spec=Part)
    mock_part_fc.function_call = mock_func_call
    # Ensure no text attribute is present on the function call part
    if hasattr(mock_part_fc, "text"):
        del mock_part_fc.text
    mock_content_fc = MagicMock(spec=Content)
    mock_content_fc.parts = [mock_part_fc]
    mock_candidate_fc = MagicMock(spec=Candidate)
    mock_candidate_fc.content = mock_content_fc
    mock_candidate_fc.finish_reason = FinishReason.TOOL_CALLS  # Or appropriate reason
    mock_response_fc = MagicMock()
    mock_response_fc.candidates = [mock_candidate_fc]

    # Mock 2: Final Text Response (after tool execution)
    mock_part_text = MagicMock(spec=Part)
    mock_part_text.text = "Final response after tool call"
    mock_content_text = MagicMock(spec=Content)
    mock_content_text.parts = [mock_part_text]
    mock_candidate_text = MagicMock(spec=Candidate)
    mock_candidate_text.content = mock_content_text
    mock_candidate_text.finish_reason = FinishReason.STOP
    mock_response_text = MagicMock()
    mock_response_text.candidates = [mock_candidate_text]

    # Setup side_effect for generate_content_async
    gemini_model.model.generate_content_async.side_effect = [mock_response_fc, mock_response_text]

    # Mock the tool execution itself (called via tool_registry)
    mock_tool_instance = gemini_model.tool_registry.execute_tool.return_value
    mock_tool_instance.execute = AsyncMock(return_value={"status": "mock tool success"})

    # Call generate
    user_message = ChatMessage(role=Role.USER, content="Use the mock tool")
    response_message = await gemini_model.generate(user_message)

    # Assertions
    assert gemini_model.model.generate_content_async.call_count == 2  # Called once for FC, once for final text
    mock_tool_instance.execute.assert_awaited_once_with(arg1="value1")  # Check tool execution
    assert response_message.role == Role.MODEL
    assert response_message.content == "Final response after tool call"  # Check final text

    # Check history includes user msg, model FC, user tool resp, final model text
    assert len(gemini_model.history) == 6
    assert gemini_model.history[-4]["role"] == "user"  # User request
    assert gemini_model.history[-3]["role"] == "model"  # Model requests FC
    assert gemini_model.history[-3]["parts"][0]["function_call"].name == "mock_tool"
    assert gemini_model.history[-2]["role"] == "user"  # User provides tool result
    assert gemini_model.history[-2]["parts"][0]["function_response"].name == "mock_tool"
    assert gemini_model.history[-1]["role"] == "model"  # Final model text response
    assert gemini_model.history[-1]["parts"][0]["text"] == "Final response after tool call"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_generate_with_task_complete(gemini_model):
    """Test the agent loop ending with the task_complete tool."""
    # Mock 1: task_complete Function Call Response
    mock_func_call = MagicMock()
    mock_func_call.name = "task_complete"
    mock_func_call.args = {"summary": "Task finished successfully."}
    mock_part_fc = MagicMock(spec=Part)
    mock_part_fc.function_call = mock_func_call
    if hasattr(mock_part_fc, "text"):
        del mock_part_fc.text
    mock_content_fc = MagicMock(spec=Content)
    mock_content_fc.parts = [mock_part_fc]
    mock_candidate_fc = MagicMock(spec=Candidate)
    mock_candidate_fc.content = mock_content_fc
    mock_candidate_fc.finish_reason = FinishReason.TOOL_CALLS
    mock_response_fc = MagicMock()
    mock_response_fc.candidates = [mock_candidate_fc]

    # Setup generate_content_async to return the task_complete call
    gemini_model.model.generate_content_async.return_value = mock_response_fc

    # Call generate
    user_message = ChatMessage(role=Role.USER, content="Finish the task.")
    response_message = await gemini_model.generate(user_message)

    # Assertions
    gemini_model.model.generate_content_async.assert_awaited_once()  # Only one call needed
    assert response_message.role == Role.MODEL  # Should return the final summary from task_complete
    assert response_message.content == "Task finished successfully."

    # Check history includes user msg, model FC (task_complete), user ack
    assert len(gemini_model.history) == 5
    assert gemini_model.history[-3]["role"] == "user"  # User request
    assert gemini_model.history[-2]["role"] == "model"  # Model calls task_complete
    assert gemini_model.history[-2]["parts"][0]["function_call"].name == "task_complete"
    assert gemini_model.history[-1]["role"] == "user"  # User acknowledges task_complete
    assert gemini_model.history[-1]["parts"][0]["function_response"].name == "task_complete"
    assert gemini_model.history[-1]["parts"][0]["function_response"].response["status"] == "acknowledged"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_generate_reaches_max_iterations(gemini_model):
    """Test the agent loop hitting the max iteration limit."""
    # Mock responses to always return a non-terminal function call
    mock_func_call = MagicMock()
    mock_func_call.name = "mock_tool"
    mock_func_call.args = {"arg1": "looping"}
    mock_part_fc = MagicMock(spec=Part)
    mock_part_fc.function_call = mock_func_call
    if hasattr(mock_part_fc, "text"):
        del mock_part_fc.text
    mock_content_fc = MagicMock(spec=Content)
    mock_content_fc.parts = [mock_part_fc]
    mock_candidate_fc = MagicMock(spec=Candidate)
    mock_candidate_fc.content = mock_content_fc
    mock_candidate_fc.finish_reason = FinishReason.TOOL_CALLS
    mock_response_fc = MagicMock()
    mock_response_fc.candidates = [mock_candidate_fc]

    # Make generate_content_async always return the looping function call
    gemini_model.model.generate_content_async.return_value = mock_response_fc

    # Mock the tool execution
    mock_tool_instance = gemini_model.tool_registry.execute_tool.return_value
    mock_tool_instance.execute = AsyncMock(return_value={"status": "still looping"})

    # Temporarily reduce max iterations for the test
    with patch("src.cli_code.models.gemini.MAX_AGENT_ITERATIONS", 3):
        user_message = ChatMessage(role=Role.USER, content="Start loop")
        response_message = await gemini_model.generate(user_message)

    # Assertions
    # Called once per iteration until max iterations reached
    assert gemini_model.model.generate_content_async.call_count == 3
    assert mock_tool_instance.execute.call_count == 2  # Tool executed in iter 1 and 2
    assert response_message.role == Role.MODEL
    assert "Task exceeded max iterations (3)" in response_message.content


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_generate_handles_tool_error(gemini_model):
    """Test handling when a tool execution raises an exception."""
    # Mock 1: Function Call Response
    mock_func_call = MagicMock()
    mock_func_call.name = "mock_tool"
    mock_func_call.args = {"arg1": "fail_me"}
    mock_part_fc = MagicMock(spec=Part)
    mock_part_fc.function_call = mock_func_call
    if hasattr(mock_part_fc, "text"):
        del mock_part_fc.text
    mock_content_fc = MagicMock(spec=Content)
    mock_content_fc.parts = [mock_part_fc]
    mock_candidate_fc = MagicMock(spec=Candidate)
    mock_candidate_fc.content = mock_content_fc
    mock_candidate_fc.finish_reason = FinishReason.TOOL_CALLS
    mock_response_fc = MagicMock()
    mock_response_fc.candidates = [mock_candidate_fc]

    # Mock 2: Final Text Response (after tool error) - Model should ideally report error
    mock_part_text = MagicMock(spec=Part)
    mock_part_text.text = "Error occurred."  # Or whatever model says
    mock_content_text = MagicMock(spec=Content)
    mock_content_text.parts = [mock_part_text]
    mock_candidate_text = MagicMock(spec=Candidate)
    mock_candidate_text.content = mock_content_text
    mock_candidate_text.finish_reason = FinishReason.STOP
    mock_response_text = MagicMock()
    mock_response_text.candidates = [mock_candidate_text]

    # Setup side_effect for generate_content_async
    gemini_model.model.generate_content_async.side_effect = [mock_response_fc, mock_response_text]

    # Mock the tool execution to raise an error
    mock_tool_instance = gemini_model.tool_registry.execute_tool.return_value
    tool_error_message = "Tool failed spectacularly"
    mock_tool_instance.execute = AsyncMock(side_effect=ValueError(tool_error_message))

    # Call generate
    user_message = ChatMessage(role=Role.USER, content="Use the failing tool")
    response_message = await gemini_model.generate(user_message)

    # Assertions
    assert (
        gemini_model.model.generate_content_async.call_count == 2
    )  # Called for FC, then called again after error response
    mock_tool_instance.execute.assert_awaited_once_with(arg1="fail_me")  # Tool was called
    assert response_message.role == Role.MODEL
    assert response_message.content == "Error occurred."  # Check final text

    # Check history includes the error response from the tool execution attempt
    assert len(gemini_model.history) == 6
    assert gemini_model.history[-2]["role"] == "user"  # User provides tool result (which is an error)
    assert gemini_model.history[-2]["parts"][0]["function_response"].name == "mock_tool"
    # The _execute_function_call wraps the exception message
    assert (
        f"Error: Tool execution error with mock_tool: {tool_error_message}"
        in gemini_model.history[-2]["parts"][0]["function_response"].response["output"]
    )
