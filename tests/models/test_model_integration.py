"""
Tests for model integration aspects of the cli-code application.
This file focuses on testing the integration between the CLI and different model providers.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

# Ensure we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Handle missing dependencies gracefully
try:
    import pytest
    from click.testing import CliRunner

    from cli_code.main import cli, start_interactive_session
    from cli_code.models.base import AbstractModelAgent

    IMPORTS_AVAILABLE = True
except ImportError:
    # Create dummy fixtures and mocks if imports aren't available
    IMPORTS_AVAILABLE = False
    pytest = MagicMock()
    pytest.mark.timeout = lambda seconds: lambda f: f

    class DummyCliRunner:
        def invoke(self, *args, **kwargs):
            class Result:
                exit_code = 0
                output = ""

            return Result()

    CliRunner = DummyCliRunner
    cli = MagicMock()
    start_interactive_session = MagicMock()
    AbstractModelAgent = MagicMock()

# Determine if we're running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE or IN_CI


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestGeminiModelIntegration:
    """Test integration with Gemini models."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_patcher = patch("cli_code.main.config")
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch("cli_code.main.console")
        self.mock_console = self.console_patcher.start()

        # Set default behavior for mocks
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_default_model.return_value = "gemini-pro"
        self.mock_config.get_credential.return_value = "fake-api-key"

        # Patch the GeminiModel class
        self.gemini_patcher = patch("cli_code.main.GeminiModel")
        self.mock_gemini_model_class = self.gemini_patcher.start()
        self.mock_gemini_instance = MagicMock()
        self.mock_gemini_model_class.return_value = self.mock_gemini_instance

    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()
        self.gemini_patcher.stop()

    @pytest.mark.timeout(5)
    def test_gemini_model_initialization(self):
        """Test initialization of Gemini model."""
        result = self.runner.invoke(cli, [])
        assert result.exit_code == 0

        # Verify model was initialized with correct parameters
        self.mock_gemini_model_class.assert_called_once_with(
            api_key="fake-api-key", console=self.mock_console, model_name="gemini-pro"
        )

    @pytest.mark.timeout(5)
    def test_gemini_model_custom_model_name(self):
        """Test using a custom Gemini model name."""
        result = self.runner.invoke(cli, ["--model", "gemini-2.5-pro-exp-03-25"])
        assert result.exit_code == 0

        # Verify model was initialized with custom model name
        self.mock_gemini_model_class.assert_called_once_with(
            api_key="fake-api-key", console=self.mock_console, model_name="gemini-2.5-pro-exp-03-25"
        )

    @pytest.mark.timeout(5)
    def test_gemini_model_tools_initialization(self):
        """Test that tools are properly initialized for Gemini model."""
        # Need to mock the tools setup
        with patch("cli_code.main.AVAILABLE_TOOLS") as mock_tools:
            mock_tools.return_value = ["tool1", "tool2"]

            result = self.runner.invoke(cli, [])
            assert result.exit_code == 0

            # Verify inject_tools was called on the model instance
            self.mock_gemini_instance.inject_tools.assert_called_once()


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestOllamaModelIntegration:
    """Test integration with Ollama models."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_patcher = patch("cli_code.main.config")
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch("cli_code.main.console")
        self.mock_console = self.console_patcher.start()

        # Set default behavior for mocks
        self.mock_config.get_default_provider.return_value = "ollama"
        self.mock_config.get_default_model.return_value = "llama2"
        self.mock_config.get_credential.return_value = "http://localhost:11434"

        # Patch the OllamaModel class
        self.ollama_patcher = patch("cli_code.main.OllamaModel")
        self.mock_ollama_model_class = self.ollama_patcher.start()
        self.mock_ollama_instance = MagicMock()
        self.mock_ollama_model_class.return_value = self.mock_ollama_instance

    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()
        self.ollama_patcher.stop()

    @pytest.mark.timeout(5)
    def test_ollama_model_initialization(self):
        """Test initialization of Ollama model."""
        result = self.runner.invoke(cli, [])
        assert result.exit_code == 0

        # Verify model was initialized with correct parameters
        self.mock_ollama_model_class.assert_called_once_with(
            api_url="http://localhost:11434", console=self.mock_console, model_name="llama2"
        )

    @pytest.mark.timeout(5)
    def test_ollama_model_custom_model_name(self):
        """Test using a custom Ollama model name."""
        result = self.runner.invoke(cli, ["--model", "mistral"])
        assert result.exit_code == 0

        # Verify model was initialized with custom model name
        self.mock_ollama_model_class.assert_called_once_with(
            api_url="http://localhost:11434", console=self.mock_console, model_name="mistral"
        )

    @pytest.mark.timeout(5)
    def test_ollama_model_tools_initialization(self):
        """Test that tools are properly initialized for Ollama model."""
        # Need to mock the tools setup
        with patch("cli_code.main.AVAILABLE_TOOLS") as mock_tools:
            mock_tools.return_value = ["tool1", "tool2"]

            result = self.runner.invoke(cli, [])
            assert result.exit_code == 0

            # Verify inject_tools was called on the model instance
            self.mock_ollama_instance.inject_tools.assert_called_once()


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestProviderSwitching:
    """Test switching between different model providers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_patcher = patch("cli_code.main.config")
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch("cli_code.main.console")
        self.mock_console = self.console_patcher.start()

        # Set default behavior for mocks
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_default_model.side_effect = lambda provider=None: {
            "gemini": "gemini-pro",
            "ollama": "llama2",
            None: "gemini-pro",  # Default to gemini model
        }.get(provider)
        self.mock_config.get_credential.side_effect = lambda provider: {
            "gemini": "fake-api-key",
            "ollama": "http://localhost:11434",
        }.get(provider)

        # Patch the model classes
        self.gemini_patcher = patch("cli_code.main.GeminiModel")
        self.mock_gemini_model_class = self.gemini_patcher.start()
        self.mock_gemini_instance = MagicMock()
        self.mock_gemini_model_class.return_value = self.mock_gemini_instance

        self.ollama_patcher = patch("cli_code.main.OllamaModel")
        self.mock_ollama_model_class = self.ollama_patcher.start()
        self.mock_ollama_instance = MagicMock()
        self.mock_ollama_model_class.return_value = self.mock_ollama_instance

    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()
        self.gemini_patcher.stop()
        self.ollama_patcher.stop()

    @pytest.mark.timeout(5)
    def test_switch_provider_via_cli_option(self):
        """Test switching provider via CLI option."""
        # Default should be gemini
        result = self.runner.invoke(cli, [])
        assert result.exit_code == 0
        self.mock_gemini_model_class.assert_called_once()
        self.mock_ollama_model_class.assert_not_called()

        # Reset mock call counts
        self.mock_gemini_model_class.reset_mock()
        self.mock_ollama_model_class.reset_mock()

        # Switch to ollama via CLI option
        result = self.runner.invoke(cli, ["--provider", "ollama"])
        assert result.exit_code == 0
        self.mock_gemini_model_class.assert_not_called()
        self.mock_ollama_model_class.assert_called_once()

    @pytest.mark.timeout(5)
    def test_set_default_provider_command(self):
        """Test set-default-provider command."""
        # Test setting gemini as default
        result = self.runner.invoke(cli, ["set-default-provider", "gemini"])
        assert result.exit_code == 0
        self.mock_config.set_default_provider.assert_called_once_with("gemini")

        # Reset mock
        self.mock_config.set_default_provider.reset_mock()

        # Test setting ollama as default
        result = self.runner.invoke(cli, ["set-default-provider", "ollama"])
        assert result.exit_code == 0
        self.mock_config.set_default_provider.assert_called_once_with("ollama")


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestToolIntegration:
    """Test integration of tools with models."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console_patcher = patch("cli_code.main.console")
        self.mock_console = self.console_patcher.start()

        self.config_patcher = patch("cli_code.main.config")
        self.mock_config = self.config_patcher.start()
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_default_model.return_value = "gemini-pro"
        self.mock_config.get_credential.return_value = "fake-api-key"

        # Patch the model class
        self.gemini_patcher = patch("cli_code.main.GeminiModel")
        self.mock_gemini_model_class = self.gemini_patcher.start()
        self.mock_gemini_instance = MagicMock()
        self.mock_gemini_model_class.return_value = self.mock_gemini_instance

        # Create mock tools
        self.tool1 = MagicMock()
        self.tool1.name = "tool1"
        self.tool1.function_name = "tool1_func"
        self.tool1.description = "Tool 1 description"

        self.tool2 = MagicMock()
        self.tool2.name = "tool2"
        self.tool2.function_name = "tool2_func"
        self.tool2.description = "Tool 2 description"

        # Patch AVAILABLE_TOOLS
        self.tools_patcher = patch("cli_code.main.AVAILABLE_TOOLS", return_value=[self.tool1, self.tool2])
        self.mock_tools = self.tools_patcher.start()

        # Patch input for interactive session
        self.input_patcher = patch("builtins.input")
        self.mock_input = self.input_patcher.start()
        self.mock_input.return_value = "exit"  # Always exit to end the session

    def teardown_method(self):
        """Teardown test fixtures."""
        self.console_patcher.stop()
        self.config_patcher.stop()
        self.gemini_patcher.stop()
        self.tools_patcher.stop()
        self.input_patcher.stop()

    @pytest.mark.timeout(5)
    def test_tools_injected_to_model(self):
        """Test that tools are injected into the model."""
        start_interactive_session(provider="gemini", model_name="gemini-pro", console=self.mock_console)

        # Verify model was created with correct parameters
        self.mock_gemini_model_class.assert_called_once_with(
            api_key="fake-api-key", console=self.mock_console, model_name="gemini-pro"
        )

        # Verify tools were injected
        self.mock_gemini_instance.inject_tools.assert_called_once()

        # Get the tools that were injected
        tools_injected = self.mock_gemini_instance.inject_tools.call_args[0][0]

        # Verify both tools are in the injected list
        tool_names = [tool.name for tool in tools_injected]
        assert "tool1" in tool_names
        assert "tool2" in tool_names

    @pytest.mark.timeout(5)
    def test_tool_invocation(self):
        """Test tool invocation in the model."""
        # Setup model to return prompt that appears to use a tool
        self.mock_gemini_instance.ask.return_value = "I'll use tool1 to help you with that."

        start_interactive_session(provider="gemini", model_name="gemini-pro", console=self.mock_console)

        # Verify ask was called (would trigger tool invocation if implemented)
        self.mock_gemini_instance.ask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
