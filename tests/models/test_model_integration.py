"""
Tests for model integration aspects of the cli-code application.
This file focuses on testing the integration between the CLI and different model providers.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, mock_open, patch

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


# Base class for model integration tests
class BaseModelIntegrationTest(unittest.TestCase):
    def setup_method(self, method):
        """Set up test fixtures."""
        self.runner = CliRunner()
        # Mock the config *instance* used in main.py
        self.config_patcher = patch("cli_code.main.config", new_callable=MagicMock)
        self.mock_config_instance = self.config_patcher.start()
        # No need for mock_config_class anymore

        # Configure the mock instance directly
        self.mock_config_instance.get_credential.side_effect = lambda provider: {
            "gemini": "fake-gemini-key",
            "ollama": "http://fake-ollama-url:11434",
            "github": "fake-gh-token",
            "mcp": "fake-mcp-key",
        }.get(provider)
        self.mock_config_instance.get_default_provider.return_value = "gemini"  # Default to Gemini
        self.mock_config_instance.get_default_model.side_effect = lambda provider=None: {
            "gemini": "gemini-pro",
            "ollama": "llama3",
        }.get(provider or self.mock_config_instance.get_default_provider())
        self.mock_config_instance.config = {  # Mock the raw config dict if needed elsewhere
            "google_api_key": "fake-gemini-key",
            "ollama_api_url": "http://fake-ollama-url:11434",
            "default_provider": "gemini",
            "default_model": "gemini-pro",
            "ollama_default_model": "llama3",
            "settings": {
                "max_tokens": 8000,
                "temperature": 0.7,
                "token_warning_threshold": 6000,
                "auto_compact_threshold": 7500,
            },
            "mcp_host_url": "http://fake-mcp-url",
            "tools": {"use_github_cli": False},
        }
        self.mock_config_instance.get_setting.side_effect = (
            lambda key, default=None: self.mock_config_instance.config.get("settings", {}).get(key, default)
        )

        # Basic console mock - specific methods might need further mocking in subclasses
        self.console_patcher = patch("rich.console.Console")
        self.mock_console = self.console_patcher.start()

        # Stop patches after test using addCleanup for robustness
        self.addCleanup(self.config_patcher.stop)
        self.addCleanup(self.console_patcher.stop)

    def teardown_method(self, method):
        # Cleanup is handled by addCleanup in setup_method
        pass


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestGeminiModelIntegration(BaseModelIntegrationTest):
    """Test integration with Gemini models."""

    def setup_method(self, method):
        """Set up specifically for Gemini tests."""
        super().setup_method(method)
        # Set default provider to gemini for this test class
        self.mock_config_instance.get_default_provider.return_value = "gemini"

        # Mock the GeminiModel class *where it's imported/used in main.py*
        self.gemini_patcher = patch("cli_code.main.GeminiModel")
        self.mock_gemini_model_class = self.gemini_patcher.start()
        self.mock_gemini_instance = MagicMock(spec=AbstractModelAgent)  # Use AbstractModelAgent spec
        self.mock_gemini_model_class.return_value = self.mock_gemini_instance

        # Set a default return value for generate to avoid TypeErrors during invoke
        self.mock_gemini_instance.generate.return_value = "Mocked Gemini Response"

        # Add cleanup for the Gemini patcher
        self.addCleanup(self.gemini_patcher.stop)

    @pytest.mark.timeout(5)
    def test_gemini_model_initialization(self):
        """Test initialization of Gemini model."""
        # Set default provider to gemini for this test
        self.mock_config_instance.get_default_provider.return_value = "gemini"

        result = self.runner.invoke(cli, input="hello\n/exit\n")  # Need input for session
        assert result.exit_code == 0

        # Verify model was initialized with correct parameters from mocked config
        self.mock_gemini_model_class.assert_called_once_with(
            api_key="fake-gemini-key", console=ANY, model_name="gemini-pro"
        )

    @pytest.mark.timeout(5)
    def test_gemini_model_custom_model_name(self):
        """Test using a custom Gemini model name."""
        # Mocking is done in setup
        # Set default provider to gemini for this test
        self.mock_config_instance.get_default_provider.return_value = "gemini"

        result = self.runner.invoke(cli, ["--model", "gemini-2.5-pro-exp-03-25"], input="hello\n/exit\n")
        assert result.exit_code == 0

        # Verify model was initialized with custom model name
        self.mock_gemini_model_class.assert_called_once_with(
            api_key="fake-gemini-key", console=ANY, model_name="gemini-2.5-pro-exp-03-25"
        )


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestOllamaModelIntegration(BaseModelIntegrationTest):
    """Integration tests specifically for the Ollama provider."""

    def setup_method(self, method):
        """Set up specifically for Ollama tests."""
        super().setup_method(method)
        # Set default provider to ollama for this test class
        self.mock_config_instance.get_default_provider.return_value = "ollama"
        # Ensure the API URL is mocked correctly (via get_credential)
        self.mock_config_instance.get_credential.side_effect = lambda provider: {
            "gemini": "fake-gemini-key",
            "ollama": "http://fake-ollama-url:11434",
            "github": "fake-gh-token",
            "mcp": "fake-mcp-key",
        }.get(provider)
        # Ensure the default ollama model is returned correctly
        self.mock_config_instance.get_default_model.side_effect = lambda provider=None: {
            "gemini": "gemini-pro",
            "ollama": "llama3",
        }.get(provider or self.mock_config_instance.get_default_provider())

        # Mock the OllamaModel class *where it's imported/used in main.py*
        self.ollama_patcher = patch("cli_code.main.OllamaModel")
        self.mock_ollama_model_class = self.ollama_patcher.start()
        self.mock_ollama_instance = MagicMock(spec=AbstractModelAgent)  # Use AbstractModelAgent spec
        self.mock_ollama_model_class.return_value = self.mock_ollama_instance

        # Set a default return value for generate to avoid TypeErrors during invoke
        self.mock_ollama_instance.generate.return_value = "Mocked Ollama Response"

        # Add cleanup for the Ollama patcher
        self.addCleanup(self.ollama_patcher.stop)

    @pytest.mark.timeout(5)
    def test_ollama_default_model(self):
        """Test using the default Ollama model."""
        result = self.runner.invoke(cli, input="Explain pytest.\n/exit\n")
        assert result.exit_code == 0
        # Verify OllamaModel was called with default model from mocked config
        self.mock_ollama_model_class.assert_called_once_with(
            api_url="http://fake-ollama-url:11434",
            console=ANY,  # Check for any Console instance
            model_name="llama3",  # Default Ollama model from mocked config
        )
        # Verify generate was called at least once (via the mocked session)
        self.mock_ollama_instance.generate.assert_called()
        # Can't reliably check prompt contents if interactive session is complex

    @pytest.mark.timeout(5)
    def test_ollama_model_custom_model_name(self):
        """Test using a custom Ollama model name."""
        result = self.runner.invoke(cli, args=["--model", "mistral"], input="Explain pytest.\n/exit\n")
        assert result.exit_code == 0

        # Verify OllamaModel was initialized with the custom model name
        self.mock_ollama_model_class.assert_called_once_with(
            api_url="http://fake-ollama-url:11434",
            console=ANY,  # Check for any Console instance
            model_name="mistral",  # Expect the custom model name
        )
        # Verify generate was called at least once
        self.mock_ollama_instance.generate.assert_called()


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestProviderSwitching(BaseModelIntegrationTest):
    """Test switching between different model providers."""

    def setup_method(self, method):
        """Set up test fixtures."""
        super().setup_method(method)  # Gets runner and patched config instance

        # Patch Console where it's used in main.py
        self.console_patcher = patch("cli_code.main.console")  # Patch console directly in main
        self.mock_console = self.console_patcher.start()
        self.addCleanup(self.console_patcher.stop)

        # Patch the model classes where they are used in main.py
        self.gemini_patcher = patch("cli_code.main.GeminiModel")
        self.mock_gemini_model_class = self.gemini_patcher.start()
        self.mock_gemini_instance = MagicMock(spec=AbstractModelAgent)
        self.mock_gemini_model_class.return_value = self.mock_gemini_instance
        self.mock_gemini_instance.generate.return_value = "Mock Gemini"
        self.addCleanup(self.gemini_patcher.stop)

        self.ollama_patcher = patch("cli_code.main.OllamaModel")
        self.mock_ollama_model_class = self.ollama_patcher.start()
        self.mock_ollama_instance = MagicMock(spec=AbstractModelAgent)
        self.mock_ollama_model_class.return_value = self.mock_ollama_instance
        self.mock_ollama_instance.generate.return_value = "Mock Ollama"
        self.addCleanup(self.ollama_patcher.stop)

        # Mock start_interactive_session to prevent actual loop and check which agent is passed
        self.session_patcher = patch("cli_code.main.start_interactive_session")
        self.mock_start_session = self.session_patcher.start()
        self.addCleanup(self.session_patcher.stop)

    @pytest.mark.timeout(5)
    def test_switch_provider_via_cli_option(self):
        """Test switching provider via CLI option."""
        # Default should be gemini (mocked config)
        self.mock_config_instance.get_default_provider.return_value = "gemini"
        result = self.runner.invoke(cli, [])  # No input needed if session is mocked
        assert result.exit_code == 0
        # self.mock_gemini_model_class.assert_called_once() # Removed: Model instantiated inside mocked session
        self.mock_ollama_model_class.assert_not_called()
        # Verify start_interactive_session called with correct provider and model name
        # Call args are (provider, model_name, console)
        # Agent instance is created *inside* start_interactive_session, which is mocked away
        self.mock_start_session.assert_called_once_with(provider="gemini", model_name="gemini-pro", console=ANY)

        # Reset mocks
        self.mock_gemini_model_class.reset_mock()
        self.mock_ollama_model_class.reset_mock()
        self.mock_start_session.reset_mock()

        # Switch to ollama via CLI option
        result = self.runner.invoke(cli, ["--provider", "ollama"])
        assert result.exit_code == 0
        self.mock_gemini_model_class.assert_not_called()
        # self.mock_ollama_model_class.assert_called_once() # Removed: Model instantiated inside mocked session
        # Verify start_interactive_session called with Ollama
        self.mock_start_session.assert_called_once_with(provider="ollama", model_name="llama3", console=ANY)

    @pytest.mark.timeout(5)
    def test_set_default_provider_command(self):
        """Test set-default-provider command."""
        # Unpatch start_interactive_session for this test as we're not running the session
        self.session_patcher.stop()  # Stop it temporarily

        # Test setting gemini as default
        result = self.runner.invoke(cli, ["set-default-provider", "gemini"])
        assert result.exit_code == 0
        # Check that the set_default_provider method on the *mock config instance* was called
        self.mock_config_instance.set_default_provider.assert_called_once_with("gemini")

        # Reset mock
        self.mock_config_instance.set_default_provider.reset_mock()

        # Test setting ollama as default
        result = self.runner.invoke(cli, ["set-default-provider", "ollama"])
        assert result.exit_code == 0
        self.mock_config_instance.set_default_provider.assert_called_once_with("ollama")

        # Repatch start_interactive_session (although teardown would handle it)
        self.session_patcher.start()


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestToolIntegration(BaseModelIntegrationTest):
    """Test integration of tools with models."""

    def setup_method(self, method):
        """Set up test fixtures."""
        super().setup_method(method)  # Call base setup (gets runner, patched config instance)

        # Set default provider to gemini for tool tests
        self.mock_config_instance.get_default_provider.return_value = "gemini"

        # Patch the model class specifically for these tests
        self.gemini_patcher = patch("cli_code.main.GeminiModel")
        self.mock_gemini_model_class = self.gemini_patcher.start()
        self.mock_gemini_instance = MagicMock(spec=AbstractModelAgent)
        self.mock_gemini_model_class.return_value = self.mock_gemini_instance
        self.mock_gemini_instance.generate.return_value = "Mock Gemini Tool Response"
        self.addCleanup(self.gemini_patcher.stop)  # Add cleanup for this patch

        # Mock start_interactive_session to prevent actual loop and control flow
        self.session_patcher = patch("cli_code.main.start_interactive_session")
        self.mock_start_session = self.session_patcher.start()
        self.addCleanup(self.session_patcher.stop)

    @pytest.mark.timeout(5)
    def test_tool_integration_setup(self):
        """Verify that the test setup correctly uses the patched config instance."""
        result = self.runner.invoke(cli, [])  # Invoke the CLI
        assert result.exit_code == 0
        # Config instance is already patched in base setup, check if it was accessed
        self.mock_config_instance.get_default_provider.assert_called()
        # self.mock_gemini_model_class.assert_called_once() # Removed: Model instantiated inside mocked session
        # Verify start_interactive_session was called with expected defaults
        self.mock_start_session.assert_called_once_with(provider="gemini", model_name="gemini-pro", console=ANY)


# Keep main guard for potential direct execution (though unlikely with pytest)
if __name__ == "__main__":
    pass
