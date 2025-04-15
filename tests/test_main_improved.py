"""
Improved tests for the main module to increase coverage.
This file focuses on testing error handling, edge cases, and untested code paths.
"""

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, mock_open, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from cli_code.main import cli, console, show_help, start_interactive_session
from cli_code.tools.directory_tools import LsTool

# Ensure we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Handle missing dependencies gracefully
try:
    pass  # Imports moved to top
    # import pytest
    # from click.testing import CliRunner
    # from cli_code.main import cli, start_interactive_session, show_help, console
except ImportError:
    # If imports fail, provide a helpful message and skip these tests.
    # This handles cases where optional dependencies (like click) might be missing.
    pytest.skip(
        "Missing optional dependencies (like click), skipping integration tests for main.", allow_module_level=True
    )

# Determine if we're running in CI
IS_CI = os.getenv("CI") == "true"


# Helper function for generate side_effect
def generate_sequence(responses):
    """Creates a side_effect function that yields responses then raises."""
    iterator = iter(responses)

    def side_effect(*args, **kwargs):
        try:
            return next(iterator)
        except StopIteration as err:
            raise AssertionError(f"mock_agent.generate called unexpectedly with args: {args}, kwargs: {kwargs}") from None

    return side_effect


@pytest.mark.integration
@pytest.mark.timeout(10)  # Timeout after 10 seconds
class TestMainErrorHandling:
    """Test error handling in the main module."""

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

        self.interactive_patcher = patch("cli_code.main.start_interactive_session")
        self.mock_interactive = self.interactive_patcher.start()
        self.mock_interactive.return_value = None

    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()
        self.interactive_patcher.stop()

    @pytest.mark.timeout(5)
    def test_cli_with_missing_config(self):
        """Test CLI behavior when config is None."""
        with patch("cli_code.main.config", None):
            result = self.runner.invoke(cli, [])
            assert result.exit_code == 1

    @pytest.mark.timeout(5)
    def test_cli_with_missing_model(self):
        """Test CLI behavior when no model is provided or configured."""
        # Set up config to return None for get_default_model
        self.mock_config.get_default_model.return_value = None

        result = self.runner.invoke(cli, [])
        assert result.exit_code == 1
        self.mock_console.print.assert_any_call(
            "[bold red]Error:[/bold red] No default model configured for provider 'gemini' and no model specified with --model."
        )

    @pytest.mark.timeout(5)
    def test_setup_with_missing_config(self):
        """Test setup command behavior when config is None."""
        with patch("cli_code.main.config", None):
            result = self.runner.invoke(cli, ["setup", "--provider", "gemini", "api-key"])
            assert result.exit_code == 1, "Setup should exit with 1 on config error"

    @pytest.mark.timeout(5)
    def test_setup_with_exception(self):
        """Test setup command when an exception occurs."""
        self.mock_config.set_credential.side_effect = Exception("Test error")

        result = self.runner.invoke(cli, ["setup", "--provider", "gemini", "api-key"])
        assert result.exit_code == 0

        # Check that error was printed
        self.mock_console.print.assert_any_call("[bold red]Error saving API Key:[/bold red] Test error")

    @pytest.mark.timeout(5)
    def test_set_default_provider_with_exception(self):
        """Test set-default-provider when an exception occurs."""
        self.mock_config.set_default_provider.side_effect = Exception("Test error")

        result = self.runner.invoke(cli, ["set-default-provider", "gemini"])
        assert result.exit_code == 0

        # Check that error was printed
        self.mock_console.print.assert_any_call("[bold red]Error setting default provider:[/bold red] Test error")

    @pytest.mark.timeout(5)
    def test_set_default_model_with_exception(self):
        """Test set-default-model when an exception occurs."""
        self.mock_config.set_default_model.side_effect = Exception("Test error")

        result = self.runner.invoke(cli, ["set-default-model", "gemini-pro"])
        assert result.exit_code == 0

        # Check that error was printed
        self.mock_console.print.assert_any_call(
            "[bold red]Error setting default model for gemini:[/bold red] Test error"
        )


@pytest.mark.integration
@pytest.mark.timeout(10)  # Timeout after 10 seconds
class TestListModelsCommand:
    """Test list-models command thoroughly."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_patcher = patch("cli_code.main.config")
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch("cli_code.main.console")
        self.mock_console = self.console_patcher.start()

        # Set default behavior for mocks
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_credential.return_value = "fake-api-key"
        self.mock_config.get_default_model.return_value = "gemini-pro"

    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()

    @pytest.mark.timeout(5)
    def test_list_models_with_missing_config(self):
        """Test list-models when config is None."""
        with patch("cli_code.main.config", None):
            result = self.runner.invoke(cli, ["list-models"])
            assert result.exit_code == 1, "list-models should exit with 1 on config error"

    @pytest.mark.timeout(5)
    def test_list_models_with_missing_credential(self):
        """Test list-models when credential is missing."""
        self.mock_config.get_credential.return_value = None

        result = self.runner.invoke(cli, ["list-models", "--provider", "gemini"])
        assert result.exit_code == 0

        # Check that error was printed
        self.mock_console.print.assert_any_call("[bold red]Error:[/bold red] Gemini API Key not found.")

    @pytest.mark.timeout(5)
    def test_list_models_with_empty_list(self):
        """Test list-models when no models are returned."""
        with patch("cli_code.main.GeminiModel") as mock_gemini_model:
            mock_instance = MagicMock()
            mock_instance.list_models.return_value = []
            mock_gemini_model.return_value = mock_instance

            result = self.runner.invoke(cli, ["list-models", "--provider", "gemini"])
            assert result.exit_code == 0

            # Check message about no models
            self.mock_console.print.assert_any_call(
                "[yellow]No models found or reported by provider 'gemini'.[/yellow]"
            )

    @pytest.mark.timeout(5)
    def test_list_models_with_exception(self):
        """Test list-models when an exception occurs."""
        with patch("cli_code.main.GeminiModel") as mock_gemini_model:
            mock_gemini_model.side_effect = Exception("Test error")

            result = self.runner.invoke(cli, ["list-models", "--provider", "gemini"])
            assert result.exit_code == 0

            # Check error message
            self.mock_console.print.assert_any_call("[bold red]Error listing models for gemini:[/bold red] Test error")

    @pytest.mark.timeout(5)
    def test_list_models_with_unknown_provider(self):
        """Test list-models with an unknown provider (custom mock value)."""
        # Use mock to override get_default_provider with custom, invalid value
        self.mock_config.get_default_provider.return_value = "unknown"

        # Using provider from config (let an unknown response come back)
        result = self.runner.invoke(cli, ["list-models"])
        assert result.exit_code == 0

        # Should report unknown provider
        self.mock_console.print.assert_any_call("[bold red]Error:[/bold red] Unknown provider 'unknown'.")


@pytest.mark.integration
@pytest.mark.timeout(10)  # Timeout after 10 seconds
class TestInteractiveSession:
    """Test interactive session functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_patcher = patch("cli_code.main.config")
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch("cli_code.main.console")
        self.mock_console = self.console_patcher.start()

        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_credential.return_value = "fake-api-key"
        self.mock_config.get_default_model.return_value = "gemini-pro"  # Provide default model

        # Mock model classes used in start_interactive_session
        self.gemini_patcher = patch("cli_code.main.GeminiModel")
        self.mock_gemini_model_class = self.gemini_patcher.start()
        self.ollama_patcher = patch("cli_code.main.OllamaModel")
        self.mock_ollama_model_class = self.ollama_patcher.start()

        # Mock instance returned by model classes
        self.mock_agent = MagicMock()
        self.mock_gemini_model_class.return_value = self.mock_agent
        self.mock_ollama_model_class.return_value = self.mock_agent

        # Mock file system checks used for context messages
        self.isdir_patcher = patch("cli_code.main.os.path.isdir")
        self.mock_isdir = self.isdir_patcher.start()
        self.isfile_patcher = patch("cli_code.main.os.path.isfile")
        self.mock_isfile = self.isfile_patcher.start()
        self.listdir_patcher = patch("cli_code.main.os.listdir")
        self.mock_listdir = self.listdir_patcher.start()

    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()
        self.gemini_patcher.stop()
        self.ollama_patcher.stop()
        self.isdir_patcher.stop()
        self.isfile_patcher.stop()
        self.listdir_patcher.stop()

    @pytest.mark.timeout(5)
    def test_interactive_session_with_missing_config(self):
        """Test interactive session when config is None."""
        # This test checks logic before model instantiation, so no generate mock needed
        with patch("cli_code.main.config", None):
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=self.mock_console)
            self.mock_console.print.assert_any_call("[bold red]Config error.[/bold red]")

    @pytest.mark.timeout(5)
    def test_interactive_session_with_missing_credential(self):
        """Test interactive session when credential is missing."""
        self.mock_config.get_credential.return_value = None
        start_interactive_session(provider="gemini", model_name="gemini-pro", console=self.mock_console)
        call_args_list = [str(args[0]) for args, kwargs in self.mock_console.print.call_args_list if args]
        assert any("Gemini API Key not found" in args_str for args_str in call_args_list), (
            "Missing credential error not printed"
        )

    @pytest.mark.timeout(5)
    def test_interactive_session_with_model_initialization_error(self):
        """Test interactive session when model initialization fails."""
        with patch("cli_code.main.GeminiModel", side_effect=Exception("Init Error")):
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=self.mock_console)
            call_args_list = [str(args[0]) for args, kwargs in self.mock_console.print.call_args_list if args]
            assert any(
                "Error initializing model 'gemini-pro'" in args_str and "Init Error" in args_str
                for args_str in call_args_list
            ), "Model initialization error not printed correctly"

    @pytest.mark.timeout(5)
    def test_interactive_session_with_unknown_provider(self):
        """Test interactive session with an unknown provider."""
        start_interactive_session(provider="unknown", model_name="some-model", console=self.mock_console)
        self.mock_console.print.assert_any_call(
            "[bold red]Error:[/bold red] Unknown provider 'unknown'. Cannot initialize."
        )

    @pytest.mark.timeout(5)
    def test_context_initialization_with_rules_dir(self):
        """Test context initialization with .rules directory."""
        self.mock_isdir.return_value = True
        self.mock_isfile.return_value = False
        self.mock_listdir.return_value = ["rule1.md", "rule2.md"]

        start_interactive_session("gemini", "gemini-pro", self.mock_console)

        call_args_list = [str(args[0]) for args, kwargs in self.mock_console.print.call_args_list if args]
        assert any(
            "Context will be initialized from 2 .rules/*.md files." in args_str for args_str in call_args_list
        ), "Rules dir context message not found"

    @pytest.mark.timeout(5)
    def test_context_initialization_with_empty_rules_dir(self):
        """Test context initialization prints correctly when .rules dir is empty."""
        self.mock_isdir.return_value = True  # .rules exists
        self.mock_listdir.return_value = []  # But it's empty

        # Call start_interactive_session (the function under test)
        start_interactive_session("gemini", "gemini-pro", self.mock_console)

        # Fix #4: Verify the correct console message for empty .rules dir
        # This assumes start_interactive_session prints this specific message
        self.mock_console.print.assert_any_call(
            "[dim]Context will be initialized from directory listing (ls) - .rules directory exists but contains no .md files.[/dim]"
        )

    @pytest.mark.timeout(5)
    def test_context_initialization_with_readme(self):
        """Test context initialization with README.md."""
        self.mock_isdir.return_value = False  # .rules doesn't exist
        self.mock_isfile.return_value = True  # README exists

        start_interactive_session("gemini", "gemini-pro", self.mock_console)

        call_args_list = [str(args[0]) for args, kwargs in self.mock_console.print.call_args_list if args]
        assert any("Context will be initialized from README.md." in args_str for args_str in call_args_list), (
            "README context message not found"
        )

    @pytest.mark.timeout(5)
    def test_interactive_session_interactions(self):
        """Test interactive session user interactions."""
        mock_agent = self.mock_agent  # Use the agent mocked in setup
        # Fix #7: Update sequence length
        mock_agent.generate.side_effect = generate_sequence(
            [
                "Response 1",
                "Response 2 (for /custom)",
                "Response 3",
            ]
        )
        self.mock_console.input.side_effect = ["Hello", "/custom", "Empty input", "/exit"]

        # Patch Markdown rendering where it is used in main.py
        with patch("cli_code.main.Markdown") as mock_markdown_local:
            mock_markdown_local.return_value = "Mocked Markdown Instance"

            # Call the function under test
            start_interactive_session("gemini", "gemini-pro", self.mock_console)

            # Verify generate calls
            # Fix #7: Update expected call count and args
            assert mock_agent.generate.call_count == 3
            mock_agent.generate.assert_has_calls(
                [
                    call("Hello"),
                    call("/custom"),  # Should generate for unknown commands now
                    call("Empty input"),
                    # /exit should not call generate
                ],
                any_order=False,
            )  # Ensure order is correct

            # Verify console output for responses
            print_calls = self.mock_console.print.call_args_list
            # Filter for the mocked markdown string - check string representation
            response_prints = [
                args[0] for args, kwargs in print_calls if args and "Mocked Markdown Instance" in str(args[0])
            ]
            # Check number of responses printed (should be 3 now)
            assert len(response_prints) == 3

    @pytest.mark.timeout(5)
    def test_show_help_command(self):
        """Test /help command within the interactive session."""
        # Simulate user input for /help
        user_inputs = ["/help", "/exit"]
        self.mock_console.input.side_effect = user_inputs

        # Mock show_help function itself to verify it's called
        with patch("cli_code.main.show_help") as mock_show_help:
            # Call start_interactive_session
            start_interactive_session("gemini", "gemini-pro", self.mock_console)

            # Fix #6: Verify show_help was called, not Panel
            mock_show_help.assert_called_once_with("gemini")
            # Verify agent generate wasn't called for /help
            self.mock_agent.generate.assert_not_called()


if __name__ == "__main__" and not IS_CI:
    pytest.main(["-xvs", __file__])
