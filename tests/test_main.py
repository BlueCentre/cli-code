"""
Tests for the main entry point module.
"""

import pytest
import sys

import click
from click.testing import CliRunner

from src.cli_code.main import cli


@pytest.fixture
def mock_console(mocker):
    """Provides a mocked Console object."""
    console_mock = mocker.patch("src.cli_code.main.console")
    # Make sure print method doesn't cause issues
    console_mock.print.return_value = None
    # Ensure input method is mockable
    console_mock.input = mocker.MagicMock()
    return console_mock


@pytest.fixture
def mock_config(mocker):
    """Provides a mocked Config object."""
    mock_config = mocker.patch("src.cli_code.main.config")
    mock_config.get_default_provider.return_value = "gemini"
    mock_config.get_default_model.return_value = "gemini-1.5-pro"
    mock_config.get_credential.return_value = "fake-api-key"
    return mock_config


@pytest.fixture
def cli_runner():
    """Provides a Click CLI test runner."""
    return CliRunner()


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Interactive CLI for the cli-code assistant" in result.output


def test_setup_gemini(cli_runner, mock_config):
    """Test setup command for Gemini provider."""
    result = cli_runner.invoke(cli, ["setup", "--provider", "gemini", "test-api-key"])
    
    assert result.exit_code == 0
    mock_config.set_credential.assert_called_once_with("gemini", "test-api-key")


def test_setup_ollama(cli_runner, mock_config):
    """Test setup command for Ollama provider."""
    result = cli_runner.invoke(cli, ["setup", "--provider", "ollama", "http://localhost:11434"])
    
    assert result.exit_code == 0
    mock_config.set_credential.assert_called_once_with("ollama", "http://localhost:11434")


def test_setup_error(cli_runner, mock_config):
    """Test setup command with an error."""
    mock_config.set_credential.side_effect = Exception("Test error")
    
    result = cli_runner.invoke(cli, ["setup", "--provider", "gemini", "test-api-key"], catch_exceptions=False)
    
    assert result.exit_code == 0
    assert "Error saving API Key" in result.output


def test_set_default_provider(cli_runner, mock_config):
    """Test set-default-provider command."""
    result = cli_runner.invoke(cli, ["set-default-provider", "gemini"])
    
    assert result.exit_code == 0
    mock_config.set_default_provider.assert_called_once_with("gemini")


def test_set_default_provider_error(cli_runner, mock_config):
    """Test set-default-provider command with an error."""
    mock_config.set_default_provider.side_effect = Exception("Test error")
    
    result = cli_runner.invoke(cli, ["set-default-provider", "gemini"])
    
    assert result.exit_code == 0  # Command doesn't exit with error
    assert "Error" in result.output


def test_set_default_model(cli_runner, mock_config):
    """Test set-default-model command."""
    result = cli_runner.invoke(cli, ["set-default-model", "gemini-1.5-pro"])
    
    assert result.exit_code == 0
    mock_config.set_default_model.assert_called_once_with("gemini-1.5-pro", provider="gemini")


def test_set_default_model_with_provider(cli_runner, mock_config):
    """Test set-default-model command with explicit provider."""
    result = cli_runner.invoke(cli, ["set-default-model", "--provider", "ollama", "llama2"])
    
    assert result.exit_code == 0
    mock_config.set_default_model.assert_called_once_with("llama2", provider="ollama")


def test_set_default_model_error(cli_runner, mock_config):
    """Test set-default-model command with an error."""
    mock_config.set_default_model.side_effect = Exception("Test error")
    
    result = cli_runner.invoke(cli, ["set-default-model", "gemini-1.5-pro"])
    
    assert result.exit_code == 0  # Command doesn't exit with error
    assert "Error" in result.output


def test_list_models_gemini(cli_runner, mock_config, mocker):
    """Test list-models command with Gemini provider."""
    # Mock the model classes
    mock_gemini = mocker.patch("src.cli_code.main.GeminiModel")
    
    # Mock model instance with list_models method
    mock_model_instance = mocker.MagicMock()
    mock_model_instance.list_models.return_value = [
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"}
    ]
    mock_gemini.return_value = mock_model_instance
    
    # Invoke the command
    result = cli_runner.invoke(cli, ["list-models"])
    
    assert result.exit_code == 0
    # Verify the model's list_models was called
    mock_model_instance.list_models.assert_called_once()


def test_list_models_ollama(cli_runner, mock_config, mocker):
    """Test list-models command with Ollama provider."""
    # Mock the provider selection
    mock_config.get_default_provider.return_value = "ollama"
    
    # Mock the Ollama model class
    mock_ollama = mocker.patch("src.cli_code.main.OllamaModel")
    
    # Mock model instance with list_models method
    mock_model_instance = mocker.MagicMock()
    mock_model_instance.list_models.return_value = [
        {"id": "llama2", "name": "Llama 2"}
    ]
    mock_ollama.return_value = mock_model_instance
    
    # Invoke the command
    result = cli_runner.invoke(cli, ["list-models"])
    
    assert result.exit_code == 0
    # Verify the model's list_models was called
    mock_model_instance.list_models.assert_called_once()


def test_list_models_error(cli_runner, mock_config, mocker):
    """Test list-models command with an error."""
    # Mock the model classes
    mock_gemini = mocker.patch("src.cli_code.main.GeminiModel")
    
    # Mock model instance with list_models method that raises an exception
    mock_model_instance = mocker.MagicMock()
    mock_model_instance.list_models.side_effect = Exception("Test error")
    mock_gemini.return_value = mock_model_instance
    
    # Invoke the command
    result = cli_runner.invoke(cli, ["list-models"])
    
    assert result.exit_code == 0  # Command doesn't exit with error
    assert "Error" in result.output


def test_cli_invoke_interactive(cli_runner, mock_config, mocker):
    """Test invoking the CLI with no arguments (interactive mode) using mocks."""
    # Mock the start_interactive_session function to prevent hanging
    mock_start_session = mocker.patch("src.cli_code.main.start_interactive_session")
    
    # Run CLI with no command to trigger interactive session
    result = cli_runner.invoke(cli, [])
    
    # Check the result and verify start_interactive_session was called
    assert result.exit_code == 0
    mock_start_session.assert_called_once()


def test_cli_invoke_with_provider_and_model(cli_runner, mock_config, mocker):
    """Test invoking the CLI with provider and model options."""
    # Mock interactive session
    mock_start_session = mocker.patch("src.cli_code.main.start_interactive_session")
    
    # Run CLI with provider and model options
    result = cli_runner.invoke(cli, ["--provider", "gemini", "--model", "gemini-1.5-pro"])
    
    # Verify correct parameters were passed
    assert result.exit_code == 0
    mock_start_session.assert_called_once_with(
        provider="gemini", 
        model_name="gemini-1.5-pro", 
        console=mocker.ANY
    )


def test_cli_no_model_specified(cli_runner, mock_config, mocker):
    """Test CLI behavior when no model is specified."""
    # Mock start_interactive_session
    mock_start_session = mocker.patch("src.cli_code.main.start_interactive_session")
    
    # Make get_default_model return the model
    mock_config.get_default_model.return_value = "gemini-1.5-pro"
    
    result = cli_runner.invoke(cli, [])
    
    assert result.exit_code == 0
    # Verify model was retrieved from config
    mock_config.get_default_model.assert_called_once_with("gemini")
    mock_start_session.assert_called_once_with(
        provider="gemini",
        model_name="gemini-1.5-pro",
        console=mocker.ANY
    )


def test_cli_no_default_model(cli_runner, mock_config, mocker):
    """Test CLI behavior when no default model exists."""
    # Mock the model retrieval to return None
    mock_config.get_default_model.return_value = None
    
    # Run CLI with no arguments
    result = cli_runner.invoke(cli, [])
    
    # Verify appropriate error message and exit
    assert result.exit_code == 1
    assert "No default model configured" in result.stdout


def test_start_interactive_session(mocker, mock_console, mock_config):
    """Test the start_interactive_session function."""
    from src.cli_code.main import start_interactive_session
    
    # Mock model creation and ChatSession
    mock_gemini = mocker.patch("src.cli_code.main.GeminiModel")
    mock_model_instance = mocker.MagicMock()
    mock_gemini.return_value = mock_model_instance
    
    # Mock other functions to avoid side effects
    mocker.patch("src.cli_code.main.show_help")
    
    # Ensure console input raises KeyboardInterrupt to stop the loop
    mock_console.input.side_effect = KeyboardInterrupt()
    
    # Call the function under test
    start_interactive_session(provider="gemini", model_name="gemini-1.5-pro", console=mock_console)
    
    # Verify model was created with correct parameters
    mock_gemini.assert_called_once_with(
        api_key=mock_config.get_credential.return_value,
        console=mock_console,
        model_name="gemini-1.5-pro"
    )
    
    # Verify console input was called (before interrupt)
    mock_console.input.assert_called_once()


def test_start_interactive_session_ollama(mocker, mock_console, mock_config):
    """Test the start_interactive_session function with Ollama provider."""
    from src.cli_code.main import start_interactive_session
    
    # Mock model creation and ChatSession
    mock_ollama = mocker.patch("src.cli_code.main.OllamaModel")
    mock_model_instance = mocker.MagicMock()
    mock_ollama.return_value = mock_model_instance
    
    # Mock other functions to avoid side effects
    mocker.patch("src.cli_code.main.show_help")
    
    # Ensure console input raises KeyboardInterrupt to stop the loop
    mock_console.input.side_effect = KeyboardInterrupt()
    
    # Call the function under test
    start_interactive_session(provider="ollama", model_name="llama2", console=mock_console)
    
    # Verify model was created with correct parameters
    mock_ollama.assert_called_once_with(
        api_url=mock_config.get_credential.return_value,
        console=mock_console,
        model_name="llama2"
    )
    
    # Verify console input was called (before interrupt)
    mock_console.input.assert_called_once()


def test_start_interactive_session_unknown_provider(mocker, mock_console):
    """Test start_interactive_session with unknown provider."""
    from src.cli_code.main import start_interactive_session
    
    # Call with unknown provider - should not raise, but print error
    start_interactive_session(
        provider="unknown", 
        model_name="test-model", 
        console=mock_console
    )
    
    # Assert that environment variable help message is shown 
    mock_console.print.assert_any_call('Or set the environment variable [bold]CLI_CODE_UNKNOWN_API_URL[/bold]')


def test_show_help(mocker, mock_console):
    """Test the show_help function."""
    from src.cli_code.main import show_help
    
    # Call the function
    show_help(provider="gemini")
    
    # Verify console.print was called at least once
    mock_console.print.assert_called()


def test_cli_config_error(cli_runner, mocker):
    """Test CLI behavior when config is None."""
    # Patch config to be None
    mocker.patch("src.cli_code.main.config", None)
    
    # Run CLI
    result = cli_runner.invoke(cli, [])
    
    # Verify error message and exit code
    assert result.exit_code == 1
    assert "Configuration could not be loaded" in result.stdout


def test_setup_config_none(cli_runner, mocker, mock_console):
    """Test setup command when config is None."""
    # Patch config to be None
    mocker.patch("src.cli_code.main.config", None)
    
    # Run setup command
    result = cli_runner.invoke(cli, ["setup", "--provider", "gemini", "test-key"], catch_exceptions=True)
    
    # Verify error message printed via mock_console with actual format
    mock_console.print.assert_any_call("[bold red]Configuration could not be loaded. Cannot proceed.[/bold red]")
    assert result.exit_code != 0 # Command should indicate failure


def test_list_models_no_credential(cli_runner, mock_config):
    """Test list-models command when credential is not found."""
    # Set get_credential to return None
    mock_config.get_credential.return_value = None
    
    # Run list-models command
    result = cli_runner.invoke(cli, ["list-models"])
    
    # Verify error message
    assert "Error" in result.output
    assert "not found" in result.output


def test_list_models_output_format(cli_runner, mock_config, mocker, mock_console):
    """Test the output format of list-models command."""
    mock_gemini = mocker.patch("src.cli_code.main.GeminiModel")
    mock_model_instance = mocker.MagicMock()
    mock_model_instance.list_models.return_value = [
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
        {"id": "gemini-flash", "name": "Gemini Flash"}
    ]
    mock_gemini.return_value = mock_model_instance
    mock_config.get_default_model.return_value = "gemini-1.5-pro" # Set a default

    result = cli_runner.invoke(cli, ["list-models"])

    assert result.exit_code == 0
    # Check fetching message is shown
    mock_console.print.assert_any_call("[yellow]Fetching models for provider 'gemini'...[/yellow]")
    # Check for presence of model info in actual format
    mock_console.print.assert_any_call("\n[bold cyan]Available Gemini Models:[/bold cyan]")


def test_list_models_empty(cli_runner, mock_config, mocker, mock_console):
    """Test list-models when the provider returns an empty list."""
    mock_gemini = mocker.patch("src.cli_code.main.GeminiModel")
    mock_model_instance = mocker.MagicMock()
    mock_model_instance.list_models.return_value = []
    mock_gemini.return_value = mock_model_instance
    mock_config.get_default_model.return_value = None # No default if no models

    result = cli_runner.invoke(cli, ["list-models"])

    assert result.exit_code == 0
    # Check the correct error message with actual wording
    mock_console.print.assert_any_call("[yellow]No models found or reported by provider 'gemini'.[/yellow]")


def test_list_models_unknown_provider(cli_runner, mock_config):
    """Test list-models with an unknown provider via CLI flag."""
    # Need to override the default derived from config
    result = cli_runner.invoke(cli, ["list-models", "--provider", "unknown"])

    # The command itself might exit 0 but print an error
    assert "Unknown provider" in result.output or "Invalid value for '--provider' / '-p'" in result.output


def test_start_interactive_session_config_error(mocker):
    """Test start_interactive_session when config is None."""
    from src.cli_code.main import start_interactive_session
    mocker.patch("src.cli_code.main.config", None)
    mock_console = mocker.MagicMock()

    start_interactive_session("gemini", "test-model", mock_console)

    mock_console.print.assert_any_call("[bold red]Config error.[/bold red]")


def test_start_interactive_session_no_credential(mocker, mock_config, mock_console):
    """Test start_interactive_session when credential is not found."""
    from src.cli_code.main import start_interactive_session
    mock_config.get_credential.return_value = None

    start_interactive_session("gemini", "test-model", mock_console)

    mock_config.get_credential.assert_called_once_with("gemini")
    # Look for message about setting up with actual format
    mock_console.print.assert_any_call('Or set the environment variable [bold]CLI_CODE_GEMINI_API_KEY[/bold]')


def test_start_interactive_session_init_exception(mocker, mock_config, mock_console):
    """Test start_interactive_session when model init raises exception."""
    from src.cli_code.main import start_interactive_session
    mock_gemini = mocker.patch("src.cli_code.main.GeminiModel")
    mock_gemini.side_effect = Exception("Initialization failed")

    start_interactive_session("gemini", "test-model", mock_console)

    # Check for hint about model check
    mock_console.print.assert_any_call("Please check model name, API key permissions, network. Use 'cli-code list-models'.")


def test_start_interactive_session_loop_exit(mocker, mock_config):
    """Test interactive loop handles /exit command."""
    from src.cli_code.main import start_interactive_session
    mock_console = mocker.MagicMock()
    mock_console.input.side_effect = ["/exit"] # Simulate user typing /exit
    mock_model = mocker.patch("src.cli_code.main.GeminiModel").return_value
    mocker.patch("src.cli_code.main.show_help") # Prevent help from running

    start_interactive_session("gemini", "test-model", mock_console)

    mock_console.input.assert_called_once_with("[bold blue]You:[/bold blue] ")
    mock_model.generate.assert_not_called() # Should exit before calling generate


def test_start_interactive_session_loop_unknown_command(mocker, mock_config, mock_console):
    """Test interactive loop handles unknown commands."""
    from src.cli_code.main import start_interactive_session
    # Simulate user typing an unknown command then exiting via interrupt
    mock_console.input.side_effect = ["/unknown", KeyboardInterrupt] 
    mock_model = mocker.patch("src.cli_code.main.GeminiModel").return_value
    # Return None for the generate call
    mock_model.generate.return_value = None
    mocker.patch("src.cli_code.main.show_help")

    start_interactive_session("gemini", "test-model", mock_console)

    # generate is called with the command
    mock_model.generate.assert_called_once_with("/unknown") 
    # Check for unknown command message
    mock_console.print.assert_any_call("[yellow]Unknown command:[/yellow] /unknown")


def test_start_interactive_session_loop_none_response(mocker, mock_config):
    """Test interactive loop handles None response from generate."""
    from src.cli_code.main import start_interactive_session
    mock_console = mocker.MagicMock()
    mock_console.input.side_effect = ["some input", KeyboardInterrupt] # Simulate input then interrupt
    mock_model = mocker.patch("src.cli_code.main.GeminiModel").return_value
    mock_model.generate.return_value = None # Simulate model returning None
    mocker.patch("src.cli_code.main.show_help")

    start_interactive_session("gemini", "test-model", mock_console)

    mock_model.generate.assert_called_once_with("some input")
    # Check for the specific None response message, ignoring other prints
    mock_console.print.assert_any_call("[red]Received an empty response from the model.[/red]")


def test_start_interactive_session_loop_exception(mocker, mock_config, mock_console):
    """Test interactive loop exception handling."""
    from src.cli_code.main import start_interactive_session
    mock_console.input.side_effect = ["some input", KeyboardInterrupt] # Simulate input then interrupt
    mock_model = mocker.patch("src.cli_code.main.GeminiModel").return_value
    mock_model.generate.side_effect = Exception("Generate failed") # Simulate error
    mocker.patch("src.cli_code.main.show_help")

    start_interactive_session("gemini", "test-model", mock_console)

    mock_model.generate.assert_called_once_with("some input")
    # Correct the newline and ensure exact match for the error message
    mock_console.print.assert_any_call("\n[bold red]An error occurred during the session:[/bold red] Generate failed")


def test_setup_ollama_message(cli_runner, mock_config, mock_console):
    """Test setup command shows specific message for Ollama."""
    result = cli_runner.invoke(cli, ["setup", "--provider", "ollama", "http://host:123"])

    assert result.exit_code == 0
    # Check console output via mock with corrected format
    mock_console.print.assert_any_call("[green]✓[/green] Ollama API URL saved.")
    mock_console.print.assert_any_call("[yellow]Note:[/yellow] Ensure your Ollama server is running and accessible at http://host:123")


def test_setup_gemini_message(cli_runner, mock_config, mock_console):
    """Test setup command shows specific message for Gemini."""
    mock_config.get_default_model.return_value = "default-gemini-model"
    result = cli_runner.invoke(cli, ["setup", "--provider", "gemini", "test-key"])

    assert result.exit_code == 0
    # Check console output via mock with corrected format
    mock_console.print.assert_any_call("[green]✓[/green] Gemini API Key saved.")
    mock_console.print.assert_any_call("Default model is currently set to: default-gemini-model")


def test_cli_provider_model_override_config(cli_runner, mock_config, mocker):
    """Test CLI flags override config defaults for interactive session."""
    mock_start_session = mocker.patch("src.cli_code.main.start_interactive_session")
    # Config defaults
    mock_config.get_default_provider.return_value = "ollama"
    mock_config.get_default_model.return_value = "llama2" # Default for ollama

    # Invoke with CLI flags overriding defaults
    result = cli_runner.invoke(cli, ["--provider", "gemini", "--model", "gemini-override"])

    assert result.exit_code == 0
    # Verify start_interactive_session was called with the CLI-provided values
    mock_start_session.assert_called_once_with(
        provider="gemini",
        model_name="gemini-override",
        console=mocker.ANY
    )
    # Ensure config defaults were not used for final model resolution
    mock_config.get_default_model.assert_not_called()


def test_cli_provider_uses_config(cli_runner, mock_config, mocker):
    """Test CLI uses config provider default when no flag is given."""
    mock_start_session = mocker.patch("src.cli_code.main.start_interactive_session")
    # Config defaults
    mock_config.get_default_provider.return_value = "ollama" # This should be used
    mock_config.get_default_model.return_value = "llama2" # Default for ollama

    # Invoke without --provider flag
    result = cli_runner.invoke(cli, ["--model", "some-model"]) # Provide model to avoid default model logic for now

    assert result.exit_code == 0
    # Verify start_interactive_session was called with the config provider
    mock_start_session.assert_called_once_with(
        provider="ollama", # From config
        model_name="some-model", # From CLI
        console=mocker.ANY
    )
    mock_config.get_default_provider.assert_called_once()
    # get_default_model should NOT be called here because model was specified via CLI
    mock_config.get_default_model.assert_not_called()


def test_cli_model_uses_config(cli_runner, mock_config, mocker):
    """Test CLI uses config model default when no flag is given."""
    mock_start_session = mocker.patch("src.cli_code.main.start_interactive_session")
    # Config defaults
    mock_config.get_default_provider.return_value = "gemini"
    mock_config.get_default_model.return_value = "gemini-default-model" # This should be used

    # Invoke without --model flag
    result = cli_runner.invoke(cli, []) # Use default provider and model

    assert result.exit_code == 0
    # Verify start_interactive_session was called with the config defaults
    mock_start_session.assert_called_once_with(
        provider="gemini", # From config
        model_name="gemini-default-model", # From config
        console=mocker.ANY
    )
    mock_config.get_default_provider.assert_called_once()
    # get_default_model SHOULD be called here to resolve the model for the default provider
    mock_config.get_default_model.assert_called_once_with("gemini") 