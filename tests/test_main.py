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