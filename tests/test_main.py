"""
Tests for the CLI main module.
"""
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from cli_code.main import cli

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
def mock_config():
    """Fixture to provide a mocked Config object."""
    with patch('cli_code.main.config') as mock_config:
        # Set some reasonable default behavior for the config mock
        mock_config.get_default_provider.return_value = "gemini"
        mock_config.get_default_model.return_value = "gemini-pro"
        mock_config.get_credential.return_value = "fake-api-key"
        yield mock_config


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


@patch('cli_code.main.start_interactive_session')
def test_cli_default_invocation(mock_start_session, runner, mock_config):
    """Test the default CLI invocation starts an interactive session."""
    result = runner.invoke(cli)
    assert result.exit_code == 0
    mock_start_session.assert_called_once()


def test_setup_command(runner, mock_config):
    """Test the setup command."""
    result = runner.invoke(cli, ['setup', '--provider', 'gemini', 'fake-api-key'])
    assert result.exit_code == 0
    mock_config.set_credential.assert_called_once_with('gemini', 'fake-api-key')


def test_set_default_provider(runner, mock_config):
    """Test the set-default-provider command."""
    result = runner.invoke(cli, ['set-default-provider', 'ollama'])
    assert result.exit_code == 0
    mock_config.set_default_provider.assert_called_once_with('ollama')


def test_set_default_model(runner, mock_config):
    """Test the set-default-model command."""
    result = runner.invoke(cli, ['set-default-model', '--provider', 'gemini', 'gemini-pro-vision'])
    assert result.exit_code == 0
    mock_config.set_default_model.assert_called_once_with('gemini-pro-vision', provider='gemini')


@patch('cli_code.main.GeminiModel')
def test_list_models_gemini(mock_gemini_model, runner, mock_config):
    """Test the list-models command for Gemini provider."""
    # Setup mock model instance
    mock_instance = MagicMock()
    mock_instance.list_models.return_value = [
        {"name": "gemini-pro", "displayName": "Gemini Pro"},
        {"name": "gemini-pro-vision", "displayName": "Gemini Pro Vision"}
    ]
    mock_gemini_model.return_value = mock_instance
    
    result = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
    assert result.exit_code == 0
    mock_gemini_model.assert_called_once()
    mock_instance.list_models.assert_called_once()


@patch('cli_code.main.OllamaModel')
def test_list_models_ollama(mock_ollama_model, runner, mock_config):
    """Test the list-models command for Ollama provider."""
    # Setup mock model instance
    mock_instance = MagicMock()
    mock_instance.list_models.return_value = [
        {"name": "llama2", "displayName": "Llama 2"},
        {"name": "mistral", "displayName": "Mistral"}
    ]
    mock_ollama_model.return_value = mock_instance
    
    result = runner.invoke(cli, ['list-models', '--provider', 'ollama'])
    assert result.exit_code == 0
    mock_ollama_model.assert_called_once()
    mock_instance.list_models.assert_called_once()