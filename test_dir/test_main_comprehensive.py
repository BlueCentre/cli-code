"""
Comprehensive tests for the main module to improve coverage.
This file extends the existing tests in test_main.py with more edge cases,
error conditions, and specific code paths that weren't previously tested.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner

# Add the src directory to the path to allow importing cli_code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cli_code.main import cli, start_interactive_session, show_help, PROVIDER_CHOICES, console


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


@pytest.fixture
def mock_console():
    """Fixture to provide a mocked Console object."""
    with patch('cli_code.main.console') as mock_console:
        mock_console.input.side_effect = ["/help", "/exit"]
        yield mock_console


class TestCliErrors:
    """Tests for error handling in the CLI commands."""
    
    def test_cli_with_no_config(self, runner):
        """Test CLI behavior when config is None."""
        with patch('cli_code.main.config', None), \
             patch('cli_code.main.sys.exit') as mock_exit:
            
            runner.invoke(cli)
            mock_exit.assert_called_once_with(1)
    
    def test_cli_with_missing_model(self, runner):
        """Test CLI behavior when model is not specified and no default exists."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.sys.exit') as mock_exit:
            
            mock_config.get_default_provider.return_value = "gemini"
            mock_config.get_default_model.return_value = None
            
            runner.invoke(cli)
            mock_exit.assert_called_once_with(1)
    
    def test_setup_with_no_config(self, runner):
        """Test setup command when config is None."""
        with patch('cli_code.main.config', None):
            result = runner.invoke(cli, ['setup', '--provider', 'gemini', 'fake-api-key'])
            assert result.exit_code == 0
            assert "Config error" in result.output
    
    def test_setup_with_exception(self, runner, mock_config):
        """Test setup command when an exception occurs."""
        mock_config.set_credential.side_effect = Exception("Failed to save")
        
        result = runner.invoke(cli, ['setup', '--provider', 'gemini', 'fake-api-key'])
        assert result.exit_code == 0
        assert "Error saving API Key" in result.output
    
    def test_set_default_provider_with_no_config(self, runner):
        """Test set-default-provider command when config is None."""
        with patch('cli_code.main.config', None):
            result = runner.invoke(cli, ['set-default-provider', 'gemini'])
            assert result.exit_code == 0
            assert "Config error" in result.output
    
    def test_set_default_provider_with_exception(self, runner, mock_config):
        """Test set-default-provider command when an exception occurs."""
        mock_config.set_default_provider.side_effect = Exception("Failed to set provider")
        
        result = runner.invoke(cli, ['set-default-provider', 'gemini'])
        assert result.exit_code == 0
        assert "Error setting default provider" in result.output
    
    def test_set_default_model_with_no_config(self, runner):
        """Test set-default-model command when config is None."""
        with patch('cli_code.main.config', None):
            result = runner.invoke(cli, ['set-default-model', 'gemini-pro'])
            assert result.exit_code == 0
            assert "Config error" in result.output
    
    def test_set_default_model_with_exception(self, runner, mock_config):
        """Test set-default-model command when an exception occurs."""
        mock_config.set_default_model.side_effect = Exception("Failed to set model")
        
        result = runner.invoke(cli, ['set-default-model', 'gemini-pro'])
        assert result.exit_code == 0
        assert "Error setting default model" in result.output


class TestListModelsCommand:
    """Comprehensive tests for the list-models command."""
    
    def test_list_models_with_no_config(self, runner):
        """Test list-models command when config is None."""
        with patch('cli_code.main.config', None):
            result = runner.invoke(cli, ['list-models'])
            assert result.exit_code == 0
            assert "Config error" in result.output
    
    def test_list_models_with_no_credential(self, runner):
        """Test list-models command when no credential is found."""
        with patch('cli_code.main.config') as mock_config:
            mock_config.get_default_provider.return_value = "gemini"
            mock_config.get_credential.return_value = None
            
            result = runner.invoke(cli, ['list-models'])
            assert result.exit_code == 0
            assert "Error: Gemini API Key not found" in result.output
    
    def test_list_models_with_empty_result(self, runner, mock_config):
        """Test list-models command when an empty list is returned."""
        with patch('cli_code.main.GeminiModel') as mock_model_class:
            mock_instance = MagicMock()
            mock_instance.list_models.return_value = []
            mock_model_class.return_value = mock_instance
            
            result = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result.exit_code == 0
            assert "No models found" in result.output
    
    def test_list_models_with_none_result(self, runner, mock_config):
        """Test list-models command when None is returned."""
        with patch('cli_code.main.GeminiModel') as mock_model_class:
            mock_instance = MagicMock()
            mock_instance.list_models.return_value = None
            mock_model_class.return_value = mock_instance
            
            result = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result.exit_code == 0
            # No specific error message needed as the agent's list_models handles it
    
    def test_list_models_with_exception(self, runner, mock_config):
        """Test list-models command when an exception occurs."""
        with patch('cli_code.main.GeminiModel') as mock_model_class:
            mock_model_class.side_effect = Exception("Connection error")
            
            result = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result.exit_code == 0
            assert "Error listing models" in result.output
    
    def test_list_models_with_no_default_model(self, runner, mock_config):
        """Test list-models command when no default model is set."""
        with patch('cli_code.main.GeminiModel') as mock_model_class:
            mock_instance = MagicMock()
            mock_instance.list_models.return_value = [{"id": "model1", "name": "Model 1"}]
            mock_model_class.return_value = mock_instance
            
            mock_config.get_default_model.return_value = None
            
            result = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result.exit_code == 0
            assert "No default model set" in result.output


class TestInteractiveSession:
    """Tests for the interactive session functionality."""
    
    def test_interactive_session_with_no_config(self, mock_console):
        """Test interactive session when config is None."""
        with patch('cli_code.main.config', None):
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call("[bold red]Config error.[/bold red]")
    
    def test_interactive_session_with_no_credential(self, mock_console):
        """Test interactive session when no credential is found."""
        with patch('cli_code.main.config') as mock_config:
            mock_config.get_credential.return_value = None
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "\n[bold red]Error:[/bold red] Gemini API Key not found."
            )
    
    def test_interactive_session_with_initialization_error(self, mock_console):
        """Test interactive session when model initialization fails."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_model_class.side_effect = Exception("Initialization error")
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "\n[bold red]Error initializing model 'gemini-pro':[/bold red] Initialization error"
            )
    
    def test_interactive_session_unknown_provider(self, mock_console):
        """Test interactive session with an unknown provider."""
        with patch('cli_code.main.config') as mock_config:
            mock_config.get_credential.return_value = "fake-api-key"
            
            start_interactive_session(provider="unknown", model_name="model", console=mock_console)
            mock_console.print.assert_any_call(
                "[bold red]Error:[/bold red] Unknown provider 'unknown'. Cannot initialize."
            )
    
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.listdir')
    def test_interactive_session_context_rules_dir(self, mock_listdir, mock_isfile, mock_isdir, mock_console):
        """Test interactive session context initialization with .rules directory."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_model_class.return_value = MagicMock()
            
            mock_isdir.return_value = True
            mock_isfile.return_value = False
            mock_listdir.return_value = ["rule1.md", "rule2.md"]
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[dim]Context will be initialized from 2 .rules/*.md files.[/dim]"
            )
    
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_interactive_session_context_readme(self, mock_isfile, mock_isdir, mock_console):
        """Test interactive session context initialization with README.md."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_model_class.return_value = MagicMock()
            
            mock_isdir.return_value = False
            mock_isfile.return_value = True
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[dim]Context will be initialized from README.md.[/dim]"
            )
    
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_interactive_session_context_ls(self, mock_isfile, mock_isdir, mock_console):
        """Test interactive session context initialization with directory listing."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_model_class.return_value = MagicMock()
            
            mock_isdir.return_value = False
            mock_isfile.return_value = False
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[dim]Context will be initialized from directory listing (ls).[/dim]"
            )
    
    def test_interactive_session_help_command(self, mock_console):
        """Test interactive session with help command."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class, \
             patch('cli_code.main.show_help') as mock_show_help:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_instance = MagicMock()
            # Return None for the first generate call (the /help command)
            mock_instance.generate.side_effect = [None]
            mock_model_class.return_value = mock_instance
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_show_help.assert_called_once_with("gemini")
    
    def test_interactive_session_model_error(self, mock_console):
        """Test interactive session when the model returns a None response."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_instance = MagicMock()
            # Input is not a command but generate still returns None
            mock_console.input.side_effect = ["normal question", "/exit"]
            mock_instance.generate.side_effect = [None]
            mock_model_class.return_value = mock_instance
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[red]Received an empty response from the model.[/red]"
            )
    
    def test_interactive_session_keyboard_interrupt(self, mock_console):
        """Test interactive session with KeyboardInterrupt."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_instance = MagicMock()
            mock_console.input.side_effect = KeyboardInterrupt()
            mock_model_class.return_value = mock_instance
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "\n[yellow]Session interrupted. Exiting.[/yellow]"
            )
    
    def test_interactive_session_general_exception(self, mock_console):
        """Test interactive session with a general exception."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_instance = MagicMock()
            mock_console.input.side_effect = Exception("Test exception")
            mock_model_class.return_value = mock_instance
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "\n[bold red]An error occurred during the session:[/bold red] Test exception"
            )


class TestHelpFunction:
    """Tests for the show_help function."""
    
    def test_show_help(self):
        """Test the show_help function."""
        with patch('cli_code.main.console') as mock_console, \
             patch('cli_code.main.AVAILABLE_TOOLS', {"tool1": None, "tool2": None}):
            
            show_help("gemini")
            mock_console.print.assert_called_once()
            
            # Check if the help content mentions the tools
            call_args = mock_console.print.call_args[0][0]
            assert "tool1" in call_args
            assert "tool2" in call_args
            assert "Interactive Commands" in call_args
            assert "/exit" in call_args
            assert "/help" in call_args


class TestOllamaProvider:
    """Tests specifically for the Ollama provider functionality."""
    
    def test_setup_ollama_provider(self, runner, mock_config):
        """Test setting up the Ollama provider."""
        result = runner.invoke(cli, ['setup', '--provider', 'ollama', 'http://localhost:11434'])
        assert result.exit_code == 0
        assert "Ollama API URL saved" in result.output
        assert "Ensure your Ollama server is running" in result.output
    
    def test_interactive_session_ollama(self, mock_console):
        """Test interactive session with Ollama provider."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.OllamaModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "http://localhost:11434"
            mock_instance = MagicMock()
            mock_model_class.return_value = mock_instance
            
            start_interactive_session(provider="ollama", model_name="llama2", console=mock_console)
            mock_console.print.assert_any_call(
                "[green]Ollama provider initialized successfully.[/green]"
            )
    
    @patch('cli_code.main.OllamaModel')
    def test_list_models_ollama_with_default(self, mock_ollama_model, runner, mock_config):
        """Test list-models command for Ollama with a default model set."""
        mock_instance = MagicMock()
        mock_instance.list_models.return_value = [
            {"id": "llama2", "name": "Llama 2"},
            {"id": "mistral", "name": "Mistral"}
        ]
        mock_ollama_model.return_value = mock_instance
        mock_config.get_default_model.return_value = "llama2"
        
        result = runner.invoke(cli, ['list-models', '--provider', 'ollama'])
        assert result.exit_code == 0
        assert "Current default Ollama model: llama2" in result.output 