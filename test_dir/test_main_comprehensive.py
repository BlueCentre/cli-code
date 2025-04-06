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
    
    @pytest.mark.timeout(2)
    def test_cli_with_no_config(self, runner):
        """Test CLI behavior when config is None."""
        with patch('cli_code.main.config', None), \
             patch('cli_code.main.sys.exit') as mock_exit:
            
            runner.invoke(cli)
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.timeout(2)
    def test_cli_with_missing_model(self, runner):
        """Test CLI behavior when model is not specified and no default exists."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.sys.exit') as mock_exit:
            
            mock_config.get_default_provider.return_value = "gemini"
            mock_config.get_default_model.return_value = None
            
            runner.invoke(cli)
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.timeout(2)
    def test_setup_with_no_config(self, runner):
        """Test setup command when config is None."""
        with patch('cli_code.main.config', None):
            result = runner.invoke(cli, ['setup', '--provider', 'gemini', 'fake-api-key'])
            assert result.exit_code == 0
            assert "Config error" in result.output
    
    @pytest.mark.timeout(2)
    def test_setup_with_exception(self, runner, mock_config):
        """Test setup command when an exception occurs."""
        mock_config.set_credential.side_effect = Exception("Failed to save")
        
        result = runner.invoke(cli, ['setup', '--provider', 'gemini', 'fake-api-key'])
        assert result.exit_code == 0
        assert "Error saving API Key" in result.output
    
    @pytest.mark.timeout(2)
    def test_set_default_provider_with_no_config(self, runner):
        """Test set-default-provider command when config is None."""
        with patch('cli_code.main.config', None):
            result = runner.invoke(cli, ['set-default-provider', 'gemini'])
            assert result.exit_code == 0
            assert "Config error" in result.output
    
    @pytest.mark.timeout(2)
    def test_set_default_provider_with_exception(self, runner, mock_config):
        """Test set-default-provider command when an exception occurs."""
        mock_config.set_default_provider.side_effect = Exception("Failed to set provider")
        
        result = runner.invoke(cli, ['set-default-provider', 'gemini'])
        assert result.exit_code == 0
        assert "Error setting default provider" in result.output
    
    @pytest.mark.timeout(2)
    def test_set_default_model_with_no_config(self, runner):
        """Test set-default-model command when config is None."""
        with patch('cli_code.main.config', None):
            result = runner.invoke(cli, ['set-default-model', 'gemini-pro'])
            assert result.exit_code == 0
            assert "Config error" in result.output
    
    @pytest.mark.timeout(2)
    def test_set_default_model_with_exception(self, runner, mock_config):
        """Test set-default-model command when an exception occurs."""
        mock_config.set_default_model.side_effect = Exception("Failed to set model")
        
        result = runner.invoke(cli, ['set-default-model', 'gemini-pro'])
        assert result.exit_code == 0
        assert "Error setting default model" in result.output


# Combined tests for multiple error scenarios to reduce test execution time
class TestCombinedErrorScenarios:
    """Combined tests for various error scenarios to improve test efficiency."""
    
    @pytest.mark.timeout(2)
    def test_commands_with_no_config(self, runner):
        """Test all commands when config is None."""
        with patch('cli_code.main.config', None):
            # Test set-default-provider
            result1 = runner.invoke(cli, ['set-default-provider', 'gemini'])
            assert result1.exit_code == 0
            assert "Config error" in result1.output
            
            # Test set-default-model
            result2 = runner.invoke(cli, ['set-default-model', 'gemini-pro'])
            assert result2.exit_code == 0
            assert "Config error" in result2.output
            
            # Test list-models
            result3 = runner.invoke(cli, ['list-models'])
            assert result3.exit_code == 0
            assert "Config error" in result3.output
    
    @pytest.mark.timeout(2)
    def test_commands_with_exceptions(self, runner, mock_config):
        """Test all commands when exceptions occur."""
        # Setup mock for set-default-provider exception
        mock_config.set_default_provider.side_effect = Exception("Failed to set provider")
        
        result1 = runner.invoke(cli, ['set-default-provider', 'gemini'])
        assert result1.exit_code == 0
        assert "Error setting default provider" in result1.output
        
        # Setup mock for set-default-model exception
        mock_config.set_default_model.side_effect = Exception("Failed to set model")
        
        result2 = runner.invoke(cli, ['set-default-model', 'gemini-pro'])
        assert result2.exit_code == 0
        assert "Error setting default model" in result2.output


class TestListModelsCommand:
    """Tests for the list-models command."""
    
    @pytest.mark.timeout(2)
    def test_list_models_with_no_credential(self, runner):
        """Test list-models command when no credential is found."""
        with patch('cli_code.main.config') as mock_config:
            mock_config.get_default_provider.return_value = "gemini"
            mock_config.get_credential.return_value = None
            
            result = runner.invoke(cli, ['list-models'])
            assert result.exit_code == 0
            assert "Error: Gemini API Key not found" in result.output
    
    @pytest.mark.timeout(2)
    def test_list_models_scenarios(self, runner, mock_config):
        """Test list-models command in various scenarios."""
        with patch('cli_code.main.GeminiModel') as mock_model_class:
            # Scenario 1: Empty results
            mock_instance1 = MagicMock()
            mock_instance1.list_models.return_value = []
            mock_model_class.return_value = mock_instance1
            
            result1 = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result1.exit_code == 0
            assert "No models found" in result1.output
            
            # Scenario 2: None result
            mock_instance2 = MagicMock()
            mock_instance2.list_models.return_value = None
            mock_model_class.return_value = mock_instance2
            
            result2 = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result2.exit_code == 0
            
            # Scenario 3: Exception
            mock_model_class.side_effect = Exception("Connection error")
            
            result3 = runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result3.exit_code == 0
            assert "Error listing models" in result3.output


class TestInteractiveSession:
    """Tests for the interactive session functionality."""
    
    @pytest.mark.timeout(2)
    def test_interactive_session_with_no_config_or_credential(self, mock_console):
        """Test interactive session when config is None or credential is missing."""
        # Test with no config
        with patch('cli_code.main.config', None):
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call("[bold red]Config error.[/bold red]")
        
        # Reset mock
        mock_console.reset_mock()
        
        # Test with no credential
        with patch('cli_code.main.config') as mock_config:
            mock_config.get_credential.return_value = None
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "\n[bold red]Error:[/bold red] Gemini API Key not found."
            )
    
    @pytest.mark.timeout(2)
    def test_interactive_session_initialization_errors(self, mock_console):
        """Test interactive session initialization errors."""
        with patch('cli_code.main.config') as mock_config:
            mock_config.get_credential.return_value = "fake-api-key"
            
            # Test with initialization error
            with patch('cli_code.main.GeminiModel') as mock_model_class:
                mock_model_class.side_effect = Exception("Initialization error")
                
                start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
                mock_console.print.assert_any_call(
                    "\n[bold red]Error initializing model 'gemini-pro':[/bold red] Initialization error"
                )
            
            # Reset mock
            mock_console.reset_mock()
            
            # Test with unknown provider
            start_interactive_session(provider="unknown", model_name="model", console=mock_console)
            mock_console.print.assert_any_call(
                "[bold red]Error:[/bold red] Unknown provider 'unknown'. Cannot initialize."
            )
    
    @pytest.mark.timeout(2)
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.listdir')
    def test_interactive_session_context_initialization(self, mock_listdir, mock_isfile, mock_isdir, mock_console):
        """Test interactive session context initialization from different sources."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class:
            
            mock_config.get_credential.return_value = "fake-api-key"
            mock_model_class.return_value = MagicMock()
            
            # Test with .rules directory
            mock_isdir.return_value = True
            mock_isfile.return_value = False
            mock_listdir.return_value = ["rule1.md", "rule2.md"]
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[dim]Context will be initialized from 2 .rules/*.md files.[/dim]"
            )
            
            # Reset mocks
            mock_console.reset_mock()
            
            # Test with README.md
            mock_isdir.return_value = False
            mock_isfile.return_value = True
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[dim]Context will be initialized from README.md.[/dim]"
            )
            
            # Reset mocks
            mock_console.reset_mock()
            
            # Test with directory listing
            mock_isdir.return_value = False
            mock_isfile.return_value = False
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[dim]Context will be initialized from directory listing (ls).[/dim]"
            )
    
    @pytest.mark.timeout(2)
    def test_interactive_session_commands_and_errors(self, mock_console):
        """Test interactive session commands and error handling."""
        with patch('cli_code.main.config') as mock_config, \
             patch('cli_code.main.GeminiModel') as mock_model_class, \
             patch('cli_code.main.show_help') as mock_show_help:
            
            mock_config.get_credential.return_value = "fake-api-key"
            
            # Test help command
            mock_instance1 = MagicMock()
            mock_instance1.generate.side_effect = [None]
            mock_model_class.return_value = mock_instance1
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_show_help.assert_called_once_with("gemini")
            
            # Reset mocks
            mock_console.reset_mock()
            mock_show_help.reset_mock()
            
            # Test model error (None response)
            mock_instance2 = MagicMock()
            mock_console.input.side_effect = ["normal question", "/exit"]
            mock_instance2.generate.side_effect = [None]
            mock_model_class.return_value = mock_instance2
            
            start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
            mock_console.print.assert_any_call(
                "[red]Received an empty response from the model.[/red]"
            )


class TestExceptionHandling:
    """Tests for exception handling in the interactive session."""
    
    @pytest.mark.timeout(2)
    def test_interactive_session_exceptions(self, mock_console):
        """Test interactive session exception handling."""
        with patch('cli_code.main.config') as mock_config:
            mock_config.get_credential.return_value = "fake-api-key"
            
            # Test KeyboardInterrupt
            with patch('cli_code.main.GeminiModel') as mock_model_class:
                mock_instance = MagicMock()
                mock_console.input.side_effect = KeyboardInterrupt()
                mock_model_class.return_value = mock_instance
                
                start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
                mock_console.print.assert_any_call(
                    "\n[yellow]Session interrupted. Exiting.[/yellow]"
                )
            
            # Reset mock
            mock_console.reset_mock()
            
            # Test general exception
            with patch('cli_code.main.GeminiModel') as mock_model_class:
                mock_instance = MagicMock()
                mock_console.input.side_effect = Exception("Test exception")
                mock_model_class.return_value = mock_instance
                
                start_interactive_session(provider="gemini", model_name="gemini-pro", console=mock_console)
                mock_console.print.assert_any_call(
                    "\n[bold red]An error occurred during the session:[/bold red] Test exception"
                )


class TestMiscFunctionality:
    """Tests for miscellaneous functionality."""
    
    @pytest.mark.timeout(2)
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
    
    @pytest.mark.timeout(2)
    def test_ollama_provider(self, runner, mock_config, mock_console):
        """Test Ollama provider functionality."""
        # Test setup
        result = runner.invoke(cli, ['setup', '--provider', 'ollama', 'http://localhost:11434'])
        assert result.exit_code == 0
        assert "Ollama API URL saved" in result.output
        
        # Test interactive session
        with patch('cli_code.main.OllamaModel') as mock_model_class:
            mock_config.get_credential.return_value = "http://localhost:11434"
            mock_instance = MagicMock()
            mock_model_class.return_value = mock_instance
            
            start_interactive_session(provider="ollama", model_name="llama2", console=mock_console)
            mock_console.print.assert_any_call(
                "[green]Ollama provider initialized successfully.[/green]"
            )
        
        # Test list models
        with patch('cli_code.main.OllamaModel') as mock_ollama_model:
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