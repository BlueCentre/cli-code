"""
Comprehensive tests for the main CLI module in src/cli_code/main.py.
Focusing on improving test coverage beyond the basic test_main.py
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path to allow importing cli_code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock, call, ANY

from cli_code.main import cli, start_interactive_session, show_help, log


@pytest.fixture
def mock_config():
    """Fixture to provide a mocked Config object."""
    with patch('cli_code.main.config') as mock_config:
        # Set reasonable default behavior for the config mock
        mock_config.get_default_provider.return_value = "gemini"
        mock_config.get_default_model.return_value = "gemini-1.0-pro"
        mock_config.get_credential.return_value = "fake-api-key"
        yield mock_config


@pytest.fixture
def mock_console():
    """Fixture to provide a mocked Console instance."""
    with patch('cli_code.main.console') as mock_console:
        yield mock_console


class TestInteractiveSession:
    """Tests for the start_interactive_session function."""
    
    def test_session_missing_config(self, mock_console):
        """Test start_interactive_session behavior when config is None."""
        with patch('cli_code.main.config', None):
            # Call the function
            start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
            
            # Verify error message
            mock_console.print.assert_called_with("[bold red]Config error.[/bold red]")
    
    def test_session_missing_credential(self, mock_console, mock_config):
        """Test start_interactive_session when credential is missing."""
        # Set up mock to return None for get_credential
        mock_config.get_credential.return_value = None
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify error messages
        assert mock_console.print.call_count >= 2
        assert any("Error:" in str(args) and "not found" in str(args) 
                  for args, kwargs in mock_console.print.call_args_list)
    
    @patch('cli_code.main.GeminiModel')
    @patch('cli_code.main.time.sleep')  # Mock sleep to speed up test
    def test_session_gemini_initialization(self, mock_sleep, mock_gemini_model, mock_console, mock_config):
        """Test start_interactive_session with Gemini provider initialization."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_gemini_model.return_value = mock_model_instance
        
        # Create a generator function to allow breaking out of the input loop
        user_input_values = ["Hello", KeyboardInterrupt()]
        
        def mock_input_side_effect(prompt):
            value = user_input_values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function with interruption
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify model was initialized
        mock_gemini_model.assert_called_once_with(
            api_key=mock_config.get_credential.return_value,
            console=mock_console,
            model_name="gemini-1.0-pro"
        )
        
        # Verify model generate was called with user input
        mock_model_instance.generate.assert_called_once_with("Hello")
        
        # Verify session end message
        assert any("Session interrupted" in str(args) for args, kwargs in mock_console.print.call_args_list)
    
    @patch('cli_code.main.OllamaModel')
    @patch('cli_code.main.time.sleep')  # Mock sleep to speed up test
    def test_session_ollama_initialization(self, mock_sleep, mock_ollama_model, mock_console, mock_config):
        """Test start_interactive_session with Ollama provider initialization."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_ollama_model.return_value = mock_model_instance
        
        # Create a generator function to allow breaking out of the input loop
        user_input_values = ["/exit"]
        
        def mock_input_side_effect(prompt):
            return user_input_values.pop(0)
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function with exit command
        start_interactive_session(provider="ollama", model_name="llama3", console=mock_console)
        
        # Verify model was initialized
        mock_ollama_model.assert_called_once_with(
            api_url=mock_config.get_credential.return_value,
            console=mock_console,
            model_name="llama3"
        )
        
        # Verify the loop exited cleanly (no generate call)
        mock_model_instance.generate.assert_not_called()


class TestExitHandling:
    """Tests for exit commands handling in start_interactive_session."""
    
    @patch('cli_code.main.GeminiModel')
    @patch('cli_code.main.time.sleep')  # Mock sleep to speed up test
    def test_exit_command(self, mock_sleep, mock_model, mock_console, mock_config):
        """Test /exit command terminates the session."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Set up input with exit command
        mock_console.input.return_value = "/exit"
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify model.generate was not called
        mock_model_instance.generate.assert_not_called()
        
        # Verify exit message
        assert any("Exiting" in str(args) for args, kwargs in mock_console.print.call_args_list)


class TestShowHelp:
    """Tests for the show_help function."""
    
    def test_show_help(self, mock_console):
        """Test show_help displays correct information."""
        # Call the function
        with patch('cli_code.main.AVAILABLE_TOOLS', {'tool1': {'description': 'Test tool'}}):
            show_help("gemini")
            
            # Verify help output
            mock_console.print.assert_called_once()
            help_text = mock_console.print.call_args[0][0]
            
            # Verify basic elements are present
            assert "Interactive Commands:" in help_text
            assert "/exit" in help_text
            assert "/help" in help_text
            assert "Available Tools:" in help_text
            assert "tool1" in help_text


class TestCommandHandling:
    """Tests for command handling in start_interactive_session."""
    
    @patch('cli_code.main.GeminiModel')
    @patch('cli_code.main.show_help')
    def test_help_command(self, mock_show_help, mock_model, mock_console, mock_config):
        """Test /help command shows help and continues the session."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create a generator function for input sequence
        user_input_values = ["/help", "/exit"]
        
        def mock_input_side_effect(prompt):
            return user_input_values.pop(0)
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify help was shown for the correct provider
        mock_show_help.assert_called_once_with("gemini")
        
        # Verify model.generate was not called
        mock_model_instance.generate.assert_not_called()
    
    @patch('cli_code.main.GeminiModel')
    def test_unknown_command(self, mock_model, mock_console, mock_config):
        """Test handling of unknown command."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create a generator function for input sequence
        user_input_values = ["/unknown_command", "/exit"]
        
        def mock_input_side_effect(prompt):
            return user_input_values.pop(0)
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify unknown command message
        assert any("Unknown command" in str(args) for args, kwargs in mock_console.print.call_args_list)
        
        # Verify model.generate was not called
        mock_model_instance.generate.assert_not_called()


class TestErrorHandling:
    """Tests for error handling in start_interactive_session."""
    
    @patch('cli_code.main.GeminiModel')
    def test_model_generation_error(self, mock_model, mock_console, mock_config):
        """Test handling of errors during model generation."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model_instance.generate.side_effect = Exception("Model error")
        mock_model.return_value = mock_model_instance
        
        # Create a generator function for input sequence
        user_input_values = ["Generate something", "/exit"]
        
        def mock_input_side_effect(prompt):
            return user_input_values.pop(0)
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify error handling
        assert any("Error:" in str(args) for args, kwargs in mock_console.print.call_args_list)
        
        # Verify model.generate was called
        mock_model_instance.generate.assert_called_once_with("Generate something")


class TestCliCommands:
    """Tests for CLI commands in main.py."""
    
    @patch('cli_code.main.start_interactive_session')
    def test_cli_with_provider_override(self, mock_start_session, mock_config):
        """Test CLI with provider override."""
        # Set up the test
        runner = pytest.CliRunner()
        
        # Call the CLI with provider override
        result = runner.invoke(cli, ["--provider", "ollama"])
        
        # Verify start_interactive_session was called with correct provider
        mock_start_session.assert_called_once()
        assert mock_start_session.call_args[1]["provider"] == "ollama"
        assert mock_start_session.call_args[1]["model_name"] == mock_config.get_default_model.return_value
    
    @patch('cli_code.main.start_interactive_session')
    def test_cli_with_model_override(self, mock_start_session, mock_config):
        """Test CLI with model override."""
        # Set up the test
        runner = pytest.CliRunner()
        
        # Call the CLI with model override
        result = runner.invoke(cli, ["--model", "custom-model"])
        
        # Verify start_interactive_session was called with correct model
        mock_start_session.assert_called_once()
        assert mock_start_session.call_args[1]["provider"] == mock_config.get_default_provider.return_value
        assert mock_start_session.call_args[1]["model_name"] == "custom-model"
    
    @patch('cli_code.main.Config')
    def test_setup_command_successful(self, mock_config_class, mock_config):
        """Test setup command for successful credential setting."""
        # Set up the test
        runner = pytest.CliRunner()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        
        # Call the setup command
        result = runner.invoke(cli, ["setup", "gemini", "test-api-key"], input="y\n")
        
        # Verify set_credential was called with correct values
        mock_config_instance.set_credential.assert_called_once_with("gemini", "test-api-key")
        
        # Verify the output indicates success
        assert "Setup complete" in result.output
    
    @patch('cli_code.main.Config')
    def test_setup_command_cancelled(self, mock_config_class, mock_config):
        """Test setup command when user cancels."""
        # Set up the test
        runner = pytest.CliRunner()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        
        # Call the setup command with user cancellation
        result = runner.invoke(cli, ["setup", "gemini", "test-api-key"], input="n\n")
        
        # Verify set_credential was not called
        mock_config_instance.set_credential.assert_not_called()
        
        # Verify the output indicates cancellation
        assert "Setup cancelled" in result.output
    
    @patch('cli_code.main.Config')
    def test_set_default_provider_command(self, mock_config_class, mock_config):
        """Test set-default-provider command."""
        # Set up the test
        runner = pytest.CliRunner()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        
        # Call the command
        result = runner.invoke(cli, ["set-default-provider", "ollama"])
        
        # Verify set_default_provider was called
        mock_config_instance.set_default_provider.assert_called_once_with("ollama")
        
        # Verify the output
        assert "Default provider set to: ollama" in result.output
    
    @patch('cli_code.main.Config')
    def test_set_default_model_command(self, mock_config_class, mock_config):
        """Test set-default-model command."""
        # Set up the test
        runner = pytest.CliRunner()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        
        # Call the command
        result = runner.invoke(cli, ["set-default-model", "gemini", "gemini-1.5-flash"])
        
        # Verify set_default_model was called
        mock_config_instance.set_default_model.assert_called_once_with("gemini", "gemini-1.5-flash")
        
        # Verify the output
        assert "Default model for provider gemini set to: gemini-1.5-flash" in result.output
    
    @patch('cli_code.main.Config')
    @patch('cli_code.main.GeminiModel')
    def test_list_models_gemini_command(self, mock_gemini_model, mock_config_class, mock_config):
        """Test list-models command for Gemini provider."""
        # Set up the test
        runner = pytest.CliRunner()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        mock_config_instance.get_credential.return_value = "test-api-key"
        
        # Mock the list_models method
        mock_model_instance = MagicMock()
        mock_model_instance.list_models.return_value = ["gemini-1.5-pro", "gemini-1.5-flash"]
        mock_gemini_model.return_value = mock_model_instance
        
        # Call the command
        result = runner.invoke(cli, ["list-models", "gemini"])
        
        # Verify the GeminiModel was created and list_models was called
        mock_gemini_model.assert_called_once_with(api_key="test-api-key", console=ANY, model_name=None)
        mock_model_instance.list_models.assert_called_once()
        
        # Verify the output
        assert "Available models for provider: gemini" in result.output
        assert "gemini-1.5-pro" in result.output
        assert "gemini-1.5-flash" in result.output


class TestToolCommandHandling:
    """Tests for tool command handling in start_interactive_session."""
    
    @patch('cli_code.main.GeminiModel')
    @patch('cli_code.main.process_tool_command')
    def test_tool_command_processing(self, mock_process_tool, mock_model, mock_console, mock_config):
        """Test processing of tool commands."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_process_tool.return_value = True  # Indicate command was processed
        
        # Create input sequence
        user_input_values = ["/tree", "/exit"]
        
        def mock_input_side_effect(prompt):
            return user_input_values.pop(0)
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify process_tool_command was called with correct arguments
        mock_process_tool.assert_called_once_with("/tree", mock_model_instance, mock_console)
        
        # Verify model.generate was not called (since tool was processed)
        mock_model_instance.generate.assert_not_called()
    
    @patch('cli_code.main.GeminiModel')
    @patch('cli_code.main.process_tool_command')
    def test_non_tool_command_processing(self, mock_process_tool, mock_model, mock_console, mock_config):
        """Test processing of non-tool commands that start with '/'."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_process_tool.return_value = False  # Indicate command was not processed as a tool
        
        # Create input sequence
        user_input_values = ["/not_a_tool", "/exit"]
        
        def mock_input_side_effect(prompt):
            return user_input_values.pop(0)
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify process_tool_command was called
        mock_process_tool.assert_called_once_with("/not_a_tool", mock_model_instance, mock_console)
        
        # Verify unknown command message was printed (since not a tool)
        assert any("Unknown command" in str(args) for args, kwargs in mock_console.print.call_args_list)


class TestSpecialCases:
    """Tests for special cases and edge conditions in main.py."""
    
    @patch('cli_code.main.show_ascii_art')
    @patch('cli_code.main.start_interactive_session')
    def test_cli_with_no_config(self, mock_start_session, mock_show_ascii, mock_config):
        """Test CLI behavior when config is None."""
        # Set config to None for this test
        with patch('cli_code.main.config', None):
            # Set up the test
            runner = pytest.CliRunner()
            
            # Call the CLI
            result = runner.invoke(cli)
            
            # Verify start_interactive_session was not called
            mock_start_session.assert_not_called()
            
            # Verify error message
            assert "Error:" in result.output
            assert "configuration" in result.output.lower()
    
    @patch('cli_code.main.GeminiModel')
    def test_multiline_input(self, mock_model, mock_console, mock_config):
        """Test handling of multiline input in interactive session."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create a generator function for multiline input
        user_input_values = ["```\nMultiline\nInput\n```", "/exit"]
        
        def mock_input_side_effect(prompt):
            return user_input_values.pop(0)
        
        mock_console.input.side_effect = mock_input_side_effect
        
        # Call the function
        start_interactive_session(provider="gemini", model_name="gemini-1.0-pro", console=mock_console)
        
        # Verify model.generate was called with the processed multiline input
        mock_model_instance.generate.assert_called_once()
        # Strip the backticks and check the actual content
        assert "Multiline\nInput" in mock_model_instance.generate.call_args[0][0] 