"""
Additional tests to improve coverage for cli_code.main module.
This file focuses on areas that aren't well covered by existing tests.
"""

import os
import sys
from unittest import TestCase, mock
from unittest.mock import MagicMock, call, patch

# Import pytest only if available
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy decorator that does nothing if pytest is not available
    class pytest:
        @staticmethod
        def mark(klass=None, **kwargs):
            if klass is None:
                # Used as a decorator factory
                def decorator(f):
                    return f
                return decorator
            return klass
            
        @staticmethod
        def skipif(condition, reason=""):
            def decorator(f):
                return f
            return decorator

# Handle import errors gracefully - this helps with CI compatibility
try:
    from cli_code.config import Config
    from cli_code.main import (
        cli,
        list_models,
        set_default_model,
        set_default_provider,
        setup_command,
        start_interactive_session,
    )
    # Import click testing if available
    try:
        from click.testing import CliRunner
        CLICK_TESTING_AVAILABLE = True
    except ImportError:
        CLICK_TESTING_AVAILABLE = False
        # Create dummy CliRunner for type checking
        class CliRunner:
            def invoke(self, *args, **kwargs):
                return None
                
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    class Config: pass
    class CliRunner: pass
    def cli(): pass
    def set_default_provider(): pass
    def set_default_model(): pass
    def list_models(): pass
    def setup_command(): pass
    def start_interactive_session(): pass

# Create a decorator to skip tests if imports aren't available
skip_if_imports_unavailable = pytest.mark.skipif(
    not IMPORTS_AVAILABLE or not CLICK_TESTING_AVAILABLE,
    reason="Required imports not available"
)

@skip_if_imports_unavailable
class TestMainCLICommands(TestCase):
    """Tests for main CLI commands that weren't well covered previously."""
    
    @patch('cli_code.main.Config')
    def test_setup_command_with_invalid_provider(self, mock_config):
        """Test setup command rejects invalid providers."""
        runner = CliRunner()
        result = runner.invoke(setup_command, ["--provider=invalid", "api-key-123"])
        assert result.exit_code != 0
        assert "Invalid provider" in result.output
        mock_config.return_value.set_credential.assert_not_called()

    @patch('cli_code.main.Config')
    def test_set_default_provider_command_with_invalid_provider(self, mock_config):
        """Test set_default_provider command validates providers."""
        runner = CliRunner()
        result = runner.invoke(set_default_provider, ["invalid"])
        assert result.exit_code != 0
        assert "Invalid provider" in result.output
        mock_config.return_value.set_default_provider.assert_not_called()
        
    @patch('cli_code.main.Config')
    def test_set_default_model_command_validates_provider(self, mock_config):
        """Test set_default_model command validates provider."""
        runner = CliRunner()
        result = runner.invoke(set_default_model, ["--provider=invalid", "model-name"])
        assert result.exit_code != 0
        assert "Invalid provider" in result.output
        mock_config.return_value.set_default_model.assert_not_called()
        
    @patch('cli_code.main.Config')
    @patch('cli_code.main.get_available_models')
    def test_list_models_invalid_provider(self, mock_get_models, mock_config):
        """Test list_models handles invalid provider."""
        runner = CliRunner()
        result = runner.invoke(list_models, ["invalid"])
        assert result.exit_code != 0
        assert "Invalid provider" in result.output
        mock_get_models.assert_not_called()

@skip_if_imports_unavailable
class TestInteractiveSession(TestCase):
    """Tests for interactive session functionality."""
    
    @patch('cli_code.main.get_agent_for_provider')
    @patch('cli_code.main.Config')
    def test_interactive_session_credential_check(self, mock_config, mock_get_agent):
        """Test interactive session checks for credential."""
        # Setup mock to return None for credential check
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_credential.return_value = None
        
        result = start_interactive_session("gemini", "model-id")
        assert result is False
        mock_config_instance.get_credential.assert_called_once()
        mock_get_agent.assert_not_called()
        
    @patch('cli_code.main.get_agent_for_provider')
    @patch('cli_code.main.Config')
    @patch('cli_code.main.rich')
    def test_interactive_session_help_command(self, mock_rich, mock_config, mock_get_agent):
        """Test interactive session handles help command."""
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_credential.return_value = "api-key"
        
        # Setup agent mock
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent
        
        # Setup input mock to return help command then exit
        with patch('builtins.input', side_effect=["/help", "/exit"]):
            start_interactive_session("gemini", "model-id")
            
        # Verify help text was printed
        assert any("help" in str(call_args).lower() for call_args in mock_rich.print.call_args_list)
        # Agent should not be called for help command
        mock_agent.chat.assert_not_called()
        
    @patch('cli_code.main.get_agent_for_provider')
    @patch('cli_code.main.Config')
    @patch('cli_code.main.rich')
    def test_interactive_session_clear_command(self, mock_rich, mock_config, mock_get_agent):
        """Test interactive session handles clear command."""
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_credential.return_value = "api-key"
        
        # Setup agent mock
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent
        
        # Setup input mock to return clear command then exit
        with patch('builtins.input', side_effect=["/clear", "/exit"]):
            start_interactive_session("gemini", "model-id")
            
        # Agent's chat history should be cleared
        mock_agent.clear_history.assert_called_once()
        
    @patch('cli_code.main.get_agent_for_provider')
    @patch('cli_code.main.Config')
    @patch('cli_code.main.rich')
    def test_main_cli_with_args(self, mock_rich, mock_config, mock_get_agent):
        """Test main CLI with command line arguments instead of interactive."""
        # Setup agent mock for interactive mode
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_credential.return_value = "api-key"
        mock_config_instance.get_default_provider.return_value = "gemini"
        mock_config_instance.get_default_model.return_value = "gemini-model"
        
        with patch('sys.argv', ['cli-code-agent', 'Write a hello world program']):
            # We're mocking argv, so we can't use runner.invoke()
            # Instead, patch the actual start_interactive_session function
            with patch('cli_code.main.start_interactive_session') as mock_start:
                mock_start.return_value = True
                cli()
                # Should be called with default provider and model
                mock_start.assert_called_once_with('gemini', 'gemini-model', 
                                                  initial_prompt='Write a hello world program') 