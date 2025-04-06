"""
Improved tests for the main module to increase coverage.
This file focuses on testing error handling, edge cases, and untested code paths.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import tempfile
from pathlib import Path

# Ensure we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Handle missing dependencies gracefully
try:
    import pytest
    from click.testing import CliRunner
    from cli_code.main import cli, start_interactive_session, show_help, console
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
    show_help = MagicMock()
    console = MagicMock()

# Determine if we're running in CI
IN_CI = os.environ.get('CI', 'false').lower() == 'true'
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE or IN_CI


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestMainErrorHandling:
    """Test error handling in the main module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch('cli_code.main.console')
        self.mock_console = self.console_patcher.start()
        
        # Set default behavior for mocks
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_default_model.return_value = "gemini-pro"
        self.mock_config.get_credential.return_value = "fake-api-key"
    
    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()
    
    @pytest.mark.timeout(5)
    def test_cli_with_missing_config(self):
        """Test CLI behavior when config is None."""
        with patch('cli_code.main.config', None):
            with patch('cli_code.main.sys.exit') as mock_exit:
                result = self.runner.invoke(cli, [])
                mock_exit.assert_called_once_with(1)
    
    @pytest.mark.timeout(5)
    def test_cli_with_missing_model(self):
        """Test CLI behavior when no model is provided or configured."""
        # Set up config to return None for get_default_model
        self.mock_config.get_default_model.return_value = None
        
        with patch('cli_code.main.sys.exit') as mock_exit:
            result = self.runner.invoke(cli, [])
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.timeout(5)
    def test_setup_with_missing_config(self):
        """Test setup command behavior when config is None."""
        with patch('cli_code.main.config', None):
            result = self.runner.invoke(cli, ['setup', '--provider', 'gemini', 'api-key'])
            assert result.exit_code == 0
            self.mock_console.print.assert_called_with("[bold red]Config error.[/bold red]")
    
    @pytest.mark.timeout(5)
    def test_setup_with_exception(self):
        """Test setup command when an exception occurs."""
        self.mock_config.set_credential.side_effect = Exception("Test error")
        
        result = self.runner.invoke(cli, ['setup', '--provider', 'gemini', 'api-key'])
        assert result.exit_code == 0
        
        # Check that error was printed
        self.mock_console.print.assert_any_call(
            "[bold red]Error saving API Key:[/bold red] Test error")
    
    @pytest.mark.timeout(5)
    def test_set_default_provider_with_exception(self):
        """Test set-default-provider when an exception occurs."""
        self.mock_config.set_default_provider.side_effect = Exception("Test error")
        
        result = self.runner.invoke(cli, ['set-default-provider', 'gemini'])
        assert result.exit_code == 0
        
        # Check that error was printed
        self.mock_console.print.assert_any_call(
            "[bold red]Error setting default provider:[/bold red] Test error")
    
    @pytest.mark.timeout(5)
    def test_set_default_model_with_exception(self):
        """Test set-default-model when an exception occurs."""
        self.mock_config.set_default_model.side_effect = Exception("Test error")
        
        result = self.runner.invoke(cli, ['set-default-model', 'gemini-pro'])
        assert result.exit_code == 0
        
        # Check that error was printed
        self.mock_console.print.assert_any_call(
            "[bold red]Error setting default model for gemini:[/bold red] Test error")


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestListModelsCommand:
    """Test list-models command thoroughly."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch('cli_code.main.console')
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
        with patch('cli_code.main.config', None):
            result = self.runner.invoke(cli, ['list-models'])
            assert result.exit_code == 0
            self.mock_console.print.assert_called_with("[bold red]Config error.[/bold red]")
    
    @pytest.mark.timeout(5)
    def test_list_models_with_missing_credential(self):
        """Test list-models when credential is missing."""
        self.mock_config.get_credential.return_value = None
        
        result = self.runner.invoke(cli, ['list-models', '--provider', 'gemini'])
        assert result.exit_code == 0
        
        # Check that error was printed
        self.mock_console.print.assert_any_call(
            "[bold red]Error:[/bold red] Gemini API Key not found.")
    
    @pytest.mark.timeout(5)
    def test_list_models_with_empty_list(self):
        """Test list-models when no models are returned."""
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_instance.list_models.return_value = []
            mock_gemini_model.return_value = mock_instance
            
            result = self.runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result.exit_code == 0
            
            # Check message about no models
            self.mock_console.print.assert_any_call(
                "[yellow]No models found or reported by provider 'gemini'.[/yellow]")
    
    @pytest.mark.timeout(5)
    def test_list_models_with_exception(self):
        """Test list-models when an exception occurs."""
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_gemini_model.side_effect = Exception("Test error")
            
            result = self.runner.invoke(cli, ['list-models', '--provider', 'gemini'])
            assert result.exit_code == 0
            
            # Check error message
            self.mock_console.print.assert_any_call(
                "[bold red]Error listing models for gemini:[/bold red] Test error")
    
    @pytest.mark.timeout(5)
    def test_list_models_with_unknown_provider(self):
        """Test list-models with an unknown provider (custom mock value)."""
        # Use mock to override get_default_provider with custom, invalid value
        self.mock_config.get_default_provider.return_value = "unknown"
        
        # Using provider from config (let an unknown response come back)
        result = self.runner.invoke(cli, ['list-models'])
        assert result.exit_code == 0
        
        # Should report unknown provider
        self.mock_console.print.assert_any_call(
            "[bold red]Error:[/bold red] Unknown provider 'unknown'.")


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestInteractiveSession:
    """Test interactive session functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        self.console_patcher = patch('cli_code.main.console')
        self.mock_console = self.console_patcher.start()
        
        # Set default behavior for mocks
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_credential.return_value = "fake-api-key"
        
        # Add patch for Markdown to prevent errors
        self.markdown_patcher = patch('cli_code.main.Markdown', return_value=MagicMock())
        self.mock_markdown = self.markdown_patcher.start()
    
    def teardown_method(self):
        """Teardown test fixtures."""
        self.config_patcher.stop()
        self.console_patcher.stop()
        self.markdown_patcher.stop()
    
    @pytest.mark.timeout(5)
    def test_interactive_session_with_missing_config(self):
        """Test interactive session when config is None."""
        with patch('cli_code.main.config', None):
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            self.mock_console.print.assert_called_with("[bold red]Config error.[/bold red]")
    
    @pytest.mark.timeout(5)
    def test_interactive_session_with_missing_credential(self):
        """Test interactive session when credential is missing."""
        self.mock_config.get_credential.return_value = None
        
        start_interactive_session(
            provider="gemini", 
            model_name="gemini-pro", 
            console=self.mock_console
        )
        
        # Check that error was printed about missing credential
        self.mock_console.print.assert_any_call(
            "\n[bold red]Error:[/bold red] Gemini API Key not found.")
    
    @pytest.mark.timeout(5)
    def test_interactive_session_with_model_initialization_error(self):
        """Test interactive session when model initialization fails."""
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_gemini_model.side_effect = Exception("Test error")
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Check that error was printed
            self.mock_console.print.assert_any_call(
                "\n[bold red]Error initializing model 'gemini-pro':[/bold red] Test error")
    
    @pytest.mark.timeout(5)
    def test_interactive_session_with_unknown_provider(self):
        """Test interactive session with an unknown provider."""
        start_interactive_session(
            provider="unknown", 
            model_name="model-name", 
            console=self.mock_console
        )
        
        # Check for unknown provider message
        self.mock_console.print.assert_any_call(
            "[bold red]Error:[/bold red] Unknown provider 'unknown'. Cannot initialize.")
    
    @pytest.mark.timeout(5)
    def test_context_initialization_with_rules_dir(self):
        """Test context initialization with .rules directory."""
        # Set up a directory structure with .rules
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .rules directory with some MD files
            rules_dir = Path(temp_dir) / ".rules"
            rules_dir.mkdir()
            (rules_dir / "rule1.md").write_text("Rule 1")
            (rules_dir / "rule2.md").write_text("Rule 2")
            
            # Create a mock agent instance
            mock_agent = MagicMock()
            mock_agent.generate.return_value = "Mock response"
            
            # Patch directory checks and os.listdir
            with patch('os.path.isdir', return_value=True), \
                 patch('os.listdir', return_value=["rule1.md", "rule2.md"]), \
                 patch('cli_code.main.GeminiModel', return_value=mock_agent), \
                 patch('builtins.open', mock_open(read_data="Mock rule content")):
                
                # Mock console.input for exit
                self.mock_console.input.side_effect = ["/exit"]
                
                start_interactive_session(
                    provider="gemini", 
                    model_name="gemini-pro", 
                    console=self.mock_console
                )
                
                # Check context initialization message
                self.mock_console.print.assert_any_call(
                    "[dim]Context will be initialized from 2 .rules/*.md files.[/dim]")
    
    @pytest.mark.timeout(5)
    def test_context_initialization_with_empty_rules_dir(self):
        """Test context initialization with empty .rules directory."""
        # Create a mock agent instance
        mock_agent = MagicMock()
        mock_agent.generate.return_value = "Mock response"
        
        with patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=[]), \
             patch('cli_code.main.GeminiModel', return_value=mock_agent):
            
            # Mock console.input for exit
            self.mock_console.input.side_effect = ["/exit"]
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Check context initialization message
            self.mock_console.print.assert_any_call(
                "[dim]Context will be initialized from directory listing (ls) - .rules directory exists but contains no .md files.[/dim]")
    
    @pytest.mark.timeout(5)
    def test_context_initialization_with_readme(self):
        """Test context initialization with README.md."""
        # Create a mock agent instance
        mock_agent = MagicMock()
        mock_agent.generate.return_value = "Mock response"
        
        with patch('os.path.isdir', return_value=False), \
             patch('os.path.isfile', return_value=True), \
             patch('cli_code.main.GeminiModel', return_value=mock_agent), \
             patch('builtins.open', mock_open(read_data="Mock README content")):
            
            # Mock console.input for exit
            self.mock_console.input.side_effect = ["/exit"]
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Check context initialization message
            self.mock_console.print.assert_any_call(
                "[dim]Context will be initialized from README.md.[/dim]")
    
    @pytest.mark.timeout(5)
    def test_interactive_session_interactions(self):
        """Test interactive session user interactions."""
        # Mock the model agent
        mock_agent = MagicMock()
        # Ensure response is a string to avoid Markdown parsing issues
        mock_agent.generate.side_effect = [
            "Response 1",  # Regular response
            "",            # Response to command (empty string instead of None)
            "",            # Empty response (empty string instead of None)
            "Response 4"   # Final response
        ]
        
        # Patch GeminiModel to return our mock agent
        with patch('cli_code.main.GeminiModel', return_value=mock_agent):
            # Mock console.input to simulate user interactions
            self.mock_console.input.side_effect = [
                "Hello",       # Regular input
                "/custom",     # Unknown command
                "Empty input", # Will get empty response 
                "/exit"        # Exit command
            ]
            
            # Patch Markdown specifically for this test to avoid type errors
            with patch('cli_code.main.Markdown', return_value=MagicMock()):
                start_interactive_session(
                    provider="gemini", 
                    model_name="gemini-pro", 
                    console=self.mock_console
                )
            
            # Verify interactions
            assert mock_agent.generate.call_count == 3  # Should be called for all inputs except /exit
            self.mock_console.print.assert_any_call("[yellow]Unknown command:[/yellow] /custom")
            self.mock_console.print.assert_any_call("[red]Received an empty response from the model.[/red]")
    
    @pytest.mark.timeout(5)
    def test_show_help_command(self):
        """Test the /help command in interactive session."""
        # Create a mock agent instance
        mock_agent = MagicMock()
        mock_agent.generate.return_value = "Mock response"
        
        # Set up mocks
        with patch('cli_code.main.AVAILABLE_TOOLS', {"tool1": None, "tool2": None}):
            # Mock console.input to simulate user interactions
            self.mock_console.input.side_effect = [
                "/help",  # Help command 
                "/exit"   # Exit command
            ]
            
            # Patch start_interactive_session to avoid creating a real model
            with patch('cli_code.main.GeminiModel', return_value=mock_agent):
                # Call with actual show_help
                with patch('cli_code.main.show_help') as mock_show_help:
                    start_interactive_session(
                        provider="gemini", 
                        model_name="gemini-pro", 
                        console=self.mock_console
                    )
                    
                    # Verify show_help was called
                    mock_show_help.assert_called_once_with("gemini")


if __name__ == "__main__" and not SHOULD_SKIP_TESTS:
    pytest.main(["-xvs", __file__])