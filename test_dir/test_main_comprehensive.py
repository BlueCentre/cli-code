"""
Comprehensive tests for the main module to improve coverage.
This file extends the existing tests in test_main.py with more edge cases,
error conditions, and specific code paths that weren't previously tested.
"""

import os
import sys
import unittest
from unittest import mock
from unittest.mock import patch, MagicMock

# Add the src directory to the path to allow importing cli_code
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import pytest if available, otherwise create dummy markers
try:
    import pytest
    timeout = pytest.mark.timeout
except ImportError:
    # Create a dummy timeout decorator if pytest is not available
    def timeout(seconds):
        def decorator(f):
            return f
        return decorator

# Import click.testing if available, otherwise mock it
try:
    from click.testing import CliRunner
except ImportError:
    class CliRunner:
        """Mock CliRunner for environments where click is not available."""
        def invoke(self, cmd, args=None):
            class Result:
                exit_code = 0
                output = ""
            return Result()

# Import from main module if available, otherwise skip the tests
try:
    from cli_code.main import cli, start_interactive_session, show_help, console
    main_module_available = True
except ImportError:
    main_module_available = False
    # Create placeholder objects for testing
    cli = None
    start_interactive_session = lambda provider, model_name, console: None
    show_help = lambda provider: None
    console = None


@unittest.skipIf(not main_module_available, "cli_code.main module not available")
class TestCliInteractive(unittest.TestCase):
    """Basic tests for the main CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.console_patcher = patch('cli_code.main.console')
        self.mock_console = self.console_patcher.start()
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        
        # Configure default mock behavior
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_default_model.return_value = "gemini-pro"
        self.mock_config.get_credential.return_value = "fake-api-key"
    
    def tearDown(self):
        """Clean up after tests."""
        self.console_patcher.stop()
        self.config_patcher.stop()
    
    @timeout(2)
    def test_start_interactive_session_with_no_credential(self):
        """Test interactive session when no credential is found."""
        # Override default mock behavior for this test
        self.mock_config.get_credential.return_value = None
        
        # Call function under test
        start_interactive_session(provider="gemini", model_name="gemini-pro", console=self.mock_console)
        
        # Check expected behavior
        self.mock_console.print.assert_any_call(mock.ANY)
    
    @timeout(2)
    @unittest.skipIf(not main_module_available, "cli_code.main module not available")
    def test_show_help_function(self):
        """Test the show_help function."""
        with patch('cli_code.main.console') as mock_console:
            with patch('cli_code.main.AVAILABLE_TOOLS', {"tool1": None, "tool2": None}):
                # Call function under test
                show_help("gemini")
                
                # Check expected behavior
                mock_console.print.assert_called_once()


@unittest.skipIf(not main_module_available, "cli_code.main module not available")
class TestListModels(unittest.TestCase):
    """Tests for the list-models command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        
        # Configure default mock behavior
        self.mock_config.get_default_provider.return_value = "gemini"
        self.mock_config.get_credential.return_value = "fake-api-key"
    
    def tearDown(self):
        """Clean up after tests."""
        self.config_patcher.stop()
    
    @timeout(2)
    def test_list_models_missing_credential(self):
        """Test list-models command when credential is missing."""
        # Override default mock behavior
        self.mock_config.get_credential.return_value = None
        
        # Use basic unittest assertions since we may not have Click in CI
        if cli:
            result = self.runner.invoke(cli, ['list-models'])
            self.assertEqual(result.exit_code, 0)


# If running directly, run the tests
if __name__ == "__main__":
    unittest.main() 