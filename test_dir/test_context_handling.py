"""
Tests for context handling and file processing in interactive sessions.
This file focuses on testing how the CLI code handles context initialization from files.
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
    from cli_code.main import cli, start_interactive_session
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

# Determine if we're running in CI
IN_CI = os.environ.get('CI', 'false').lower() == 'true'
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE or IN_CI


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestDirectoryContext:
    """Test context initialization with different directory structures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.console_patcher = patch('cli_code.main.console')
        self.mock_console = self.console_patcher.start()
        
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.get_credential.return_value = "fake-api-key"
        
        # Add patch for Markdown to prevent errors
        self.markdown_patcher = patch('cli_code.main.Markdown', return_value=MagicMock())
        self.mock_markdown = self.markdown_patcher.start()
        
        # Patch builtins.open to mock file operations
        self.open_patcher = patch('builtins.open', new_callable=mock_open)
        self.mock_open = self.open_patcher.start()
        
        # Patch os.path.exists for file existence checks
        self.exists_patcher = patch('os.path.exists')
        self.mock_exists = self.exists_patcher.start()
        
        # Patch os.walk for directory traversal
        self.walk_patcher = patch('os.walk')
        self.mock_walk = self.walk_patcher.start()
        
        # Patch input
        self.input_patcher = patch('builtins.input')
        self.mock_input = self.input_patcher.start()
        self.mock_input.return_value = "exit"  # Always exit to end the session
    
    def teardown_method(self):
        """Teardown test fixtures."""
        self.console_patcher.stop()
        self.config_patcher.stop()
        self.markdown_patcher.stop()
        self.open_patcher.stop()
        self.exists_patcher.stop()
        self.walk_patcher.stop()
        self.input_patcher.stop()
    
    @pytest.mark.timeout(5)
    def test_initialize_with_empty_directory(self):
        """Test context initialization with an empty directory."""
        # Simulate an empty directory - no files found
        self.mock_walk.return_value = [
            (os.getcwd(), [], [])  # (dirpath, dirnames, filenames)
        ]
        
        # Make os.path.exists return True for directory, False for files
        self.mock_exists.side_effect = lambda path: not path.endswith(('.md', '.txt', '.py'))
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Verify that model was initialized properly
            mock_gemini_model.assert_called_once()
            
            # Verify appropriate message about working with empty directory
            empty_dir_message_shown = any(
                'directory' in str(args[0]).lower() 
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert empty_dir_message_shown, "Should display message about directory context"
    
    @pytest.mark.timeout(5)
    def test_initialize_with_python_project(self):
        """Test context initialization with a Python project structure."""
        # Simulate a Python project with various files
        self.mock_walk.return_value = [
            (os.getcwd(), ['src', 'tests'], ['README.md', 'requirements.txt', 'setup.py']),
            (os.path.join(os.getcwd(), 'src'), [], ['__init__.py', 'main.py', 'utils.py']),
            (os.path.join(os.getcwd(), 'tests'), [], ['test_main.py', 'test_utils.py'])
        ]
        
        # Make all files exist
        self.mock_exists.return_value = True
        
        # Mock file content for README.md
        readme_content = "# Test Project\nA test Python project for unit testing."
        
        # Configure mock_open to return different content based on file path
        self.mock_open.return_value.__enter__.return_value.read.side_effect = lambda: readme_content
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Verify that context includes Python files
            context_message = any(
                'Python' in str(args[0]) or '.py' in str(args[0])
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert context_message, "Should include Python project structure in context"
            
            # Check that README was read
            self.mock_open.assert_any_call('README.md', 'r', encoding='utf-8')
    
    @pytest.mark.timeout(5)
    def test_initialize_with_large_codebase(self):
        """Test context initialization with a large codebase (many files)."""
        # Create a large list of files
        base_dir = os.getcwd()
        src_dir = os.path.join(base_dir, 'src')
        many_files = ['file{}.py'.format(i) for i in range(100)]
        
        self.mock_walk.return_value = [
            (base_dir, ['src'], ['README.md']),
            (src_dir, [], many_files)
        ]
        
        # Make all files exist
        self.mock_exists.return_value = True
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Check for truncation or handling of large file lists
            truncation_message = any(
                'files' in str(args[0]).lower() and ('many' in str(args[0]).lower() or 'truncated' in str(args[0]).lower())
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert truncation_message, "Should indicate handling of many files in the context"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestRulesDirectoryHandling:
    """Test handling of .rules directory during context initialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.console_patcher = patch('cli_code.main.console')
        self.mock_console = self.console_patcher.start()
        
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.get_credential.return_value = "fake-api-key"
        
        # Add patch for Markdown to prevent errors
        self.markdown_patcher = patch('cli_code.main.Markdown', return_value=MagicMock())
        self.mock_markdown = self.markdown_patcher.start()
        
        # Patch builtins.open to mock file operations
        self.open_patcher = patch('builtins.open', new_callable=mock_open)
        self.mock_open = self.open_patcher.start()
        
        # Patch os.path.exists and os.path.isdir for file/directory existence checks
        self.exists_patcher = patch('os.path.exists')
        self.mock_exists = self.exists_patcher.start()
        
        self.isdir_patcher = patch('os.path.isdir')
        self.mock_isdir = self.isdir_patcher.start()
        
        # Patch os.listdir for directory contents
        self.listdir_patcher = patch('os.listdir')
        self.mock_listdir = self.listdir_patcher.start()
        
        # Patch input
        self.input_patcher = patch('builtins.input')
        self.mock_input = self.input_patcher.start()
        self.mock_input.return_value = "exit"  # Always exit to end the session
    
    def teardown_method(self):
        """Teardown test fixtures."""
        self.console_patcher.stop()
        self.config_patcher.stop()
        self.markdown_patcher.stop()
        self.open_patcher.stop()
        self.exists_patcher.stop()
        self.isdir_patcher.stop()
        self.listdir_patcher.stop()
        self.input_patcher.stop()
    
    @pytest.mark.timeout(5)
    def test_rules_directory_processing(self):
        """Test processing of .rules directory with multiple rule files."""
        # Make .rules directory exist
        self.mock_exists.side_effect = lambda path: '.rules' in path
        self.mock_isdir.side_effect = lambda path: '.rules' in path
        
        # Set up rule files in .rules directory
        rule_files = ['rule1.md', 'rule2.md', 'rule3.md']
        self.mock_listdir.return_value = rule_files
        
        # Set up content for rule files
        rule_contents = {
            '.rules/rule1.md': "# Rule 1\nFollow this coding style rule.",
            '.rules/rule2.md': "# Rule 2\nAnother important rule to follow.",
            '.rules/rule3.md': "# Rule 3\nThe final rule in the set."
        }
        
        self.mock_open.return_value.__enter__.return_value.read.side_effect = lambda: rule_contents.get(
            self.mock_open.call_args[0][0], "Default content"
        )
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Verify that .rules directory was processed
            rules_message = any(
                'rules' in str(args[0]).lower() or 'guidelines' in str(args[0]).lower()
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert rules_message, "Should mention rules being loaded"
            
            # Check that rule files were opened
            for rule_file in rule_files:
                rule_path = os.path.join('.rules', rule_file)
                self.mock_open.assert_any_call(rule_path, 'r', encoding='utf-8')
    
    @pytest.mark.timeout(5)
    def test_rules_directory_processing_empty(self):
        """Test processing of empty .rules directory."""
        # Make .rules directory exist but empty
        self.mock_exists.side_effect = lambda path: '.rules' in path
        self.mock_isdir.side_effect = lambda path: '.rules' in path
        self.mock_listdir.return_value = []  # Empty directory
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Verify appropriate message about empty rules directory
            empty_rules_message = any(
                'rules' in str(args[0]).lower() and ('empty' in str(args[0]).lower() or 'no rules' in str(args[0]).lower())
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert empty_rules_message, "Should indicate empty rules directory"


@pytest.mark.skipif(SHOULD_SKIP_TESTS, reason="Required imports not available or running in CI")
class TestFileTypeSupport:
    """Test support for different file types during context initialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.console_patcher = patch('cli_code.main.console')
        self.mock_console = self.console_patcher.start()
        
        self.config_patcher = patch('cli_code.main.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.get_credential.return_value = "fake-api-key"
        
        # Add patch for Markdown to prevent errors
        self.markdown_patcher = patch('cli_code.main.Markdown', return_value=MagicMock())
        self.mock_markdown = self.markdown_patcher.start()
        
        # Patch builtins.open to mock file operations
        self.open_patcher = patch('builtins.open', new_callable=mock_open)
        self.mock_open = self.open_patcher.start()
        
        # Patch os.path.exists
        self.exists_patcher = patch('os.path.exists')
        self.mock_exists = self.exists_patcher.start()
        
        # Patch os.walk
        self.walk_patcher = patch('os.walk')
        self.mock_walk = self.walk_patcher.start()
        
        # Patch input
        self.input_patcher = patch('builtins.input')
        self.mock_input = self.input_patcher.start()
        self.mock_input.return_value = "exit"  # Always exit to end the session
    
    def teardown_method(self):
        """Teardown test fixtures."""
        self.console_patcher.stop()
        self.config_patcher.stop()
        self.markdown_patcher.stop()
        self.open_patcher.stop()
        self.exists_patcher.stop()
        self.walk_patcher.stop()
        self.input_patcher.stop()
    
    @pytest.mark.timeout(5)
    def test_python_file_detection(self):
        """Test detection and handling of Python files."""
        # Set up Python project files
        self.mock_walk.return_value = [
            (os.getcwd(), [], ['main.py', 'utils.py', 'README.md'])
        ]
        
        # Make all files exist
        self.mock_exists.return_value = True
        
        # Mock file content
        file_contents = {
            'main.py': "def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()",
            'utils.py': "def helper_function():\n    return 'Helper'",
            'README.md': "# Test Project\nA simple Python project."
        }
        
        self.mock_open.return_value.__enter__.return_value.read.side_effect = lambda: file_contents.get(
            self.mock_open.call_args[0][0], "Default content"
        )
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Verify Python files were detected
            python_detection = any(
                'Python' in str(args[0]) or '.py' in str(args[0])
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert python_detection, "Should detect Python files"
            
            # Check README was read
            self.mock_open.assert_any_call('README.md', 'r', encoding='utf-8')
    
    @pytest.mark.timeout(5)
    def test_javascript_file_detection(self):
        """Test detection and handling of JavaScript files."""
        # Set up JavaScript project files
        self.mock_walk.return_value = [
            (os.getcwd(), ['src'], ['package.json', 'README.md']),
            (os.path.join(os.getcwd(), 'src'), [], ['index.js', 'app.js'])
        ]
        
        # Make all files exist
        self.mock_exists.return_value = True
        
        # Mock file content
        file_contents = {
            'package.json': '{"name": "test-project", "version": "1.0.0"}',
            'README.md': "# JS Test Project\nA simple JavaScript project.",
            'src/index.js': "console.log('Hello, world!');",
            'src/app.js': "function app() { return 'App'; }"
        }
        
        self.mock_open.return_value.__enter__.return_value.read.side_effect = lambda: file_contents.get(
            self.mock_open.call_args[0][0], "Default content"
        )
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Verify JavaScript files were detected
            js_detection = any(
                'JavaScript' in str(args[0]) or '.js' in str(args[0]) or 'package.json' in str(args[0])
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert js_detection, "Should detect JavaScript files"
            
            # Check package.json was recognized
            package_json_detection = any(
                'package.json' in str(args[0])
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert package_json_detection, "Should recognize package.json as a project file"
    
    @pytest.mark.timeout(5)
    def test_mixed_language_project(self):
        """Test handling of a mixed-language project."""
        # Set up mixed language project files
        self.mock_walk.return_value = [
            (os.getcwd(), ['backend', 'frontend'], ['README.md']),
            (os.path.join(os.getcwd(), 'backend'), [], ['app.py', 'models.py']),
            (os.path.join(os.getcwd(), 'frontend'), [], ['index.html', 'script.js', 'style.css'])
        ]
        
        # Make all files exist
        self.mock_exists.return_value = True
        
        # Mock file content
        file_contents = {
            'README.md': "# Full Stack Project\nA mixed-language project with Python backend and web frontend."
        }
        
        self.mock_open.return_value.__enter__.return_value.read.side_effect = lambda: file_contents.get(
            self.mock_open.call_args[0][0], "Default content"
        )
        
        with patch('cli_code.main.GeminiModel') as mock_gemini_model:
            mock_instance = MagicMock()
            mock_gemini_model.return_value = mock_instance
            
            start_interactive_session(
                provider="gemini", 
                model_name="gemini-pro", 
                console=self.mock_console
            )
            
            # Verify mixed language detection
            mixed_detection = any(
                ('full stack' in str(args[0]).lower() or 'mixed' in str(args[0]).lower() or 
                 ('Python' in str(args[0]) and ('JavaScript' in str(args[0]) or 'HTML' in str(args[0]) or 'CSS' in str(args[0]))))
                for call_args in self.mock_console.print.call_args_list 
                for args in (call_args[0], [])
            )
            assert mixed_detection, "Should detect and report mixed language project"


if __name__ == "__main__":
    unittest.main() 