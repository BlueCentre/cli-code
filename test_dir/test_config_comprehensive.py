"""
Comprehensive tests for the config module in src/cli_code/config.py.
Focusing on improving test coverage beyond the basic test_config.py
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Add the src directory to the path to allow importing cli_code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from cli_code.config import Config, log


@pytest.fixture
def mock_home():
    """Create a temporary directory to use as home directory."""
    with patch.dict(os.environ, {"HOME": "/mock/home"}, clear=False):
        yield Path("/mock/home")


@pytest.fixture
def config_instance():
    """Provide a minimal Config instance for testing individual methods."""
    with patch.object(Config, "__init__", return_value=None):
        config = Config()
        config.config_dir = Path('/fake/config/dir')
        config.config_file = Path('/fake/config/dir/config.yaml')
        config.config = {}
        yield config


@pytest.fixture
def default_config_data():
    """Return default configuration data."""
    return {
        'google_api_key': 'fake-key',
        'default_provider': 'gemini',
        'default_model': 'gemini-pro',
        'ollama_api_url': 'http://localhost:11434',
        'ollama_default_model': 'llama2',
        'settings': {
            'max_tokens': 1000000,
            'temperature': 0.5
        }
    }


class TestDotEnvLoading:
    """Tests for the _load_dotenv method."""
    
    def test_load_dotenv_file_not_exists(self, config_instance):
        """Test _load_dotenv when .env file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._load_dotenv()
            
            # Verify appropriate logging
            mock_logger.debug.assert_called_once()
            assert "No .env or .env.example file found" in mock_logger.debug.call_args[0][0]
    
    @pytest.mark.parametrize("env_content,expected_vars", [
        ("""
        # This is a comment
        CLI_CODE_GOOGLE_API_KEY=test-key
        CLI_CODE_OLLAMA_API_URL=http://localhost:11434
        """, 
        {"CLI_CODE_GOOGLE_API_KEY": "test-key", "CLI_CODE_OLLAMA_API_URL": "http://localhost:11434"}),
        
        ("""
        CLI_CODE_GOOGLE_API_KEY="quoted-key-value"
        CLI_CODE_OLLAMA_API_URL='quoted-url'
        """,
        {"CLI_CODE_GOOGLE_API_KEY": "quoted-key-value", "CLI_CODE_OLLAMA_API_URL": "quoted-url"}),
        
        ("""
        # Comment line
        
        INVALID_LINE_NO_PREFIX
        CLI_CODE_VALID_KEY=valid-value
        =missing_key
        CLI_CODE_MISSING_VALUE=
        """,
        {"CLI_CODE_VALID_KEY": "valid-value", "CLI_CODE_MISSING_VALUE": ""})
    ])
    def test_load_dotenv_variations(self, config_instance, env_content, expected_vars):
        """Test _load_dotenv with various input formats."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=env_content)), \
             patch.dict(os.environ, {}, clear=False), \
             patch('cli_code.config.log'):
            
            config_instance._load_dotenv()
            
            # Verify environment variables were loaded correctly
            for key, value in expected_vars.items():
                assert os.environ.get(key) == value
    
    def test_load_dotenv_file_read_error(self, config_instance):
        """Test _load_dotenv when there's an error reading the .env file."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', side_effect=Exception("Failed to open file")), \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._load_dotenv()
            
            # Verify error is logged
            mock_logger.warning.assert_called_once()
            assert "Error loading .env file" in mock_logger.warning.call_args[0][0]


class TestConfigErrorHandling:
    """Tests for error handling in the Config class."""
    
    def test_ensure_config_exists_file_creation(self, config_instance):
        """Test _ensure_config_exists creates default file when it doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('yaml.dump') as mock_yaml_dump, \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._ensure_config_exists()
            
            # Verify directory was created
            assert config_instance.config_dir.mkdir.called
            
            # Verify file was opened for writing
            mock_file.assert_called_once_with(config_instance.config_file, 'w')
            
            # Verify yaml.dump was called
            mock_yaml_dump.assert_called_once()
            
            # Verify logging
            mock_logger.info.assert_called_once()
    
    def test_load_config_invalid_yaml(self, config_instance):
        """Test _load_config with invalid YAML file."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid: yaml: content")), \
             patch('yaml.safe_load', side_effect=Exception("YAML parsing error")), \
             patch('cli_code.config.log') as mock_logger:
            
            result = config_instance._load_config()
            
            # Verify error is logged and empty dict is returned
            mock_logger.error.assert_called_once()
            assert result == {}
    
    def test_ensure_config_directory_error(self, config_instance):
        """Test error handling when creating config directory fails."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir', side_effect=Exception("mkdir error")), \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._ensure_config_exists()
            
            # Verify error is logged
            mock_logger.error.assert_called_once()
            assert "Error creating config directory" in mock_logger.error.call_args[0][0]
    
    def test_save_config_file_write_error(self, config_instance):
        """Test _save_config when there's an error writing to the file."""
        with patch('builtins.open', side_effect=Exception("File write error")), \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance.config = {"test": "data"}
            config_instance._save_config()
            
            # Verify error is logged
            mock_logger.error.assert_called_once()
            assert "Error saving config file" in mock_logger.error.call_args[0][0]


class TestMigrationFunctionality:
    """Tests for the migration functionality in Config class."""
    
    def test_migrate_old_keys(self, config_instance):
        """Test migration from old api_keys format to new format."""
        old_config = {
            'api_keys': {
                'google': 'old-key'
            },
            'default_provider': 'gemini'
        }
        
        expected_new_config = {
            'google_api_key': 'old-key',
            'default_provider': 'gemini'
        }
        
        with patch.object(Config, '_save_config') as mock_save:
            config_instance.config = old_config.copy()
            config_instance._migrate_old_keys()
            
            # Verify the config was migrated and saved
            assert config_instance.config == expected_new_config
            mock_save.assert_called_once()


class TestCredentialAndProviderFunctions:
    """Tests for credential, provider, and model getter and setter methods."""
    
    @pytest.mark.parametrize("provider,config_key,config_value,expected", [
        ('gemini', 'google_api_key', 'test-key', 'test-key'),
        ('ollama', 'ollama_api_url', 'test-url', 'test-url'),
        ('unknown', None, None, None),
    ])
    def test_get_credential(self, config_instance, provider, config_key, config_value, expected):
        """Test getting credentials for different providers."""
        if config_key:
            config_instance.config = {config_key: config_value}
        else:
            config_instance.config = {}
            
        with patch('cli_code.config.log'):
            assert config_instance.get_credential(provider) == expected
    
    @pytest.mark.parametrize("provider,expected_key,value", [
        ('gemini', 'google_api_key', 'new-key'),
        ('ollama', 'ollama_api_url', 'new-url'),
    ])
    def test_set_credential_valid_providers(self, config_instance, provider, expected_key, value):
        """Test setting credentials for valid providers."""
        with patch.object(Config, '_save_config') as mock_save:
            config_instance.config = {}
            config_instance.set_credential(provider, value)
            
            assert config_instance.config[expected_key] == value
            mock_save.assert_called_once()
    
    def test_set_credential_unknown_provider(self, config_instance):
        """Test setting credential for unknown provider."""
        with patch.object(Config, '_save_config') as mock_save, \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance.config = {}
            config_instance.set_credential('unknown', 'value')
            
            # Verify error was logged and config not saved
            mock_logger.error.assert_called_once()
            mock_save.assert_not_called()
    
    @pytest.mark.parametrize("config_data,provider,expected", [
        ({'default_provider': 'ollama'}, None, 'ollama'),
        ({}, None, 'gemini'),  # Default when not set
        (None, None, 'gemini'),  # Default when config is None
    ])
    def test_get_default_provider(self, config_instance, config_data, provider, expected):
        """Test getting the default provider under different conditions."""
        config_instance.config = config_data
        assert config_instance.get_default_provider() == expected
    
    @pytest.mark.parametrize("provider,model,config_key", [
        ('gemini', 'new-model', 'default_model'),
        ('ollama', 'new-model', 'ollama_default_model'),
    ])
    def test_set_default_model(self, config_instance, provider, model, config_key):
        """Test setting default model for different providers."""
        with patch.object(Config, '_save_config') as mock_save:
            config_instance.config = {}
            config_instance.set_default_model(model, provider)
            
            assert config_instance.config[config_key] == model
            mock_save.assert_called_once()


class TestSettingFunctions:
    """Tests for setting getter and setter methods."""
    
    @pytest.mark.parametrize("config_data,setting,default,expected", [
        ({'settings': {'max_tokens': 1000}}, 'max_tokens', None, 1000),
        ({'settings': {}}, 'missing', 'default-value', 'default-value'),
        ({}, 'any-setting', 'fallback', 'fallback'),
        (None, 'any-setting', 'fallback', 'fallback'),
    ])
    def test_get_setting(self, config_instance, config_data, setting, default, expected):
        """Test get_setting method with various inputs."""
        config_instance.config = config_data
        assert config_instance.get_setting(setting, default=default) == expected
    
    def test_set_setting(self, config_instance):
        """Test set_setting method."""
        with patch.object(Config, '_save_config') as mock_save:
            # Test with existing settings
            config_instance.config = {'settings': {'existing': 'old'}}
            config_instance.set_setting('new_setting', 'value')
            
            assert config_instance.config['settings']['new_setting'] == 'value'
            assert config_instance.config['settings']['existing'] == 'old'
            
            # Test when settings dict doesn't exist
            config_instance.config = {}
            config_instance.set_setting('another', 'value')
            
            assert config_instance.config['settings']['another'] == 'value'
            
            # Test when config is None
            config_instance.config = None
            config_instance.set_setting('third', 'value')
            
            assert config_instance.config['settings']['third'] == 'value'
            
            # Verify save was called appropriate number of times
            assert mock_save.call_count == 3


class TestConfigInitialization:
    """Tests for the Config class initialization and environment variable handling."""
    
    @pytest.mark.timeout(2)  # Reduce timeout to 2 seconds
    def test_config_init_with_env_vars(self):
        """Test that environment variables are correctly loaded during initialization."""
        test_env = {
            'CLI_CODE_GOOGLE_API_KEY': 'env-google-key',
            'CLI_CODE_DEFAULT_PROVIDER': 'env-provider',
            'CLI_CODE_DEFAULT_MODEL': 'env-model',
            'CLI_CODE_OLLAMA_API_URL': 'env-ollama-url',
            'CLI_CODE_OLLAMA_DEFAULT_MODEL': 'env-ollama-model',
            'CLI_CODE_SETTINGS_MAX_TOKENS': '5000',
            'CLI_CODE_SETTINGS_TEMPERATURE': '0.8'
        }
        
        with patch.dict(os.environ, test_env, clear=False), \
             patch.object(Config, '_load_dotenv'), \
             patch.object(Config, '_ensure_config_exists'), \
             patch.object(Config, '_load_config', return_value={}), \
             patch.object(Config, '_migrate_old_keys'):
            
            config = Config()
            
            # Verify environment variables override config values
            assert config.config.get('google_api_key') == 'env-google-key'
            assert config.config.get('default_provider') == 'env-provider'
            assert config.config.get('default_model') == 'env-model'
            assert config.config.get('ollama_api_url') == 'env-ollama-url'
            assert config.config.get('ollama_default_model') == 'env-ollama-model'
            assert config.config.get('settings', {}).get('max_tokens') == 5000
            assert config.config.get('settings', {}).get('temperature') == 0.8

    @pytest.mark.timeout(2)  # Reduce timeout to 2 seconds
    def test_paths_initialization(self):
        """Test the initialization of paths in Config class."""
        with patch('os.path.expanduser', return_value='/mock/home'), \
             patch.object(Config, '_load_dotenv'), \
             patch.object(Config, '_ensure_config_exists'), \
             patch.object(Config, '_load_config', return_value={}), \
             patch.object(Config, '_migrate_old_keys'):
            
            config = Config()
            
            # Verify paths are correctly initialized
            assert config.config_dir == Path('/mock/home/.config/cli-code')
            assert config.config_file == Path('/mock/home/.config/cli-code/config.yaml')


class TestDotEnvEdgeCases:
    """Test edge cases for the _load_dotenv method."""
    
    @pytest.mark.timeout(2)  # Reduce timeout to 2 seconds
    def test_load_dotenv_with_example_file(self, config_instance):
        """Test _load_dotenv with .env.example file when .env doesn't exist."""
        example_content = """
        # Example configuration
        CLI_CODE_GOOGLE_API_KEY=example-key
        """
        
        with patch('pathlib.Path.exists', side_effect=[False, True]), \
             patch('builtins.open', mock_open(read_data=example_content)), \
             patch.dict(os.environ, {}, clear=False), \
             patch('cli_code.config.log'):
            
            config_instance._load_dotenv()
            
            # Verify environment variables were loaded from example file
            assert os.environ.get('CLI_CODE_GOOGLE_API_KEY') == 'example-key'


# Optimized test that combines several edge cases in one test
class TestEdgeCases:
    """Combined tests for various edge cases."""
    
    @pytest.mark.parametrize("method_name,args,config_state,expected_result,should_log_error", [
        ('get_credential', ('unknown',), {}, None, False),
        ('get_default_provider', (), None, 'gemini', False),
        ('get_default_model', ('gemini',), None, 'gemini-1.5-pro', False),
        ('get_default_model', ('ollama',), None, 'llama2', False), 
        ('get_default_model', ('unknown_provider',), {}, None, False),
        ('get_setting', ('any_setting', 'fallback'), None, 'fallback', False),
        ('get_config_value', ('any_key', 'fallback'), None, 'fallback', False),
    ])
    def test_edge_cases(self, config_instance, method_name, args, config_state, expected_result, should_log_error):
        """Test various edge cases with parametrized inputs."""
        with patch('cli_code.config.log') as mock_logger:
            config_instance.config = config_state
            method = getattr(config_instance, method_name)
            result = method(*args)
            
            assert result == expected_result
            
            if should_log_error:
                assert mock_logger.error.called or mock_logger.warning.called 