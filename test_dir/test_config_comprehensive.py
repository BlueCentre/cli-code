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
    with tempfile.TemporaryDirectory() as temp_dir:
        original_home = os.environ.get("HOME")
        os.environ["HOME"] = temp_dir
        yield Path(temp_dir)
        if original_home:
            os.environ["HOME"] = original_home
        else:
            del os.environ["HOME"]


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
    
    def test_load_dotenv_valid_env_file(self, config_instance):
        """Test _load_dotenv with a valid .env file."""
        env_content = """
        # This is a comment
        CLI_CODE_GOOGLE_API_KEY=test-key
        CLI_CODE_OLLAMA_API_URL=http://localhost:11434
        """
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=env_content)), \
             patch.dict(os.environ, {}, clear=True), \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._load_dotenv()
            
            # Verify environment variables were loaded
            assert os.environ.get('CLI_CODE_GOOGLE_API_KEY') == 'test-key'
            assert os.environ.get('CLI_CODE_OLLAMA_API_URL') == 'http://localhost:11434'
            
            # Verify logging
            mock_logger.info.assert_called()
    
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
        mock_file_data = {}
        
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('cli_code.config.yaml.dump') as mock_yaml_dump, \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._ensure_config_exists()
            
            # Verify directory was created
            assert config_instance.config_dir.mkdir.called
            
            # Verify file was opened for writing
            mock_file.assert_called_once_with(config_instance.config_file, 'w')
            
            # Verify yaml.dump was called to write default config
            mock_yaml_dump.assert_called_once()
            
            # Verify logging
            mock_logger.info.assert_called_once()
            assert "Created default config file" in mock_logger.info.call_args[0][0]
    
    def test_load_config_invalid_yaml(self, config_instance):
        """Test _load_config with invalid YAML file."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid: yaml: content")), \
             patch('cli_code.config.yaml.safe_load', side_effect=Exception("YAML parsing error")), \
             patch('cli_code.config.log') as mock_logger:
            
            result = config_instance._load_config()
            
            # Verify error is logged and empty dict is returned
            mock_logger.error.assert_called_once()
            assert "Error loading config file" in mock_logger.error.call_args[0][0]
            assert result == {}
    
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
        
        with patch.object(Config, '_save_config') as mock_save, \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance.config = old_config.copy()
            config_instance._migrate_old_keys()
            
            # Verify the config was migrated
            assert 'api_keys' not in config_instance.config
            assert config_instance.config.get('google_api_key') == 'old-key'
            assert config_instance.config == expected_new_config
            
            # Verify the config was saved
            mock_save.assert_called_once()
            
            # Verify logging occurred
            assert mock_logger.info.called


class TestCredentialFunctions:
    """Tests for credential getter and setter methods."""
    
    def test_get_credential_gemini(self, config_instance, default_config_data):
        """Test getting credentials for Gemini provider."""
        config_instance.config = default_config_data.copy()
        
        # Test getting credential for Gemini
        assert config_instance.get_credential('gemini') == 'fake-key'
    
    def test_get_credential_ollama(self, config_instance, default_config_data):
        """Test getting credentials for Ollama provider."""
        config_instance.config = default_config_data.copy()
        
        # Test getting credential for Ollama
        assert config_instance.get_credential('ollama') == 'http://localhost:11434'
    
    def test_get_credential_unknown_provider(self, config_instance, default_config_data):
        """Test getting credentials for an unknown provider."""
        with patch('cli_code.config.log') as mock_logger:
            config_instance.config = default_config_data.copy()
            
            # Test getting credential for unknown provider
            assert config_instance.get_credential('unknown') is None
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "unknown provider" in mock_logger.warning.call_args[0][0]
    
    def test_set_credential(self, config_instance):
        """Test setting credentials for different providers."""
        with patch.object(Config, '_save_config') as mock_save:
            config_instance.config = {}
            
            # Test setting credential for Gemini
            config_instance.set_credential('gemini', 'new-key')
            assert config_instance.config['google_api_key'] == 'new-key'
            
            # Test setting credential for Ollama
            config_instance.set_credential('ollama', 'new-url')
            assert config_instance.config['ollama_api_url'] == 'new-url'
            
            # Test setting credential for unknown provider
            with patch('cli_code.config.log') as mock_logger:
                config_instance.set_credential('unknown', 'value')
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                assert "unknown provider" in mock_logger.error.call_args[0][0]
            
            # Verify save was called for each valid provider
            assert mock_save.call_count == 2


class TestProviderAndModelFunctions:
    """Tests for provider and model getter and setter methods."""
    
    def test_get_default_provider(self, config_instance):
        """Test getting the default provider."""
        # When provider is set
        config_instance.config = {'default_provider': 'ollama'}
        assert config_instance.get_default_provider() == 'ollama'
        
        # When provider is not set
        config_instance.config = {}
        assert config_instance.get_default_provider() == 'gemini'  # Default
    
    def test_set_default_provider(self, config_instance):
        """Test setting the default provider."""
        with patch.object(Config, '_save_config') as mock_save:
            config_instance.config = {}
            
            # Test setting valid provider
            config_instance.set_default_provider('ollama')
            assert config_instance.config['default_provider'] == 'ollama'
            mock_save.assert_called_once()
            
            # Test setting invalid provider
            with patch('cli_code.config.log') as mock_logger:
                config_instance.set_default_provider('invalid')
                
                # Should log error and not change config
                mock_logger.error.assert_called_once()
                assert "unknown default provider" in mock_logger.error.call_args[0][0]
    
    def test_get_default_model(self, config_instance):
        """Test getting the default model."""
        # Set up test config
        config_instance.config = {
            'default_model': 'gemini-test-model',
            'ollama_default_model': 'ollama-test-model',
            'default_provider': 'gemini'
        }
        
        # Test getting model for gemini (default provider)
        assert config_instance.get_default_model() == 'gemini-test-model'
        
        # Test getting model for gemini (explicit provider)
        assert config_instance.get_default_model('gemini') == 'gemini-test-model'
        
        # Test getting model for ollama
        assert config_instance.get_default_model('ollama') == 'ollama-test-model'
        
        # Test fallback to hardcoded default for gemini when config is missing
        config_instance.config = {}
        assert 'gemini' in config_instance.get_default_model()
    
    def test_set_default_model(self, config_instance):
        """Test setting the default model."""
        with patch.object(Config, '_save_config') as mock_save:
            config_instance.config = {'default_provider': 'gemini'}
            
            # Test setting model for gemini (default provider)
            config_instance.set_default_model('new-gemini-model')
            assert config_instance.config['default_model'] == 'new-gemini-model'
            
            # Test setting model for ollama
            config_instance.set_default_model('new-ollama-model', 'ollama')
            assert config_instance.config['ollama_default_model'] == 'new-ollama-model'
            
            # Test setting model for unknown provider
            with patch('cli_code.config.log') as mock_logger:
                config_instance.set_default_model('model', 'unknown')
                
                # Should log error
                mock_logger.error.assert_called_once()
                assert "unknown provider" in mock_logger.error.call_args[0][0]
            
            # Verify save was called for each valid setting
            assert mock_save.call_count == 2


class TestSettingFunctions:
    """Tests for setting getter and setter methods."""
    
    def test_get_setting(self, config_instance):
        """Test get_setting method."""
        config_instance.config = {
            'settings': {
                'max_tokens': 1000,
                'temperature': 0.7
            }
        }
        
        # Test getting existing settings
        assert config_instance.get_setting('max_tokens') == 1000
        assert config_instance.get_setting('temperature') == 0.7
        
        # Test getting non-existent setting with default
        assert config_instance.get_setting('non_existent', default=42) == 42
        
        # Test getting setting when settings dict doesn't exist
        config_instance.config = {}
        assert config_instance.get_setting('anything', default='fallback') == 'fallback'
    
    def test_set_setting(self, config_instance):
        """Test set_setting method."""
        config_instance.config = {}
        
        with patch.object(Config, '_save_config') as mock_save:
            # Test setting when settings dict doesn't exist
            config_instance.set_setting('new_setting', 'new_value')
            assert config_instance.config.get('settings', {}).get('new_setting') == 'new_value'
            
            # Test updating existing setting
            config_instance.set_setting('new_setting', 'updated_value')
            assert config_instance.config.get('settings', {}).get('new_setting') == 'updated_value'
            
            # Verify config was saved
            assert mock_save.call_count == 2


# Add new test classes for uncovered areas

class TestConfigInitialization:
    """Tests for the Config class initialization and environment variable handling."""
    
    @pytest.mark.timeout(5)  # Add timeout to prevent hanging
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
        
        # Store original env vars
        original_env = {}
        for key in test_env:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            # Set test env vars without clearing everything
            for key, value in test_env.items():
                os.environ[key] = value
            
            with patch.object(Config, '_load_dotenv'), \
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
        finally:
            # Clean up environment
            for key in test_env:
                if key in original_env:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)

    @pytest.mark.timeout(5)  # Add timeout to prevent hanging
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
    
    @pytest.mark.timeout(5)  # Add timeout
    def test_load_dotenv_with_example_file(self, config_instance):
        """Test _load_dotenv with .env.example file when .env doesn't exist."""
        example_content = """
        # Example configuration
        CLI_CODE_GOOGLE_API_KEY=example-key
        """
        
        # Store original env vars
        original_value = os.environ.get('CLI_CODE_GOOGLE_API_KEY')
        
        try:
            if 'CLI_CODE_GOOGLE_API_KEY' in os.environ:
                del os.environ['CLI_CODE_GOOGLE_API_KEY']
            
            with patch('pathlib.Path.exists', side_effect=[False, True]), \
                 patch('builtins.open', mock_open(read_data=example_content)), \
                 patch('cli_code.config.log') as mock_logger:
                
                config_instance._load_dotenv()
                
                # Verify environment variables were loaded from example file
                assert os.environ.get('CLI_CODE_GOOGLE_API_KEY') == 'example-key'
                
                # Verify appropriate logging
                mock_logger.info.assert_called_once()
                assert "Loading environment from" in mock_logger.info.call_args[0][0]
        finally:
            # Restore environment
            if original_value is not None:
                os.environ['CLI_CODE_GOOGLE_API_KEY'] = original_value
            elif 'CLI_CODE_GOOGLE_API_KEY' in os.environ:
                del os.environ['CLI_CODE_GOOGLE_API_KEY']
    
    @pytest.mark.timeout(5)  # Add timeout
    def test_load_dotenv_with_quoted_values(self, config_instance):
        """Test _load_dotenv with quoted values in .env file."""
        env_content = """
        CLI_CODE_GOOGLE_API_KEY="quoted-key-value"
        CLI_CODE_OLLAMA_API_URL='quoted-url'
        """
        
        # Store original env vars
        original_env = {}
        keys_to_check = ['CLI_CODE_GOOGLE_API_KEY', 'CLI_CODE_OLLAMA_API_URL']
        for key in keys_to_check:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            # Clear test env vars
            for key in keys_to_check:
                if key in os.environ:
                    del os.environ[key]
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open', mock_open(read_data=env_content)):
                
                config_instance._load_dotenv()
                
                # Verify environment variables were loaded with quotes removed
                assert os.environ.get('CLI_CODE_GOOGLE_API_KEY') == 'quoted-key-value'
                assert os.environ.get('CLI_CODE_OLLAMA_API_URL') == 'quoted-url'
        finally:
            # Restore environment
            for key in keys_to_check:
                if key in original_env:
                    os.environ[key] = original_env[key]
                elif key in os.environ:
                    del os.environ[key]
    
    @pytest.mark.timeout(5)  # Add timeout
    def test_load_dotenv_empty_or_invalid_lines(self, config_instance):
        """Test _load_dotenv with empty or invalid lines in .env file."""
        env_content = """
        # Comment line
        
        INVALID_LINE_NO_PREFIX
        CLI_CODE_VALID_KEY=valid-value
        =missing_key
        CLI_CODE_MISSING_VALUE=
        """
        
        # Store original env vars
        original_env = {}
        keys_to_check = ['CLI_CODE_VALID_KEY', 'CLI_CODE_MISSING_VALUE', 'INVALID_LINE_NO_PREFIX']
        for key in keys_to_check:
            if key in os.environ:
                original_env[key] = os.environ[key]
        
        try:
            # Clear test env vars
            for key in keys_to_check:
                if key in os.environ:
                    del os.environ[key]
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open', mock_open(read_data=env_content)), \
                 patch('cli_code.config.log') as mock_logger:
                
                config_instance._load_dotenv()
                
                # Verify only valid environment variables were loaded
                assert os.environ.get('CLI_CODE_VALID_KEY') == 'valid-value'
                assert os.environ.get('CLI_CODE_MISSING_VALUE') == ''
                assert 'INVALID_LINE_NO_PREFIX' not in os.environ
        finally:
            # Restore environment
            for key in keys_to_check:
                if key in original_env:
                    os.environ[key] = original_env[key]
                elif key in os.environ:
                    del os.environ[key]


# Additional tests for remaining uncovered sections

class TestEnsureConfigExistsEdgeCases:
    """Tests for edge cases in _ensure_config_exists method."""
    
    def test_ensure_config_exists_error_creating_directory(self, config_instance):
        """Test _ensure_config_exists when there's an error creating the directory."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir', side_effect=Exception("Could not create directory")), \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._ensure_config_exists()
            
            # Verify error is logged
            mock_logger.error.assert_called_once()
            assert "Error creating config directory" in mock_logger.error.call_args[0][0]
    
    def test_ensure_config_exists_error_saving_config(self, config_instance):
        """Test _ensure_config_exists when there's an error saving the config file."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', side_effect=Exception("File open error")), \
             patch('cli_code.config.log') as mock_logger:
            
            config_instance._ensure_config_exists()
            
            # Verify error is logged
            mock_logger.error.assert_called_once()
            assert "Error creating default config file" in mock_logger.error.call_args[0][0]


class TestCredentialEdgeCases:
    """Tests for edge cases in credential getter and setter methods."""
    
    def test_get_credential_none_config(self, config_instance):
        """Test get_credential when config is None."""
        config_instance.config = None
        
        # Verify None is returned
        assert config_instance.get_credential("any_provider") is None
    
    def test_get_credential_for_gemini_when_key_missing(self, config_instance):
        """Test get_credential for 'gemini' provider when key is missing."""
        config_instance.config = {}
        
        # Verify None is returned
        assert config_instance.get_credential("gemini") is None
    
    def test_get_credential_for_ollama_when_key_missing(self, config_instance):
        """Test get_credential for 'ollama' provider when key is missing."""
        config_instance.config = {}
        
        # Verify None is returned
        assert config_instance.get_credential("ollama") is None


class TestProviderModelEdgeCases:
    """Tests for edge cases in provider and model getter and setter methods."""
    
    def test_get_default_provider_none_config(self, config_instance):
        """Test get_default_provider when config is None."""
        config_instance.config = None
        
        # Verify fallback value is returned
        assert config_instance.get_default_provider() == "gemini"
    
    def test_get_default_model_none_config(self, config_instance):
        """Test get_default_model when config is None."""
        config_instance.config = None
        
        # Verify correct fallback values are returned
        assert config_instance.get_default_model("gemini") == "gemini-1.5-pro"
        assert config_instance.get_default_model("ollama") == "llama2"
    
    def test_get_default_model_unknown_provider(self, config_instance):
        """Test get_default_model with unknown provider."""
        config_instance.config = {}
        
        # Verify None is returned for unknown provider
        assert config_instance.get_default_model("unknown_provider") is None


class TestSettingEdgeCases:
    """Tests for edge cases in setting getter and setter methods."""
    
    def test_get_setting_none_config(self, config_instance):
        """Test get_setting when config is None."""
        config_instance.config = None
        
        # Verify default value is returned
        assert config_instance.get_setting("any_setting", default="fallback") == "fallback"
    
    def test_set_setting_none_config(self, config_instance):
        """Test set_setting when config is None."""
        config_instance.config = None
        
        with patch.object(Config, '_save_config') as mock_save:
            # This should initialize the config dict
            config_instance.set_setting("new_setting", "value")
            
            # Verify config was initialized and setting was added
            assert config_instance.config is not None
            assert config_instance.config.get('settings', {}).get('new_setting') == "value"
            
            # Verify config was saved
            mock_save.assert_called_once()


class TestGenericGetterSetter:
    """Tests for the generic getter and setter methods."""
    
    def test_get_config_value(self, config_instance):
        """Test get_config_value method."""
        # Set up test data
        config_instance.config = {
            'test_key': 'test_value',
            'nested': {'inner': 'value'}
        }
        
        # Test getting existing values
        assert config_instance.get_config_value('test_key') == 'test_value'
        assert config_instance.get_config_value('nested') == {'inner': 'value'}
        
        # Test getting non-existent value with default
        assert config_instance.get_config_value('non_existent', default='default') == 'default'
        
        # Test when config is None
        config_instance.config = None
        assert config_instance.get_config_value('any_key', default='fallback') == 'fallback'
    
    def test_set_config_value(self, config_instance):
        """Test set_config_value method."""
        config_instance.config = {'existing': 'old_value'}
        
        with patch.object(Config, '_save_config') as mock_save:
            # Test updating existing value
            config_instance.set_config_value('existing', 'new_value')
            assert config_instance.config['existing'] == 'new_value'
            
            # Test adding new value
            config_instance.set_config_value('new_key', 'value')
            assert config_instance.config['new_key'] == 'value'
            
            # Test when config is None
            config_instance.config = None
            config_instance.set_config_value('brand_new', 'value')
            assert config_instance.config == {'brand_new': 'value'}
            
            # Verify config was saved for each change
            assert mock_save.call_count == 3


class TestAdvancedConfigFunctions:
    """Tests for more complex or specific configuration functions that were previously uncovered."""
    
    def test_set_default_provider_with_invalid_provider(self, config_instance):
        """Test set_default_provider with an invalid provider name."""
        # Set up initial config
        config_instance.config = {'default_provider': 'gemini'}
        
        with patch.object(Config, '_save_config') as mock_save, \
             patch('cli_code.config.log') as mock_logger:
            
            # Call with invalid provider
            config_instance.set_default_provider('invalid_provider')
            
            # Verify it was still set (no validation in the method)
            assert config_instance.config['default_provider'] == 'invalid_provider'
            
            # Verify config was saved
            mock_save.assert_called_once()
    
    def test_set_default_model_gemini_with_none_config(self, config_instance):
        """Test set_default_model for gemini when config is None."""
        config_instance.config = None
        
        with patch.object(Config, '_save_config') as mock_save:
            # Call with gemini provider
            config_instance.set_default_model('gemini', 'gemini-test-model')
            
            # Verify config was initialized and model was set
            assert config_instance.config is not None
            assert config_instance.config.get('default_model') == 'gemini-test-model'
            
            # Verify config was saved
            mock_save.assert_called_once()
    
    def test_set_default_model_ollama_with_none_config(self, config_instance):
        """Test set_default_model for ollama when config is None."""
        config_instance.config = None
        
        with patch.object(Config, '_save_config') as mock_save:
            # Call with ollama provider
            config_instance.set_default_model('ollama', 'llama3')
            
            # Verify config was initialized and model was set
            assert config_instance.config is not None
            assert config_instance.config.get('ollama_default_model') == 'llama3'
            
            # Verify config was saved
            mock_save.assert_called_once()
    
    def test_set_default_model_unknown_provider(self, config_instance):
        """Test set_default_model with unknown provider."""
        config_instance.config = {}
        
        with patch.object(Config, '_save_config') as mock_save, \
             patch('cli_code.config.log') as mock_logger:
            
            # Call with unknown provider
            config_instance.set_default_model('unknown', 'some-model')
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "Unsupported provider" in mock_logger.warning.call_args[0][0]
            
            # Verify config was not changed
            assert not config_instance.config
            
            # Verify config was not saved
            mock_save.assert_not_called()
    
    @pytest.mark.timeout(5)  # Add timeout to prevent hanging
    def test_get_credential_for_gemini_from_env(self, config_instance):
        """Test get_credential for gemini with environment variable override."""
        # Set up config
        config_instance.config = {'google_api_key': 'config-key'}
        
        # Store original env var
        original_value = os.environ.get('CLI_CODE_GOOGLE_API_KEY')
        
        try:
            # Set test env var
            os.environ['CLI_CODE_GOOGLE_API_KEY'] = 'env-key'
            
            # Test with environment variable
            assert config_instance.get_credential('gemini') == 'env-key'
        finally:
            # Restore environment
            if original_value is not None:
                os.environ['CLI_CODE_GOOGLE_API_KEY'] = original_value
            else:
                os.environ.pop('CLI_CODE_GOOGLE_API_KEY', None)
    
    @pytest.mark.timeout(5)  # Add timeout to prevent hanging
    def test_get_credential_for_ollama_from_env(self, config_instance):
        """Test get_credential for ollama with environment variable override."""
        # Set up config
        config_instance.config = {'ollama_api_url': 'config-url'}
        
        # Store original env var
        original_value = os.environ.get('CLI_CODE_OLLAMA_API_URL')
        
        try:
            # Set test env var
            os.environ['CLI_CODE_OLLAMA_API_URL'] = 'env-url'
            
            # Test with environment variable
            assert config_instance.get_credential('ollama') == 'env-url'
        finally:
            # Restore environment
            if original_value is not None:
                os.environ['CLI_CODE_OLLAMA_API_URL'] = original_value
            else:
                os.environ.pop('CLI_CODE_OLLAMA_API_URL', None)
    
    def test_get_credential_for_unknown_provider(self, config_instance):
        """Test get_credential with unknown provider."""
        config_instance.config = {}
        
        # Verify None is returned for unknown provider
        assert config_instance.get_credential('unknown') is None 