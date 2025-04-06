"""
Tests for Config class methods that might have been missed in existing tests.
"""

import os
import tempfile
import pytest
from pathlib import Path
import yaml
from unittest.mock import patch, mock_open

from cli_code.config import Config


@pytest.fixture
def temp_config_dir():
    """Creates a temporary directory for the config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Return a Config instance with mocked file operations."""
    with patch('cli_code.config.Config._load_dotenv'), \
         patch('cli_code.config.Config._ensure_config_exists'), \
         patch('cli_code.config.Config._load_config', return_value={}), \
         patch('cli_code.config.Config._apply_env_vars'):
        config = Config()
        # Set some test data
        config.config = {
            "google_api_key": "test-google-key",
            "default_provider": "gemini",
            "default_model": "models/gemini-1.0-pro",
            "ollama_api_url": "http://localhost:11434",
            "ollama_default_model": "llama2",
            "settings": {
                "max_tokens": 1000,
                "temperature": 0.7,
            }
        }
        yield config


def test_get_credential(mock_config):
    """Test get_credential method."""
    # Test existing provider
    assert mock_config.get_credential("google") == "test-google-key"
    
    # Test non-existing provider
    assert mock_config.get_credential("non_existing") is None
    
    # Test with empty config
    mock_config.config = {}
    assert mock_config.get_credential("google") is None


def test_set_credential(mock_config):
    """Test set_credential method."""
    # Test setting existing provider
    mock_config.set_credential("google", "new-google-key")
    assert mock_config.config["google_api_key"] == "new-google-key"
    
    # Test setting new provider
    mock_config.set_credential("openai", "test-openai-key")
    assert mock_config.config["openai_api_key"] == "test-openai-key"
    
    # Test with None value
    mock_config.set_credential("google", None)
    assert mock_config.config["google_api_key"] is None


def test_get_default_provider(mock_config):
    """Test get_default_provider method."""
    # Test with existing provider
    assert mock_config.get_default_provider() == "gemini"
    
    # Test with no provider set
    mock_config.config["default_provider"] = None
    assert mock_config.get_default_provider() == "gemini"  # Should return default
    
    # Test with empty config
    mock_config.config = {}
    assert mock_config.get_default_provider() == "gemini"  # Should return default


def test_set_default_provider(mock_config):
    """Test set_default_provider method."""
    # Test setting valid provider
    mock_config.set_default_provider("openai")
    assert mock_config.config["default_provider"] == "openai"
    
    # Test setting None (should use default)
    mock_config.set_default_provider(None)
    assert mock_config.config["default_provider"] == "gemini"


def test_get_default_model(mock_config):
    """Test get_default_model method."""
    # Test without provider (use default provider)
    assert mock_config.get_default_model() == "models/gemini-1.0-pro"
    
    # Test with specific provider
    assert mock_config.get_default_model("ollama") == "llama2"
    
    # Test with non-existing provider
    assert mock_config.get_default_model("non_existing") is None


def test_set_default_model(mock_config):
    """Test set_default_model method."""
    # Test with default provider
    mock_config.set_default_model("new-model")
    assert mock_config.config["default_model"] == "new-model"
    
    # Test with specific provider
    mock_config.set_default_model("new-ollama-model", "ollama")
    assert mock_config.config["ollama_default_model"] == "new-ollama-model"
    
    # Test with new provider
    mock_config.set_default_model("anthropic-model", "anthropic")
    assert mock_config.config["anthropic_default_model"] == "anthropic-model"


def test_get_setting(mock_config):
    """Test get_setting method."""
    # Test existing setting
    assert mock_config.get_setting("max_tokens") == 1000
    assert mock_config.get_setting("temperature") == 0.7
    
    # Test non-existing setting with default
    assert mock_config.get_setting("non_existing", "default_value") == "default_value"
    
    # Test with empty settings
    mock_config.config["settings"] = {}
    assert mock_config.get_setting("max_tokens", 2000) == 2000


def test_set_setting(mock_config):
    """Test set_setting method."""
    # Test updating existing setting
    mock_config.set_setting("max_tokens", 2000)
    assert mock_config.config["settings"]["max_tokens"] == 2000
    
    # Test adding new setting
    mock_config.set_setting("new_setting", "new_value")
    assert mock_config.config["settings"]["new_setting"] == "new_value"
    
    # Test with no settings dict
    mock_config.config.pop("settings")
    mock_config.set_setting("test_setting", "test_value")
    assert mock_config.config["settings"]["test_setting"] == "test_value"


def test_save_config():
    """Test _save_config method."""
    with patch('builtins.open', mock_open()) as mock_file, \
         patch('yaml.dump') as mock_yaml_dump, \
         patch('cli_code.config.Config._load_dotenv'), \
         patch('cli_code.config.Config._ensure_config_exists'), \
         patch('cli_code.config.Config._load_config', return_value={}), \
         patch('cli_code.config.Config._apply_env_vars'):
        
        config = Config()
        config.config = {"test": "data"}
        config._save_config()
        
        mock_file.assert_called_once()
        mock_yaml_dump.assert_called_once_with({"test": "data"}, mock_file(), default_flow_style=False)


def test_save_config_error():
    """Test error handling in _save_config method."""
    with patch('builtins.open', side_effect=PermissionError("Permission denied")), \
         patch('cli_code.config.log.error') as mock_log_error, \
         patch('cli_code.config.Config._load_dotenv'), \
         patch('cli_code.config.Config._ensure_config_exists'), \
         patch('cli_code.config.Config._load_config', return_value={}), \
         patch('cli_code.config.Config._apply_env_vars'):
        
        config = Config()
        config._save_config()
        
        # Verify error was logged
        assert mock_log_error.called 