"""
Tests for utility functions in src/cli_code/utils.py.
"""

from unittest.mock import MagicMock, patch

import pytest
import yaml

# Force module import for coverage
import src.cli_code.utils
from src.cli_code.cli_utils import count_tokens  # Updated import path

# Local Imports
from src.cli_code.config import Config  # Import the Config class

# Constants for testing
APP_NAME = "cli-code-test"  # Define test constant if needed by tests
CONFIG_FILE_NAME = "test-config.yaml"  # Use a test-specific config file name


@pytest.fixture
def temp_config_file(tmp_path):
    # Use tmp_path fixture provided by pytest for temporary directory
    test_dir = tmp_path / APP_NAME
    test_dir.mkdir()
    test_config_path = test_dir / CONFIG_FILE_NAME

    # Create a dummy config file for testing loading
    dummy_config_data = {
        # Add some initial structure if Config expects it
        "settings": {"initial_setting": 123}
    }
    with open(test_config_path, "w") as f:
        yaml.dump(dummy_config_data, f)
    return test_config_path


def test_load_config_exists(temp_config_file):
    """Test loading an existing config file using Config class."""
    config_instance = Config(config_file_path=temp_config_file)
    assert config_instance.config is not None
    assert config_instance.config.get("settings", {}).get("initial_setting") == 123


def test_load_config_nonexistent(tmp_path):
    """Test loading a non-existent config file."""
    non_existent_path = tmp_path / "nonexistent" / CONFIG_FILE_NAME
    # Initialize Config - it should create the default structure
    config_instance = Config(config_file_path=non_existent_path)
    # Check if the file was created and has default content
    assert non_existent_path.exists()
    # Assert that it loaded/created a config (might be default)
    assert isinstance(config_instance.config, dict)
    assert "settings" in config_instance.config  # Check for default keys


def test_update_config(temp_config_file):
    """Test updating the config file."""
    config_instance = Config(config_file_path=temp_config_file)
    # Update settings directly or via a method if Config class provides one
    config_instance.config["new_setting"] = "new_value"
    config_instance.config["settings"]["existing_setting"] = "updated"
    config_instance._save_config()  # Assuming a private save method

    # Create a new instance to reload and verify
    reloaded_instance = Config(config_file_path=temp_config_file)
    assert reloaded_instance.config.get("new_setting") == "new_value"
    assert reloaded_instance.config.get("settings", {}).get("existing_setting") == "updated"


def test_count_tokens_simple():
    """Test count_tokens with simple strings using tiktoken."""
    # These counts are based on gpt-4 tokenizer via tiktoken
    assert count_tokens("Hello world") == 2
    assert count_tokens("This is a test.") == 5
    assert count_tokens("") == 0
    assert count_tokens("   ") == 1  # Spaces are often single tokens


def test_count_tokens_special_chars():
    """Test count_tokens with special characters using tiktoken."""
    assert count_tokens("Hello, world! How are you?") == 8
    # Emojis can be multiple tokens
    # Note: Actual token count for emojis can vary
    assert count_tokens("Testing emojis 👍🚀") > 3


@patch("tiktoken.encoding_for_model")
def test_count_tokens_tiktoken_fallback(mock_encoding_for_model):
    """Test count_tokens fallback mechanism when tiktoken fails."""
    # Simulate tiktoken raising an exception
    mock_encoding_for_model.side_effect = Exception("Tiktoken error")

    # Test fallback (length // 4)
    assert count_tokens("This is exactly sixteen chars") == 7  # 28 // 4
    assert count_tokens("Short") == 1  # 5 // 4
    assert count_tokens("") == 0  # 0 // 4
    assert count_tokens("123") == 0  # 3 // 4
    assert count_tokens("1234") == 1  # 4 // 4


@patch("tiktoken.encoding_for_model")
def test_count_tokens_tiktoken_mocked_success(mock_encoding_for_model):
    """Test count_tokens main path with tiktoken mocked."""
    # Create a mock encoding object with a mock encode method
    mock_encode = MagicMock()
    mock_encode.encode.return_value = [1, 2, 3, 4, 5]  # Simulate encoding returning 5 tokens

    # Configure the mock context manager returned by encoding_for_model
    mock_encoding_for_model.return_value = mock_encode

    assert count_tokens("Some text that doesn't matter now") == 5
    mock_encoding_for_model.assert_called_once_with("gpt-4")
    mock_encode.encode.assert_called_once_with("Some text that doesn't matter now")
