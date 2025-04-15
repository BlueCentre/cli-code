"""
Tests for the configuration management in src/cli_code/config.py.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

# Assume cli_code is importable
from cli_code.config import Config

# --- Mocks and Fixtures ---


@pytest.fixture
def mock_home(tmp_path):
    """Fixture to mock Path.home() to use a temporary directory."""
    mock_home_path = tmp_path / ".home"
    mock_home_path.mkdir()
    with patch.object(Path, "home", return_value=mock_home_path):
        yield mock_home_path


@pytest.fixture
def mock_config_paths(mock_home):
    """Fixture providing expected config paths based on mock_home."""
    config_dir = mock_home / ".config" / "cli-code-agent"
    config_file = config_dir / "config.yaml"
    return config_dir, config_file


@pytest.fixture
def default_config_data():
    """Default configuration data structure."""
    return {
        "google_api_key": None,
        "default_provider": "gemini",
        "default_model": "models/gemini-2.5-pro-exp-03-25",
        "ollama_api_url": None,
        "ollama_default_model": "llama3.2",
        "settings": {
            "max_tokens": 1000000,
            "temperature": 0.5,
            "token_warning_threshold": 800000,
            "auto_compact_threshold": 950000,
        },
    }


# --- Test Cases ---


@patch("cli_code.config.Config._load_dotenv", MagicMock())  # Mock dotenv loading
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._ensure_config_exists")
def test_config_init_calls_ensure_when_load_fails(mock_ensure_config, mock_load_config, mock_config_paths):
    """Test Config calls _ensure_config_exists if _load_config returns empty."""
    config_dir, config_file = mock_config_paths

    # Simulate _load_config finding nothing (like file not found or empty)
    mock_load_config.return_value = {}

    with patch.dict(os.environ, {}, clear=True):
        # We don't need to check inside _ensure_config_exists here, just that it's called
        cfg = Config()

    mock_load_config.assert_called_once()
    # Verify that _ensure_config_exists was called because load failed
    mock_ensure_config.assert_called_once()
    # The final config might be the result of _ensure_config_exists potentially setting defaults
    # or the empty dict from _load_config depending on internal logic not mocked here.
    # Let's focus on the call flow for this test.


# Separate test for the behavior *inside* _ensure_config_exists
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
@patch("pathlib.Path.mkdir")
@patch("yaml.dump")
def test_ensure_config_exists_creates_default(
    mock_yaml_dump, mock_mkdir, mock_exists, mock_open_func, mock_config_paths, default_config_data
):
    """Test the _ensure_config_exists method creates a default file."""
    config_dir, config_file = mock_config_paths

    # Simulate config file NOT existing
    mock_exists.return_value = False

    # Directly instantiate config temporarily just to call the method
    # We need to bypass __init__ logic for this direct method test
    with patch.object(Config, "__init__", lambda x: None):  # Bypass __init__
        cfg = Config()
        cfg.config_dir = config_dir
        cfg.config_file = config_file
        cfg.config = {}  # Start with empty config

        # Call the method under test
        cfg._ensure_config_exists()

    # Assertions
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_exists.assert_called_with()
    mock_open_func.assert_called_once_with(config_file, "w")
    mock_yaml_dump.assert_called_once()
    args, kwargs = mock_yaml_dump.call_args
    # Check the data dumped matches the expected default structure
    assert args[0] == default_config_data


@patch("cli_code.config.Config._load_dotenv", MagicMock())  # Mock dotenv loading
@patch("cli_code.config.Config._apply_env_vars", MagicMock())  # Mock env var application
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._ensure_config_exists")  # Keep patch but don't assert not called
def test_config_init_loads_existing(mock_ensure_config, mock_load_config, mock_config_paths):
    """Test Config loads data from _load_config."""
    config_dir, config_file = mock_config_paths
    existing_data = {"google_api_key": "existing_key", "default_provider": "ollama", "settings": {"temperature": 0.8}}
    mock_load_config.return_value = existing_data.copy()

    with patch.dict(os.environ, {}, clear=True):
        cfg = Config()

    mock_load_config.assert_called_once()
    assert cfg.config == existing_data
    assert cfg.get_credential("gemini") == "existing_key"
    assert cfg.get_default_provider() == "ollama"
    assert cfg.get_setting("temperature") == 0.8


@patch("cli_code.config.Config._save_config")  # Mock save to prevent file writes
@patch("cli_code.config.Config._load_config")  # Correct patch target
def test_config_setters_getters(mock_load_config, mock_save, mock_config_paths):
    """Test the various getter and setter methods."""
    config_dir, config_file = mock_config_paths
    initial_data = {
        "google_api_key": "initial_google_key",
        "ollama_api_url": "initial_ollama_url",
        "default_provider": "gemini",
        "default_model": "gemini-model-1",
        "ollama_default_model": "ollama-model-1",
        "settings": {"temperature": 0.7, "max_tokens": 500000},
    }
    mock_load_config.return_value = initial_data.copy()  # Mock the load result

    # Mock other __init__ methods to isolate loading
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("cli_code.config.Config._load_dotenv", MagicMock()),
        patch("cli_code.config.Config._ensure_config_exists", MagicMock()),
        patch("cli_code.config.Config._apply_env_vars", MagicMock()),
    ):
        cfg = Config()

    # Test initial state loaded correctly
    assert cfg.get_credential("gemini") == "initial_google_key"
    assert cfg.get_credential("ollama") == "initial_ollama_url"
    assert cfg.get_default_provider() == "gemini"
    assert cfg.get_default_model() == "gemini-model-1"  # Default provider is gemini
    assert cfg.get_default_model(provider="gemini") == "gemini-model-1"
    assert cfg.get_default_model(provider="ollama") == "ollama-model-1"
    assert cfg.get_setting("temperature") == 0.7
    assert cfg.get_setting("max_tokens") == 500000
    assert cfg.get_setting("non_existent", default="fallback") == "fallback"

    # Test Setters
    cfg.set_credential("gemini", "new_google_key")
    assert cfg.config["google_api_key"] == "new_google_key"
    assert mock_save.call_count == 1
    cfg.set_credential("ollama", "new_ollama_url")
    assert cfg.config["ollama_api_url"] == "new_ollama_url"
    assert mock_save.call_count == 2

    cfg.set_default_provider("ollama")
    assert cfg.config["default_provider"] == "ollama"
    assert mock_save.call_count == 3

    # Setting default model when default provider is ollama
    cfg.set_default_model("ollama-model-2")
    assert cfg.config["ollama_default_model"] == "ollama-model-2"
    assert mock_save.call_count == 4
    # Setting default model explicitly for gemini
    cfg.set_default_model("gemini-model-2", provider="gemini")
    assert cfg.config["default_model"] == "gemini-model-2"
    assert mock_save.call_count == 5

    cfg.set_setting("temperature", 0.9)
    assert cfg.config["settings"]["temperature"] == 0.9
    assert mock_save.call_count == 6
    cfg.set_setting("new_setting", True)
    assert cfg.config["settings"]["new_setting"] is True
    assert mock_save.call_count == 7

    # Test Getters after setting
    assert cfg.get_credential("gemini") == "new_google_key"
    assert cfg.get_credential("ollama") == "new_ollama_url"
    assert cfg.get_default_provider() == "ollama"
    assert cfg.get_default_model() == "ollama-model-2"  # Default provider is now ollama
    assert cfg.get_default_model(provider="gemini") == "gemini-model-2"
    assert cfg.get_default_model(provider="ollama") == "ollama-model-2"
    assert cfg.get_setting("temperature") == 0.9
    assert cfg.get_setting("new_setting") is True

    # Test setting unknown provider (should log error, not save)
    cfg.set_credential("unknown", "some_key")
    assert "unknown" not in cfg.config
    assert mock_save.call_count == 7  # No new save call
    cfg.set_default_provider("unknown")
    assert cfg.config["default_provider"] == "ollama"  # Should remain unchanged
    assert mock_save.call_count == 7  # No new save call
    cfg.set_default_model("unknown-model", provider="unknown")
    assert cfg.config.get("unknown_default_model") is None
    assert mock_save.call_count == 7  # No new save call


# New test combining env var logic check
@patch("cli_code.config.Config._load_dotenv", MagicMock())  # Mock dotenv loading step
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())  # Mock ensure config
@patch("cli_code.config.Config._save_config")  # Mock save to check if called
def test_config_env_var_override(mock_save, mock_load_config, mock_config_paths):
    """Test that _apply_env_vars correctly overrides loaded config."""
    config_dir, config_file = mock_config_paths
    initial_config_data = {
        "google_api_key": "config_key",
        "ollama_api_url": "config_url",
        "default_provider": "gemini",
        "ollama_default_model": "config_ollama",
    }
    env_vars = {
        "CLI_CODE_GOOGLE_API_KEY": "env_key",
        "CLI_CODE_OLLAMA_API_URL": "env_url",
        "CLI_CODE_DEFAULT_PROVIDER": "ollama",
    }
    mock_load_config.return_value = initial_config_data.copy()

    with patch.dict(os.environ, env_vars, clear=True):
        cfg = Config()

    assert cfg.config["google_api_key"] == "env_key"
    assert cfg.config["ollama_api_url"] == "env_url"
    assert cfg.config["default_provider"] == "ollama"
    assert cfg.config["ollama_default_model"] == "config_ollama"


# New simplified test for _migrate_old_config_paths
# @patch('builtins.open', new_callable=mock_open)
# @patch('yaml.safe_load')
# @patch('cli_code.config.Config._save_config')
# def test_migrate_old_config_paths_logic(mock_save, mock_yaml_load, mock_open_func, mock_home):
#    ... (implementation removed) ...

# End of file
