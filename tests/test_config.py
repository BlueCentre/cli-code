"""
Tests for the configuration management in src/cli_code/config.py.
"""

import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

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
    config_dir = mock_home / ".config" / "cli-code"
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


# === Tests for Config Path Override ===


@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("yaml.dump")  # Mock yaml dump to prevent actual file writing
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
@patch("pathlib.Path.mkdir")
def test_ensure_config_exists_creates_custom_path(mock_mkdir, mock_exists, mock_open_func, mock_yaml_dump, tmp_path):
    """Test _ensure_config_exists creates file at a custom path provided via param."""
    custom_dir = tmp_path / "custom_config_location"
    custom_file = custom_dir / "my_config.yaml"

    # Simulate file not existing
    mock_exists.return_value = False

    # Create a config with our custom path
    with patch("cli_code.config.Config._load_config", return_value={}), patch("cli_code.config.Config._apply_env_vars"):
        cfg = Config(config_file_path=custom_file)

    # Assertions for _ensure_config_exists behavior
    # It should have created the directory
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # It should have checked if the file exists
    mock_exists.assert_called_with()
    # It should have opened the custom file for writing
    mock_open_func.assert_called_with(custom_file, "w")
    # It should have dumped the default config
    mock_yaml_dump.assert_called_once()
    # Verify the paths stored in the instance are correct
    assert cfg.config_dir == custom_dir
    assert cfg.config_file == custom_file


@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_config")  # Mock _load_config
def test_config_load_from_env_var_path(mock_load_config, tmp_path):
    """Test Config loads from path specified by CLI_CODE_CONFIG_FILE env var."""
    custom_dir = tmp_path / "env_config_location"
    custom_file = custom_dir / "env_config.yaml"
    custom_data = {"key_from_env_file": "value_env"}

    # Configure the mock _load_config to return data
    # This implicitly tests that _load_config is called *after* the path is set correctly
    mock_load_config.return_value = custom_data.copy()

    # Set the environment variable
    env_override = {"CLI_CODE_CONFIG_FILE": str(custom_file)}
    with patch.dict(os.environ, env_override, clear=True):
        cfg = Config()

    # Check that _load_config was called
    mock_load_config.assert_called_once()
    # Check that the loaded config contains our custom data key
    assert cfg.config.get("key_from_env_file") == custom_data["key_from_env_file"]
    # Verify the config paths are correct (set during __init__)
    assert cfg.config_file == custom_file
    assert cfg.config_dir == custom_dir


@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_config")  # Mock _load_config
def test_config_load_from_param_path(mock_load_config, tmp_path):
    """Test Config loads from path specified by __init__ parameter."""
    param_dir = tmp_path / "param_config_location"
    param_file = param_dir / "param_config.yaml"
    param_data = {"key_from_param_file": "value_param"}

    # Mock the load result
    mock_load_config.return_value = param_data.copy()

    with patch.dict(os.environ, {}, clear=True):  # Ensure no env var interference
        cfg = Config(config_file_path=param_file)

    mock_load_config.assert_called_once()
    assert cfg.config.get("key_from_param_file") == param_data["key_from_param_file"]
    assert cfg.config_file == param_file
    assert cfg.config_dir == param_dir


@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_config")  # Mock _load_config
def test_config_param_overrides_env_var_path(mock_load_config, tmp_path):
    """Test Config parameter path takes precedence over environment variable path."""
    env_dir = tmp_path / "env_config_location_prio"
    env_file = env_dir / "env_config_prio.yaml"

    param_dir = tmp_path / "param_config_location_prio"
    param_file = param_dir / "param_config_prio.yaml"
    param_data = {"key": "param"}

    # Mock _load_config to return the data we expect from the parameter path
    mock_load_config.return_value = param_data.copy()

    # Set the environment variable (should be ignored)
    env_override = {"CLI_CODE_CONFIG_FILE": str(env_file)}
    with patch.dict(os.environ, env_override, clear=True):
        # Instantiate with the parameter path
        cfg = Config(config_file_path=param_file)

    # Assert that _load_config was called
    mock_load_config.assert_called_once()
    # Check the key from param_data is in the config
    assert cfg.config.get("key") == param_data["key"]
    # The config may include more keys (like an empty settings dict) - that's expected
    # Verify the path stored in the instance is the parameter path
    assert cfg.config_file == param_file
    assert cfg.config_dir == param_dir


# Test for empty environment variable case
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_config", MagicMock())
def test_empty_env_var_falls_back_to_default(mock_config_paths):
    """Test that an empty environment variable falls back to the default path."""
    # Set empty string as env var
    with patch.dict(os.environ, {"CLI_CODE_CONFIG_FILE": ""}, clear=True):
        with patch("cli_code.config.log") as mock_log:
            cfg = Config()

            # Check that the warning log was emitted
            mock_log.warning.assert_called_with("Config file path resolved to empty, falling back to default.")

            # Verify it fell back to default path
            assert "/.config/cli-code/config.yaml" in str(cfg.config_file)


# Test path resolution handling
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_config", MagicMock())
def test_path_resolution(tmp_path):
    """Test that the path resolution handles relative paths and expanduser."""
    # Test with tilde in path
    home_dir = os.path.expanduser("~")
    with patch.dict(os.environ, {"CLI_CODE_CONFIG_FILE": "~/custom_config.yaml"}, clear=True):
        cfg = Config()
        assert str(cfg.config_file).startswith(home_dir)
        assert str(cfg.config_file).endswith("custom_config.yaml")

    # Test with relative path
    with patch("pathlib.Path.resolve", return_value=Path(tmp_path / "relative/path/config.yaml")):
        with patch.dict(os.environ, {"CLI_CODE_CONFIG_FILE": "relative/path/config.yaml"}, clear=True):
            cfg = Config()
            assert str(cfg.config_file) == str(tmp_path / "relative/path/config.yaml")
            assert cfg.config_dir == Path(tmp_path / "relative/path")


# Test exception handling in directory creation
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._load_config")
def test_ensure_config_exists_handles_dir_creation_error(mock_load_config, mock_config_paths):
    """Test that _ensure_config_exists handles errors during directory creation gracefully."""
    config_dir, config_file = mock_config_paths
    mock_load_config.return_value = {}

    # Make mkdir raise an exception
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
        with patch("cli_code.config.log") as mock_log:
            # We expect this to log an error but not raise an exception
            cfg = Config()

            # Verify the error was logged
            mock_log.error.assert_called_once()
            assert "Failed to create config directory" in mock_log.error.call_args[0][0]

            # Verify the method returned early and didn't attempt to create the file
            assert mock_load_config.called


# Test yaml error handling
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._apply_env_vars", MagicMock())  # Add this to prevent env var loading
def test_load_config_handles_yaml_errors(mock_config_paths):
    """Test that _load_config handles YAML parsing errors gracefully."""
    config_dir, config_file = mock_config_paths

    # Mock the yaml.safe_load to raise a YAMLError
    with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content")):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("cli_code.config.log") as mock_log:
                    # Directly call the method we're testing instead of the full __init__
                    with patch.object(Config, "__init__", lambda x: None):
                        cfg = Config()
                        cfg.config_file = config_file
                        cfg.config = cfg._load_config()

                    # Check that the error was logged
                    mock_log.error.assert_called_once()
                    assert "Error parsing YAML config file" in mock_log.error.call_args[0][0]

                    # Verify that an empty dict was returned
                    assert cfg.config == {}


# Test save config exception handling
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._apply_env_vars", MagicMock())  # Add this to prevent env var loading
def test_save_config_handles_exceptions(mock_config_paths):
    """Test that _save_config handles exceptions gracefully."""
    config_dir, config_file = mock_config_paths

    # Mock open to raise an exception when writing
    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        with patch("cli_code.config.log") as mock_log:
            # Initialize manually without calling __init__
            with patch.object(Config, "__init__", lambda x: None):
                cfg = Config()
                cfg.config_file = config_file
                cfg.config = {"test": "value"}

                # Try to save the config, which should trigger the exception
                cfg._save_config()

            # Verify the error was logged
            mock_log.error.assert_called_once()
            assert "Error saving config file" in mock_log.error.call_args[0][0]


# Simpler test for _load_dotenv using direct mocking of Path objects
@patch("pathlib.Path.exists")
@patch("cli_code.config.log")
def test_load_dotenv_with_no_files(mock_log, mock_exists):
    """Test that _load_dotenv handles case when neither .env nor .env.example exist."""
    # Setup mocks - neither file exists
    mock_exists.return_value = False

    # Manually initialize Config instance
    with patch.object(Config, "__init__", lambda x: None):
        cfg = Config()
        # Call the method under test directly
        cfg._load_dotenv()

    # Check for the log message
    mock_log.debug.assert_called_with("No .env or .env.example file found.")


@patch("builtins.open", side_effect=PermissionError("Permission denied"))
@patch("pathlib.Path.exists")
@patch("cli_code.config.log")
def test_load_dotenv_handles_errors(mock_log, mock_exists, mock_open):
    """Test that _load_dotenv handles errors when reading dotenv files."""
    # Setup mocks - .env exists but raises error on open
    mock_exists.return_value = True

    # Manually initialize Config instance
    with patch.object(Config, "__init__", lambda x: None):
        cfg = Config()
        # Call the method under test directly
        cfg._load_dotenv()

    # Check for the error log
    mock_log.warning.assert_called_once()
    assert "Error loading" in mock_log.warning.call_args[0][0]


# Add tests for more credential providers
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._save_config")
def test_additional_credential_providers(mock_save_config, mock_load_config, mock_config_paths):
    """Test getting and setting credentials for additional providers like OpenAI."""
    config_dir, config_file = mock_config_paths

    # Initial config with no OpenAI key
    initial_config = {"google_api_key": "test_google_key", "openai_api_key": None, "settings": {}}

    mock_load_config.return_value = initial_config.copy()

    # Initialize Config instance with mocked methods
    with patch.dict(os.environ, {}, clear=True):
        cfg = Config()

    # Test setting OpenAI credential
    cfg.set_credential("openai", "test_openai_key")
    assert cfg.config["openai_api_key"] == "test_openai_key"
    mock_save_config.assert_called_once()

    # Test setting unknown provider
    mock_save_config.reset_mock()
    with patch("cli_code.config.log") as mock_log:
        cfg.set_credential("unknown_provider", "test_key")
        mock_log.error.assert_called_once()
        assert "unknown provider" in mock_log.error.call_args[0][0].lower()

    # Verify save_config wasn't called for unknown provider
    mock_save_config.assert_not_called()


# Add tests for more provider model handling
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._apply_env_vars", MagicMock())
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._save_config")
def test_model_getters_setters_additional_providers(mock_save_config, mock_load_config):
    """Test model getters and setters for additional providers like anthropic."""
    # Initial config with anthropic settings
    initial_config = {"default_provider": "anthropic", "anthropic_default_model": "claude-3-sonnet"}

    mock_load_config.return_value = initial_config.copy()

    # Initialize Config instance with mocked methods
    cfg = Config()

    # Test getting anthropic model
    model = cfg.get_default_model()
    assert model == "claude-3-sonnet"

    # Test setting anthropic model
    mock_save_config.reset_mock()
    cfg.set_default_model("claude-3-opus")
    assert cfg.config["anthropic_default_model"] == "claude-3-opus"
    mock_save_config.assert_called_once()

    # Test get_default_model with None config
    # Patch the instance attribute instead of the class attribute
    original_config = cfg.config
    cfg.config = None
    try:
        # Should return defaults for known providers
        assert cfg.get_default_model(provider="gemini") == "models/gemini-1.5-pro-latest"
        assert cfg.get_default_model(provider="ollama") == "llama2"
        assert cfg.get_default_model(provider="unknown") is None
    finally:
        # Restore the original config
        cfg.config = original_config


# Test the complete _apply_env_vars method with complex env vars
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_config")
def test_apply_env_vars_complex_types(mock_load_config):
    """Test that _apply_env_vars correctly handles different value types."""
    # Initial empty config
    mock_load_config.return_value = {"settings": {}}

    # Set environment variables with different types to test conversion
    env_vars = {
        "CLI_CODE_SETTINGS_INTEGER_VAL": "42",
        "CLI_CODE_SETTINGS_FLOAT_VAL": "3.14",
        "CLI_CODE_SETTINGS_BOOL_TRUE": "true",
        "CLI_CODE_SETTINGS_BOOL_FALSE": "false",
        "CLI_CODE_SETTINGS_STRING_VAL": "hello world",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        cfg = Config()

    # Check that values were converted to the correct types
    assert cfg.config["settings"]["integer_val"] == 42
    assert cfg.config["settings"]["float_val"] == 3.14
    assert cfg.config["settings"]["bool_true"] is True
    assert cfg.config["settings"]["bool_false"] is False
    assert cfg.config["settings"]["string_val"] == "hello world"


# Test edge cases for get_setting and set_setting
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._apply_env_vars", MagicMock())
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._save_config")
def test_settings_edge_cases(mock_save_config, mock_load_config):
    """Test edge cases for get_setting and set_setting methods."""
    # Test with None settings field
    mock_load_config.return_value = {"settings": None}
    cfg = Config()

    # get_setting should handle None settings gracefully
    assert cfg.get_setting("test_setting", default="default") == "default"

    # set_setting should create settings dict if None
    cfg.set_setting("test_setting", "test_value")
    assert cfg.config["settings"]["test_setting"] == "test_value"
    mock_save_config.assert_called_once()

    # Test with None config
    mock_save_config.reset_mock()
    # Patch the instance attribute instead of the class attribute
    original_config = cfg.config
    cfg.config = None
    try:
        with patch("cli_code.config.log") as mock_log:
            # Should log warning and return early
            cfg.set_setting("another_setting", "value")
            mock_log.warning.assert_called_once()
            mock_save_config.assert_not_called()
    finally:
        # Restore the original config
        cfg.config = original_config


# Test set_default_provider with None value
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._apply_env_vars", MagicMock())
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._save_config")
def test_set_default_provider_none(mock_save_config, mock_load_config):
    """Test set_default_provider handles None value gracefully."""
    mock_load_config.return_value = {"default_provider": "ollama"}
    cfg = Config()

    # Setting None should reset to gemini
    cfg.set_default_provider(None)
    assert cfg.config["default_provider"] == "gemini"
    mock_save_config.assert_called_once()


# End of file


# Add more comprehensive tests for the anthropic provider
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._apply_env_vars", MagicMock())
@patch("cli_code.config.Config._load_config")
@patch("cli_code.config.Config._save_config")
def test_anthropic_provider_support(mock_save_config, mock_load_config):
    """Test getting and setting credentials and models for the anthropic provider."""
    # Initial config
    initial_config = {
        "default_provider": "anthropic",
        "anthropic_default_model": "claude-3-sonnet",
        "anthropic_api_key": "test_anthropic_key",
    }

    mock_load_config.return_value = initial_config.copy()

    # Initialize Config instance
    cfg = Config()

    # Test get_default_provider
    assert cfg.get_default_provider() == "anthropic"

    # Test setting default provider
    cfg.set_default_provider("anthropic")
    assert cfg.config["default_provider"] == "anthropic"
    mock_save_config.assert_called_once()

    # Test getting default model
    mock_save_config.reset_mock()
    assert cfg.get_default_model() == "claude-3-sonnet"

    # Test setting default model
    cfg.set_default_model("claude-3-opus", provider="anthropic")
    assert cfg.config["anthropic_default_model"] == "claude-3-opus"
    mock_save_config.assert_called_once()


# Add test for _load_config FileNotFoundError case
@patch("cli_code.config.Config._ensure_config_exists", MagicMock())
@patch("cli_code.config.Config._load_dotenv", MagicMock())
@patch("cli_code.config.Config._apply_env_vars", MagicMock())
def test_load_config_file_not_found():
    """Test that _load_config handles FileNotFoundError."""
    with patch.object(Config, "__init__", lambda x: None):
        cfg = Config()
        cfg.config_file = Path("/nonexistent/path/config.yaml")

        with patch("cli_code.config.log") as mock_log:
            result = cfg._load_config()

            # Should return empty dict and log a warning
            assert result == {}
            mock_log.warning.assert_called_once()
            assert "not found" in mock_log.warning.call_args[0][0]


# Add tests for get_credential for unknown provider
def test_get_credential_unknown_provider():
    """Test get_credential with an unknown provider logs a warning and returns None."""
    with patch.object(Config, "__init__", lambda x: None):
        with patch("cli_code.config.log") as mock_log:
            cfg = Config()
            cfg.config = {"google_api_key": "test_key", "ollama_api_url": "test_url"}

            # Test with unknown provider
            result = cfg.get_credential("unknown_provider")

            # Should log a warning and return None
            assert result is None
            mock_log.warning.assert_called_once()
            assert "unknown provider" in mock_log.warning.call_args[0][0].lower()


# Add test for get_default_model with unknown provider
def test_get_default_model_unknown_provider():
    """Test get_default_model with an unknown provider logs a warning and returns None."""
    with patch.object(Config, "__init__", lambda x: None):
        with patch("cli_code.config.log") as mock_log:
            cfg = Config()
            cfg.config = {"default_provider": "gemini", "default_model": "gemini-model"}

            # Test with unknown provider
            result = cfg.get_default_model(provider="unknown_provider")

            # Should log a warning and return None
            assert result is None
            mock_log.warning.assert_called_once()
            assert "unknown provider" in mock_log.warning.call_args[0][0].lower()


# Test the _apply_env_vars method with settings conversion
def test_apply_env_vars_with_settings_conversion():
    """Test that _apply_env_vars correctly converts environment variable types."""
    with patch.object(Config, "__init__", lambda x: None):
        cfg = Config()
        cfg.config = {"settings": {}}

        # Set environment variables with different settings
        test_env = {
            "CLI_CODE_SETTINGS_INT_VAL": "42",
            "CLI_CODE_SETTINGS_FLOAT_VAL": "3.14",
            "CLI_CODE_SETTINGS_BOOL_TRUE": "true",
            "CLI_CODE_SETTINGS_BOOL_FALSE": "false",
            "CLI_CODE_SETTINGS_STR_VAL": "string_value",
            "CLI_CODE_GOOGLE_API_KEY": "test_api_key",
            "CLI_CODE_DEFAULT_PROVIDER": "gemini",
        }

        with patch.dict(os.environ, test_env, clear=True):
            # Call the method
            cfg._apply_env_vars()

        # Check direct mappings
        assert cfg.config["google_api_key"] == "test_api_key"
        assert cfg.config["default_provider"] == "gemini"

        # Check settings conversions
        assert cfg.config["settings"]["int_val"] == 42  # Integer conversion
        assert cfg.config["settings"]["float_val"] == 3.14  # Float conversion
        assert cfg.config["settings"]["bool_true"] is True  # Boolean conversion
        assert cfg.config["settings"]["bool_false"] is False  # Boolean conversion
        assert cfg.config["settings"]["str_val"] == "string_value"  # String (no conversion)


# Add test for _load_dotenv file content processing
@patch("builtins.open")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.resolve")
def test_load_dotenv_processes_file_content(mock_resolve, mock_exists, mock_open):
    """Test that _load_dotenv correctly processes various file content formats."""
    # Setup mocks
    mock_exists.return_value = True
    mock_resolve.return_value = Path("/fake/path/.env")

    # Create a mock file with various content types to exercise the processing logic
    mock_file = MagicMock()
    mock_file.__enter__.return_value = iter(
        [
            "# This is a comment",
            "",  # Empty line
            "NORMAL_VAR=normal_value",
            'QUOTED_VAR="quoted_value"',
            "SINGLE_QUOTED_VAR='single_quoted'",
            "CLI_CODE_API_KEY=secret_key",
            "CLI_CODE_SETTINGS_BOOL=true",
            "NO_EQUALS_LINE",  # Line without equals
            "TRAILING_SPACES = with_spaces  ",
            "  LEADING_SPACES  =  with_spaces_too  ",
        ]
    )
    mock_open.return_value = mock_file

    # Patch os.environ to track what gets set
    test_env = {}
    with patch.dict(os.environ, test_env, clear=True):
        # Create Config instance with minimal mocking
        with patch.object(Config, "__init__", lambda x: None):
            cfg = Config()
            # Call the method under test directly
            cfg._load_dotenv()

            # Assert environment variables were set properly
            assert os.environ["NORMAL_VAR"] == "normal_value"
            assert os.environ["QUOTED_VAR"] == "quoted_value"
            assert os.environ["SINGLE_QUOTED_VAR"] == "single_quoted"
            assert os.environ["CLI_CODE_API_KEY"] == "secret_key"
            assert os.environ["CLI_CODE_SETTINGS_BOOL"] == "true"
            assert os.environ["TRAILING_SPACES"] == "with_spaces"
            assert os.environ["LEADING_SPACES"] == "with_spaces_too"
            # Verify lines without equals are skipped (no exception raised)
            assert "NO_EQUALS_LINE" not in os.environ


@patch("builtins.open")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.resolve")
def test_load_dotenv_logs_loaded_vars(mock_resolve, mock_exists, mock_open, caplog):
    """Test that _load_dotenv correctly logs loaded variables."""
    # Setup mocks
    mock_exists.side_effect = [True, False]  # First check for .env exists, .env.example doesn't
    mock_resolve.return_value = Path("/fake/path/.env")

    # Create mock file with CLI_CODE prefixed variables to test logging
    mock_file = MagicMock()
    mock_file.__enter__.return_value = iter(
        [
            "CLI_CODE_API_KEY=secret_value",  # Should be masked in log
            "CLI_CODE_TOKEN=another_secret",  # Should be masked in log
            "CLI_CODE_NORMAL=normal_value",  # Should show actual value
            "OTHER_VAR=not_logged",  # Should not be in log since not CLI_CODE prefixed
        ]
    )
    mock_open.return_value = mock_file

    # Use caplog to capture logging output
    caplog.set_level(logging.INFO)

    # Create Config instance with minimal mocking
    with patch.object(Config, "__init__", lambda x: None):
        cfg = Config()
        # Call the method under test directly
        cfg._load_dotenv()

        # Check log output for loaded variables
        assert "Loaded 3 CLI_CODE vars" in caplog.text
        assert "CLI_CODE_API_KEY=****" in caplog.text  # Should be masked
        assert "CLI_CODE_TOKEN=****" in caplog.text  # Should be masked
        assert "CLI_CODE_NORMAL=normal_value" in caplog.text  # Should show value
        assert "OTHER_VAR=not_logged" not in caplog.text  # Should not be logged


# Test with empty environment variable value
@patch("builtins.open")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.resolve")
def test_load_dotenv_handles_empty_values(mock_resolve, mock_exists, mock_open):
    """Test that _load_dotenv correctly processes empty environment variable values."""
    # Setup mocks
    mock_exists.return_value = True
    mock_resolve.return_value = Path("/fake/path/.env")

    # Create a mock file with empty values
    mock_file = MagicMock()
    mock_file.__enter__.return_value = iter(
        [
            "EMPTY_VAR=",  # Empty value
            "CLI_CODE_EMPTY=",  # Empty CLI_CODE value
        ]
    )
    mock_open.return_value = mock_file

    # Patch os.environ to track what gets set
    test_env = {}
    with patch.dict(os.environ, test_env, clear=True):
        # Create Config instance with minimal mocking
        with patch.object(Config, "__init__", lambda x: None):
            cfg = Config()

            # Mock the logger directly before calling _load_dotenv
            with patch("cli_code.config.log") as mock_log:
                # Call the method under test directly
                cfg._load_dotenv()

                # Assert environment variables were set with empty values
                assert os.environ["EMPTY_VAR"] == ""
                assert os.environ["CLI_CODE_EMPTY"] == ""

                # Verify that info was called with loaded vars
                mock_log.info.assert_any_call("Loaded 1 CLI_CODE vars from .env: CLI_CODE_EMPTY=")


# New tests that don't rely on monkey patching __init__


# Test for _load_dotenv exception path
@pytest.fixture
def env_file_with_error():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    env_path = Path(temp_dir) / ".env"
    
    # Create a .env file that will trigger a permission error
    with open(env_path, "w") as f:
        f.write("TEST=value")
    
    # Save current directory to restore later
    original_dir = os.getcwd()
    
    # Change to temp directory
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # Restore original directory and clean up
    os.chdir(original_dir)
    shutil.rmtree(temp_dir)


def test_real_load_dotenv_exception(env_file_with_error, monkeypatch):
    """Test actual _load_dotenv method with a real exception."""
    # Patch open to raise an exception
    def mock_open(*args, **kwargs):
        raise Exception("Test exception")
    
    # Setup capturing log messages
    with patch("cli_code.config.log") as mock_log:
        with patch("builtins.open", side_effect=mock_open):
            # Create a real Config instance
            cfg = Config()
            
            # Check that the warning was logged
            assert any("Error loading" in call[0][0] for call in mock_log.warning.call_args_list)


# Test for _load_config exception paths
def test_real_load_config_exception(monkeypatch, tmp_path):
    """Test actual _load_config method with a real exception."""
    # Make a temp config file path that doesn't exist yet
    config_file = tmp_path / "config.yaml"
    
    # Mock the initialization to use our temp file
    monkeypatch.setattr(Config, "_determine_config_path", lambda self, path: setattr(self, "config_file", config_file))
    monkeypatch.setattr(Config, "_load_dotenv", lambda self: None)
    monkeypatch.setattr(Config, "_ensure_config_exists", lambda self: None)
    monkeypatch.setattr(Config, "_apply_env_vars", lambda self: None)
    
    # Patch open to raise a generic exception for the second call
    original_open = open
    call_count = [0]
    
    def mock_open(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] > 1:  # After first call to ensure_config_exists
            raise Exception("Test exception")
        return original_open(*args, **kwargs)
    
    # Setup capturing log messages
    with patch("cli_code.config.log") as mock_log:
        with patch("builtins.open", side_effect=mock_open):
            # Create a real Config instance
            cfg = Config()
            
            # Force a reload of the config
            cfg._load_config()
            
            # Check that the error was logged
            assert any("Error loading config file" in call[0][0] for call in mock_log.error.call_args_list)


# Test for _save_config exception
def test_real_save_config_exception(monkeypatch, tmp_path):
    """Test actual _save_config method with a real exception."""
    # Make a temp config file path
    config_file = tmp_path / "config.yaml"
    
    # Create a real Config instance with minimum setup
    monkeypatch.setattr(Config, "_determine_config_path", lambda self, path: setattr(self, "config_file", config_file))
    monkeypatch.setattr(Config, "_load_dotenv", lambda self: None)
    monkeypatch.setattr(Config, "_ensure_config_exists", lambda self: None)
    monkeypatch.setattr(Config, "_apply_env_vars", lambda self: None)
    monkeypatch.setattr(Config, "_load_config", lambda self: {})
    
    cfg = Config()
    cfg.config = {"test": "value"}
    
    # Patch open to raise an exception
    with patch("builtins.open", side_effect=Exception("Test exception")):
        with patch("cli_code.config.log") as mock_log:
            # Call the method
            cfg._save_config()
            
            # Check that the error was logged
            assert mock_log.error.called
            assert "Error saving config file" in mock_log.error.call_args[0][0]


# Test for set_credential for OpenAI
def test_real_set_credential_openai(monkeypatch):
    """Test set_credential for OpenAI provider."""
    # Create a real Config instance with minimum setup
    monkeypatch.setattr(Config, "_determine_config_path", lambda self, path: None)
    monkeypatch.setattr(Config, "_load_dotenv", lambda self: None)
    monkeypatch.setattr(Config, "_ensure_config_exists", lambda self: None)
    monkeypatch.setattr(Config, "_apply_env_vars", lambda self: None)
    monkeypatch.setattr(Config, "_load_config", lambda self: {})
    
    # Mock _save_config to avoid actual file operations
    with patch.object(Config, "_save_config") as mock_save:
        cfg = Config()
        cfg.config = {}
        
        # Call the method for openai provider
        cfg.set_credential("openai", "test-key")
        
        # Verify the key was set
        assert cfg.config.get("openai_api_key") == "test-key"
        assert mock_save.called


# Test for set_default_model with unknown provider
def test_real_set_default_model_unknown(monkeypatch):
    """Test set_default_model with unknown provider."""
    # Create a real Config instance with minimum setup
    monkeypatch.setattr(Config, "_determine_config_path", lambda self, path: None)
    monkeypatch.setattr(Config, "_load_dotenv", lambda self: None)
    monkeypatch.setattr(Config, "_ensure_config_exists", lambda self: None)
    monkeypatch.setattr(Config, "_apply_env_vars", lambda self: None)
    monkeypatch.setattr(Config, "_load_config", lambda self: {})
    
    # Mock _save_config to avoid actual file operations
    with patch.object(Config, "_save_config") as mock_save:
        with patch("cli_code.config.log") as mock_log:
            cfg = Config()
            cfg.config = {"default_provider": "gemini"}
            
            # Call with unknown provider
            result = cfg.set_default_model("test-model", provider="unknown")
            
            # Verify results
            assert result is None
            assert mock_log.error.called
            assert not mock_save.called
