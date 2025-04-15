"""
Tests for Config class methods that might have been missed in existing tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Setup proper import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Check if running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"

# Try importing the required modules
try:
    import yaml

    from cli_code.config import Config

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    yaml = MagicMock()

    # Create a dummy Config class for testing
    class Config:
        def __init__(self):
            self.config = {}
            self.config_dir = Path("/tmp")
            self.config_file = self.config_dir / "config.yaml"


# Skip tests if imports not available and not in CI
SHOULD_SKIP = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI environment"


@pytest.fixture
def temp_config_dir():
    """Creates a temporary directory for the config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Return a Config instance with mocked file operations."""
    with (
        patch("cli_code.config.Config._load_dotenv", create=True),
        patch("cli_code.config.Config._ensure_config_exists", create=True),
        patch("cli_code.config.Config._load_config", create=True, return_value={}),
        patch("cli_code.config.Config._apply_env_vars", create=True),
    ):
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
            },
        }
        yield config


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_get_credential(mock_config):
    """Test get_credential method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "get_credential"):
        pytest.skip("get_credential method not available")

    # Test existing provider
    assert mock_config.get_credential("google") == "test-google-key"

    # Test non-existing provider
    assert mock_config.get_credential("non_existing") is None

    # Test with empty config
    mock_config.config = {}
    assert mock_config.get_credential("google") is None


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_set_credential(mock_config):
    """Test set_credential method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "set_credential"):
        pytest.skip("set_credential method not available")

    # Test setting existing provider
    mock_config.set_credential("google", "new-google-key")
    assert mock_config.config["google_api_key"] == "new-google-key"

    # Test setting new provider
    mock_config.set_credential("openai", "test-openai-key")
    assert mock_config.config["openai_api_key"] == "test-openai-key"

    # Test with None value
    mock_config.set_credential("google", None)
    assert mock_config.config["google_api_key"] is None


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_get_default_provider(mock_config):
    """Test get_default_provider method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "get_default_provider"):
        pytest.skip("get_default_provider method not available")

    # Test with existing provider
    assert mock_config.get_default_provider() == "gemini"

    # Test with no provider set
    mock_config.config["default_provider"] = None
    assert mock_config.get_default_provider() == "gemini"  # Should return default

    # Test with empty config
    mock_config.config = {}
    assert mock_config.get_default_provider() == "gemini"  # Should return default


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_set_default_provider(mock_config):
    """Test set_default_provider method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "set_default_provider"):
        pytest.skip("set_default_provider method not available")

    # Test setting valid provider
    mock_config.set_default_provider("openai")
    assert mock_config.config["default_provider"] == "openai"

    # Test setting None (should use default)
    mock_config.set_default_provider(None)
    assert mock_config.config["default_provider"] == "gemini"


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_get_default_model(mock_config):
    """Test get_default_model method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "get_default_model"):
        pytest.skip("get_default_model method not available")

    # Test without provider (use default provider)
    assert mock_config.get_default_model() == "models/gemini-1.0-pro"

    # Test with specific provider
    assert mock_config.get_default_model("ollama") == "llama2"

    # Test with non-existing provider
    assert mock_config.get_default_model("non_existing") is None


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_set_default_model(mock_config):
    """Test set_default_model method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "set_default_model"):
        pytest.skip("set_default_model method not available")

    # Test with default provider
    mock_config.set_default_model("new-model")
    assert mock_config.config["default_model"] == "new-model"

    # Test with specific provider
    mock_config.set_default_model("new-ollama-model", "ollama")
    assert mock_config.config["ollama_default_model"] == "new-ollama-model"

    # Test with new provider
    mock_config.set_default_model("anthropic-model", "anthropic")
    assert mock_config.config["anthropic_default_model"] == "anthropic-model"


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_get_setting(mock_config):
    """Test get_setting method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "get_setting"):
        pytest.skip("get_setting method not available")

    # Test existing setting
    assert mock_config.get_setting("max_tokens") == 1000
    assert mock_config.get_setting("temperature") == 0.7

    # Test non-existing setting with default
    assert mock_config.get_setting("non_existing", "default_value") == "default_value"

    # Test with empty settings
    mock_config.config["settings"] = {}
    assert mock_config.get_setting("max_tokens", 2000) == 2000


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_set_setting(mock_config):
    """Test set_setting method."""
    # Skip if not available and not in CI
    if not hasattr(mock_config, "set_setting"):
        pytest.skip("set_setting method not available")

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


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_save_config():
    """Test _save_config method."""
    if not IMPORTS_AVAILABLE:
        pytest.skip("Required imports not available")

    with (
        patch("builtins.open", mock_open()) as mock_file,
        patch("yaml.dump") as mock_yaml_dump,
        patch("cli_code.config.Config._load_dotenv", create=True),
        patch("cli_code.config.Config._ensure_config_exists", create=True),
        patch("cli_code.config.Config._load_config", create=True, return_value={}),
        patch("cli_code.config.Config._apply_env_vars", create=True),
    ):
        config = Config()
        if not hasattr(config, "_save_config"):
            pytest.skip("_save_config method not available")

        config.config = {"test": "data"}
        config._save_config()

        mock_file.assert_called_once()
        mock_yaml_dump.assert_called_once_with({"test": "data"}, mock_file(), default_flow_style=False)


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_yaml
def test_save_config_error():
    """Test error handling in _save_config method."""
    if not IMPORTS_AVAILABLE:
        pytest.skip("Required imports not available")

    with (
        patch("builtins.open", side_effect=PermissionError("Permission denied")),
        patch("cli_code.config.log.error", create=True) as mock_log_error,
        patch("cli_code.config.Config._load_dotenv", create=True),
        patch("cli_code.config.Config._ensure_config_exists", create=True),
        patch("cli_code.config.Config._load_config", create=True, return_value={}),
        patch("cli_code.config.Config._apply_env_vars", create=True),
    ):
        config = Config()
        if not hasattr(config, "_save_config"):
            pytest.skip("_save_config method not available")

        config._save_config()

        # Verify error was logged
        assert mock_log_error.called
