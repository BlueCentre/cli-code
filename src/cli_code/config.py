"""
Configuration management for the CLI Code application.

This module provides a Config class that manages configuration for the CLI Code
application, including loading environment variables, ensuring the configuration
file exists, and loading and saving configuration.

Configuration in CLI Code supports two approaches:
1. File-based configuration (.yaml): Primary approach for end users who install from pip
2. Environment variables: Used mainly during development for quick experimentation

Both approaches are supported simultaneously - there is no migration needed as both
configuration methods can coexist. Environment variables take precedence over file-based
configuration when both are present.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union  # Added Optional and Union

import yaml

log = logging.getLogger(__name__)


class Config:
    """
    Configuration management for the CLI Code application.

    This class manages loading configuration from a YAML file, creating a
    default configuration file if one doesn't exist, and loading environment
    variables.

    The configuration file path can be overridden in the following order of precedence:
    1. `config_file_path` parameter passed during initialization.
    2. `CLI_CODE_CONFIG_FILE` environment variable.
    3. Default path (`~/.config/cli-code/config.yaml`).

    Configuration values are loaded in the following order of precedence:
    1. Environment variables (highest precedence)
    2. Configuration file
    3. Default values (lowest precedence)
    """

    def __init__(self, config_file_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration.

        Loads environment variables, determines the configuration file path based on
        the provided parameter, environment variables, or defaults, ensures the
        configuration file exists, and loads the configuration from the file.

        Args:
            config_file_path: Optional path to the configuration file. If provided,
                              it overrides the environment variable and default path.
        """
        # Determine config file path
        self._determine_config_path(config_file_path)

        # Load environment variables (consider loading after determining path?)
        # No, .env loading should happen relative to CWD, not config path.
        self._load_dotenv()

        # Ensure config file exists at the determined path
        self._ensure_config_exists()

        # Load config from the determined file
        self.config = self._load_config()

        # Apply environment variable overrides
        self._apply_env_vars()

    def _determine_config_path(self, config_file_path_param: Optional[Union[str, Path]] = None):
        """Determine the configuration file path based on parameter, env var, or default."""
        config_path_str = None
        source = "default"

        # 1. Check __init__ parameter
        if config_file_path_param:
            config_path_str = str(config_file_path_param)
            source = "parameter"
        # 2. Check environment variable
        elif "CLI_CODE_CONFIG_FILE" in os.environ:
            config_path_str = os.environ["CLI_CODE_CONFIG_FILE"]
            source = "environment variable"
        # 3. Use default path
        else:
            home_dir = Path(os.path.expanduser("~"))
            default_dir = home_dir / ".config" / "cli-code"
            self.config_dir = default_dir
            self.config_file = default_dir / "config.yaml"
            log.debug(f"Using default config path: {self.config_file}")
            return  # Already set default paths

        if not config_path_str:
            # Fallback if env var was empty string (should not happen with default logic, but safety)
            log.warning("Config file path resolved to empty, falling back to default.")
            home_dir = Path(os.path.expanduser("~"))
            default_dir = home_dir / ".config" / "cli-code"
            self.config_dir = default_dir
            self.config_file = default_dir / "config.yaml"
            source = "fallback default"
        else:
            # Resolve the path (handles relative paths, expands ~)
            self.config_file = Path(config_path_str).expanduser().resolve()
            self.config_dir = self.config_file.parent
            log.info(f"Using config path from {source}: {self.config_file}")

    def _load_dotenv(self):
        """Load environment variables from .env file if it exists, fallback to .env.example."""
        env_file = Path(".env")
        env_example_file = Path(".env.example")
        file_to_load = None
        loaded_source = None

        if env_file.exists():
            file_to_load = env_file
            loaded_source = ".env"
        elif env_example_file.exists():
            file_to_load = env_example_file
            loaded_source = ".env.example"
            log.info(".env not found, loading from .env.example.")
        else:
            log.debug("No .env or .env.example file found.")
            return

        if file_to_load:
            try:
                log.info(f"Loading environment variables from {file_to_load.resolve()}")
                loaded_vars_log = []
                with open(file_to_load, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        if "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip()

                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or (
                                value.startswith("'") and value.endswith("'")
                            ):
                                value = value[1:-1]

                            # Fix: Set env var even if value is empty, but only if key exists
                            if key:
                                os.environ[key] = value
                                # Only add to log list if it's a CLI_CODE variable
                                if key.startswith("CLI_CODE_"):
                                    log_value = "****" if "KEY" in key or "TOKEN" in key else value
                                    loaded_vars_log.append(f"{key}={log_value}")
                        else:
                            # Handle lines without '=' if necessary, or log warning
                            log.debug(f"Skipping line without '=' in {loaded_source}: {line}")

                if loaded_vars_log:
                    log.info(
                        f"Loaded {len(loaded_vars_log)} CLI_CODE vars from {loaded_source}: {', '.join(loaded_vars_log)}"
                    )
                else:
                    log.debug(f"No CLI_CODE environment variables found in {loaded_source}")
            except Exception as e:
                log.warning(f"Error loading {loaded_source} file: {e}", exc_info=True)

    def _apply_env_vars(self):
        """
        Apply environment variable overrides to the configuration.

        Environment variables take precedence over configuration file values.
        Environment variables are formatted as:
        CLI_CODE_SETTING_NAME

        For example:
        CLI_CODE_GOOGLE_API_KEY=my-api-key
        CLI_CODE_DEFAULT_PROVIDER=gemini
        CLI_CODE_SETTINGS_MAX_TOKENS=4096
        """

        # Direct mappings from env to config keys
        env_mappings = {
            "CLI_CODE_GOOGLE_API_KEY": "google_api_key",
            "CLI_CODE_DEFAULT_PROVIDER": "default_provider",
            "CLI_CODE_DEFAULT_MODEL": "default_model",
            "CLI_CODE_OLLAMA_API_URL": "ollama_api_url",
            "CLI_CODE_OLLAMA_DEFAULT_MODEL": "ollama_default_model",
        }

        # Apply direct mappings
        for env_key, config_key in env_mappings.items():
            if env_key in os.environ:
                self.config[config_key] = os.environ[env_key]

        # Settings with CLI_CODE_SETTINGS_ prefix go into settings dict
        if "settings" not in self.config:
            self.config["settings"] = {}

        for env_key, env_value in os.environ.items():
            if env_key.startswith("CLI_CODE_SETTINGS_"):
                setting_name = env_key[len("CLI_CODE_SETTINGS_") :].lower()

                # Try to convert to appropriate type (int, float, bool)
                if env_value.isdigit():
                    self.config["settings"][setting_name] = int(env_value)
                elif env_value.replace(".", "", 1).isdigit() and env_value.count(".") <= 1:
                    self.config["settings"][setting_name] = float(env_value)
                elif env_value.lower() in ("true", "false"):
                    self.config["settings"][setting_name] = env_value.lower() == "true"
                else:
                    self.config["settings"][setting_name] = env_value

    def _ensure_config_exists(self):
        """Create config directory and file with defaults if they don't exist."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error(f"Failed to create config directory {self.config_dir}: {e}", exc_info=True)
            # Decide if we should raise here or just log and potentially fail later
            # For now, log and continue, config loading will likely fail
            return  # Exit early if dir creation fails

        if not self.config_file.exists():
            default_config = {
                "google_api_key": None,
                "default_provider": "gemini",
                "default_model": "models/gemini-2.5-pro-exp-03-25",
                "ollama_api_url": None,  # http://localhost:11434/v1
                "ollama_default_model": "llama3.2",
                "settings": {
                    "max_tokens": 1000000,
                    "temperature": 0.5,
                    "token_warning_threshold": 800000,
                    "auto_compact_threshold": 950000,
                },
            }

            try:
                with open(self.config_file, "w") as f:
                    yaml.dump(default_config, f)
                log.info(f"Created default config file at: {self.config_file}")
            except Exception as e:
                log.error(f"Failed to create default config file at {self.config_file}: {e}", exc_info=True)
                raise

    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            log.warning(f"Config file not found at {self.config_file}. A default one will be created.")
            return {}
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML config file {self.config_file}: {e}")
            return {}
        except Exception as e:
            log.error(f"Error loading config file {self.config_file}: {e}", exc_info=True)
            return {}

    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            log.error(f"Error saving config file {self.config_file}: {e}", exc_info=True)

    def get_credential(self, provider: str) -> str | None:
        """Get the credential (API key or URL) for a specific provider."""
        if provider == "gemini" or provider == "google":  # Added "google" as an alias for "gemini"
            return self.config.get("google_api_key")
        elif provider == "ollama":
            return self.config.get("ollama_api_url")
        else:
            log.warning(f"Attempted to get credential for unknown provider: {provider}")
            return None

    def set_credential(self, provider: str, credential: str):
        """Set the credential (API key or URL) for a specific provider."""
        if provider == "gemini" or provider == "google":  # Added "google" as an alias for "gemini"
            self.config["google_api_key"] = credential
        elif provider == "ollama":
            self.config["ollama_api_url"] = credential
        elif provider == "openai":  # Added support for openai provider
            self.config["openai_api_key"] = credential
        else:
            log.error(f"Attempted to set credential for unknown provider: {provider}")
            return
        self._save_config()

    def get_default_provider(self) -> str:
        """Get the default provider."""
        if not self.config:
            return "gemini"  # Default if config is None
        # Return "gemini" as fallback if default_provider is None or not set
        return self.config.get("default_provider") or "gemini"

    def set_default_provider(self, provider: str):
        """Set the default provider."""
        if provider is None:  # Handle None by setting default to gemini
            self.config["default_provider"] = "gemini"
            self._save_config()
        elif provider in ["gemini", "ollama", "openai", "anthropic"]:  # Added "openai" and "anthropic"
            self.config["default_provider"] = provider
            self._save_config()
        else:
            log.error(f"Attempted to set unknown default provider: {provider}")

    def get_default_model(self, provider: str | None = None) -> str | None:
        """Get the default model, optionally for a specific provider."""
        # Handle if config is None early
        if not self.config:
            # Return hardcoded defaults if config doesn't exist
            temp_provider = provider or "gemini"  # Determine provider based on input or default
            if temp_provider == "gemini":
                return "models/gemini-1.5-pro-latest"  # Or your actual default
            elif temp_provider == "ollama":
                return "llama2"  # Or your actual default
            else:
                return None

        target_provider = provider or self.get_default_provider()
        if target_provider == "gemini":
            # Use actual default from constants or hardcoded
            return self.config.get("default_model", "models/gemini-1.5-pro-latest")
        elif target_provider == "ollama":
            # Use actual default from constants or hardcoded
            return self.config.get("ollama_default_model", "llama2")
        elif target_provider in ["openai", "anthropic"]:
            # Handle known providers that might have specific config keys
            return self.config.get(f"{target_provider}_default_model")
        else:
            # Return None for unknown providers
            log.warning(f"Attempted to get default model for unknown provider: {target_provider}")
            return None

    def set_default_model(self, model: str, provider: str | None = None):
        """Set the default model, optionally for a specific provider."""
        target_provider = provider or self.get_default_provider()
        if target_provider == "gemini":
            self.config["default_model"] = model
        elif target_provider == "ollama":
            self.config["ollama_default_model"] = model
        elif target_provider == "anthropic":  # Added support for anthropic provider
            self.config["anthropic_default_model"] = model
        else:
            log.error(f"Cannot set default model for unknown provider: {target_provider}")
            return None
        self._save_config()

    def get_setting(self, setting, default=None):
        """Get a specific setting value from the 'settings' section."""
        settings_dict = self.config.get("settings", {}) if self.config else {}
        # Handle case where 'settings' key exists but value is None, or self.config is None
        if settings_dict is None:
            settings_dict = {}
        return settings_dict.get(setting, default)

    def set_setting(self, setting, value):
        """Set a specific setting value in the 'settings' section."""
        # Handle if config is None
        if self.config is None:
            log.warning("Attempted to set setting, but config is not loaded. Ignoring.")
            # Or initialize self.config = {} here if preferred?
            # For now, just return to avoid error
            return

        if "settings" not in self.config or self.config["settings"] is None:
            self.config["settings"] = {}
        self.config["settings"][setting] = value
        self._save_config()
