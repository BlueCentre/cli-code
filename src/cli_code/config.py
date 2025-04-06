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

import yaml

log = logging.getLogger(__name__)


class Config:
    """
    Configuration management for the CLI Code application.
    
    This class manages loading configuration from a YAML file, creating a
    default configuration file if one doesn't exist, and loading environment
    variables.
    
    The configuration is loaded in the following order of precedence:
    1. Environment variables (highest precedence)
    2. Configuration file
    3. Default values (lowest precedence)
    """

    def __init__(self):
        """
        Initialize the configuration.
        
        This will load environment variables, ensure the configuration file
        exists, and load the configuration from the file.
        """
        self.config_dir = Path(os.path.expanduser("~/.config/cli-code"))
        self.config_file = self.config_dir / "config.yaml"
        
        # Load environment variables
        self._load_dotenv()
        
        # Ensure config file exists
        self._ensure_config_exists()
        
        # Load config from file
        self.config = self._load_config()
        
        # Apply environment variable overrides
        self._apply_env_vars()

    def _load_dotenv(self):
        """Load environment variables from .env file if it exists."""
        env_file = Path(".env")
        env_example_file = Path(".env.example")

        if env_file.exists():
            try:
                log.info(f"Loading environment variables from {env_file.resolve()}")
                loaded_vars = []
                with open(env_file, "r") as f:
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

                            if key and value:
                                os.environ[key] = value
                                # Only add to list if it's a CLI_CODE variable to avoid logging sensitive data
                                if key.startswith("CLI_CODE_"):
                                    log_value = "****" if "KEY" in key or "TOKEN" in key else value
                                    loaded_vars.append(f"{key}={log_value}")

                if loaded_vars:
                    log.info(f"Loaded {len(loaded_vars)} CLI_CODE environment variables: {', '.join(loaded_vars)}")
                else:
                    log.debug("No CLI_CODE environment variables found in .env file")
            except Exception as e:
                log.warning(f"Error loading .env file: {e}", exc_info=True)
        elif env_example_file.exists():
            log.info(f".env file not found, but .env.example exists. Consider creating a .env file from the example.")
        else:
            log.debug("No .env or .env.example file found in current directory")

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
            'CLI_CODE_GOOGLE_API_KEY': 'google_api_key',
            'CLI_CODE_DEFAULT_PROVIDER': 'default_provider',
            'CLI_CODE_DEFAULT_MODEL': 'default_model',
            'CLI_CODE_OLLAMA_API_URL': 'ollama_api_url',
            'CLI_CODE_OLLAMA_DEFAULT_MODEL': 'ollama_default_model',
        }
        
        # Apply direct mappings
        for env_key, config_key in env_mappings.items():
            if env_key in os.environ:
                self.config[config_key] = os.environ[env_key]
        
        # Settings with CLI_CODE_SETTINGS_ prefix go into settings dict
        if 'settings' not in self.config:
            self.config['settings'] = {}
            
        for env_key, env_value in os.environ.items():
            if env_key.startswith('CLI_CODE_SETTINGS_'):
                setting_name = env_key[len('CLI_CODE_SETTINGS_'):].lower()
                
                # Try to convert to appropriate type (int, float, bool)
                if env_value.isdigit():
                    self.config['settings'][setting_name] = int(env_value)
                elif env_value.replace('.', '', 1).isdigit() and env_value.count('.') <= 1:
                    self.config['settings'][setting_name] = float(env_value)
                elif env_value.lower() in ('true', 'false'):
                    self.config['settings'][setting_name] = env_value.lower() == 'true'
                else:
                    self.config['settings'][setting_name] = env_value

    def _ensure_config_exists(self):
        """Create config directory and file with defaults if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            default_config = {
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
        if provider == "gemini":
            return self.config.get("google_api_key")
        elif provider == "ollama":
            return self.config.get("ollama_api_url")
        else:
            log.warning(f"Attempted to get credential for unknown provider: {provider}")
            return None

    def set_credential(self, provider: str, credential: str):
        """Set the credential (API key or URL) for a specific provider."""
        if provider == "gemini":
            self.config["google_api_key"] = credential
        elif provider == "ollama":
            self.config["ollama_api_url"] = credential
        else:
            log.error(f"Attempted to set credential for unknown provider: {provider}")
            return
        self._save_config()

    def get_default_provider(self) -> str:
        """Get the default provider."""
        return self.config.get("default_provider", "gemini")

    def set_default_provider(self, provider: str):
        """Set the default provider."""
        if provider in ["gemini", "ollama"]:
            self.config["default_provider"] = provider
            self._save_config()
        else:
            log.error(f"Attempted to set unknown default provider: {provider}")

    def get_default_model(self, provider: str | None = None) -> str | None:
        """Get the default model, optionally for a specific provider."""
        target_provider = provider or self.get_default_provider()
        if target_provider == "gemini":
            return self.config.get("default_model") or "models/gemini-2.5-pro-exp-03-25"
        elif target_provider == "ollama":
            return self.config.get("ollama_default_model")
        else:
            return self.config.get("default_model")

    def set_default_model(self, model: str, provider: str | None = None):
        """Set the default model for a specific provider (or the default provider if None)."""
        target_provider = provider or self.get_default_provider()
        if target_provider == "gemini":
            self.config["default_model"] = model
        elif target_provider == "ollama":
            self.config["ollama_default_model"] = model
        else:
            log.error(f"Cannot set default model for unknown provider: {target_provider}")
            return
        self._save_config()

    def get_setting(self, setting, default=None):
        """Get a specific setting value from the 'settings' section."""
        return self.config.get("settings", {}).get(setting, default)

    def set_setting(self, setting, value):
        """Set a specific setting value in the 'settings' section."""
        if "settings" not in self.config:
            self.config["settings"] = {}
        self.config["settings"][setting] = value
        self._save_config()
