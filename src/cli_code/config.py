"""
Configuration management for Gemini CLI.
"""

import logging
import os
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


class Config:
    """Manages configuration for the cli-code application."""

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "cli-code-agent"
        self.config_file = self.config_dir / "config.yaml"
        self.config = {}
        
        # First load environment variables from .env file if it exists
        self._load_dotenv()
        
        try:
            self._ensure_config_exists()
            self.config = self._load_config()
            self._migrate_old_keys()
            
            # Override config with environment variables if they exist
            self._apply_env_vars()
        except Exception as e:
            log.error(f"Error initializing configuration from {self.config_file}: {e}", exc_info=True)

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
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
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
        """Apply environment variables to override config settings."""
        # Map of environment variable names to config keys
        env_var_mapping = {
            "CLI_CODE_GOOGLE_API_KEY": "google_api_key",
            "CLI_CODE_OLLAMA_API_URL": "ollama_api_url",
            "CLI_CODE_DEFAULT_PROVIDER": "default_provider",
            "CLI_CODE_DEFAULT_MODEL": "default_model",
            "CLI_CODE_OLLAMA_DEFAULT_MODEL": "ollama_default_model"
        }
        
        for env_var, config_key in env_var_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Mask sensitive values in logs
                log_value = "****" if "KEY" in env_var or "TOKEN" in env_var else value
                log.info(f"Using environment variable {env_var}={log_value} to override config key '{config_key}'")
                self.config[config_key] = value
                
        # Apply and save if environment variables were found
        if any(env_var in os.environ for env_var in env_var_mapping):
            self._save_config()
            # Log all the current config values for debugging
            safe_config = self.config.copy()
            if "google_api_key" in safe_config and safe_config["google_api_key"]:
                safe_config["google_api_key"] = "****"
            log.debug(f"Final config after applying environment variables: {safe_config}")

    def _ensure_config_exists(self):
        """Create config directory and file with defaults if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            default_config = {
                "google_api_key": None,
                "ollama_api_url": None,
                "default_provider": "gemini",
                "default_model": "models/gemini-2.5-pro-exp-03-25",
                "ollama_default_model": None,
                "settings": {
                    "max_tokens": 1000000,
                    "temperature": 0.7,
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

    def _migrate_old_keys(self):
        """Migrate from old nested 'api_keys': {'google': ...} structure if present."""
        if "api_keys" in self.config and isinstance(self.config["api_keys"], dict):
            log.info("Migrating old 'api_keys' structure in config file.")
            if "google" in self.config["api_keys"] and "google_api_key" not in self.config:
                self.config["google_api_key"] = self.config["api_keys"]["google"]
            del self.config["api_keys"]
            self._save_config()
            log.info("Finished migrating 'api_keys'.")
        
        # Check for old config paths and migrate if needed
        self._migrate_old_config_paths()
    
    def _migrate_old_config_paths(self):
        """Check for and migrate config from older versions with different path names."""
        old_paths = [
            Path.home() / ".config" / "gemini-code" / "config.yaml",
            Path.home() / ".config" / "cli-code" / "config.yaml"
        ]
        
        for old_path in old_paths:
            if old_path.exists() and not self.config_file.exists():
                log.info(f"Found old config at {old_path}. Migrating to {self.config_file}...")
                try:
                    # Ensure new directory exists
                    self.config_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Read old config
                    with open(old_path, "r") as old_file:
                        old_config = yaml.safe_load(old_file) or {}
                    
                    # Update our config with old values
                    self.config.update(old_config)
                    
                    # Save to new location
                    self._save_config()
                    log.info(f"Successfully migrated config from {old_path} to {self.config_file}")
                except Exception as e:
                    log.error(f"Error migrating config from {old_path}: {e}", exc_info=True)
                    # Continue trying other paths on failure

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
        """Get a specific setting."""
        return self.config.get("settings", {}).get(setting, default)

    def set_setting(self, setting, value):
        """Set a specific setting."""
        if "settings" not in self.config:
            self.config["settings"] = {}

        self.config["settings"][setting] = value
        self._save_config()
