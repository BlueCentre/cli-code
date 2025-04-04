"""
Configuration management for Gemini CLI.
"""

import yaml
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class Config:
    """Manages configuration for the cli-code application."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "cli-code"
        self.config_file = self.config_dir / "config.yaml"
        self.config = {}
        try:
            self._ensure_config_exists()
            self.config = self._load_config()
            self._migrate_old_keys()
        except Exception as e:
            log.error(f"Error initializing configuration from {self.config_file}: {e}", exc_info=True)
    
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
                }
            }
            
            try:
                with open(self.config_file, 'w') as f:
                    yaml.dump(default_config, f)
                log.info(f"Created default config file at: {self.config_file}")
            except Exception as e:
                log.error(f"Failed to create default config file at {self.config_file}: {e}", exc_info=True)
                raise
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
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
            with open(self.config_file, 'w') as f:
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