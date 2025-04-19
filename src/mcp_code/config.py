# chuk_mcp/config.py
import json
import logging
from typing import Any, Dict, Union

# mcp_client imports
from mcp_code.mcp_client.transport.stdio.stdio_server_parameters import StdioServerParameters


async def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the server configuration dictionary from a JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        The entire configuration dictionary loaded from the JSON file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the configuration file contains invalid JSON.
        ValueError: Propagated from lower levels if needed (though less likely now).
    """
    try:
        logging.debug(f"Loading config from {config_path}")

        # Read the configuration file
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        logging.debug(f"Loaded config dictionary: {config}")
        return config

    except FileNotFoundError:
        # Log an error and raise the exception
        error_msg = f"Configuration file not found: {config_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg) from None
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in configuration file: {e.msg}"
        logging.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e
    except Exception as e:
        # Catch any other unexpected errors during loading
        error_msg = f"An unexpected error occurred while loading config: {config_path} - {e}"
        logging.exception(error_msg)  # Log the full traceback
        raise  # Re-raise the original exception
