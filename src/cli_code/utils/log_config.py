"""
Placeholder for Logging Configuration utility.
"""

import logging


def get_logger(name):
    """Placeholder function to get a logger instance."""
    logger = logging.getLogger(name)
    # Basic configuration if not already configured
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Set a default level (e.g., DEBUG)
        logger.propagate = False  # Prevent duplicate logging if root logger is configured
    print(f"[Debug] Logger '{name}' requested/configured.")  # Debug print
    return logger
