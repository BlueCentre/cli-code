"""
Pytest configuration file that handles CI-specific test configuration.
This file ensures comprehensive tests are skipped in CI environments.
"""

import os

# Only import pytest if the module is available
try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


def pytest_ignore_collect(path, config):
    """Ignore tests containing '_comprehensive' in their path when CI=true."""
    # if os.environ.get("CI") == "true" and "_comprehensive" in str(path):
    #     print(f"Ignoring comprehensive test in CI: {path}")
    #     return True
    # return False
    pass  # Keep the function valid syntax, but effectively do nothing.
