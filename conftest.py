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

def pytest_ignore_collect(path):
    """
    Determine which test files to ignore during collection.
    
    Args:
        path: Path object representing a test file or directory
    
    Returns:
        bool: True if the file should be ignored, False otherwise
    """
    # Check if we're running in CI
    in_ci = os.environ.get('CI', 'false').lower() == 'true'
    
    if in_ci:
        # Skip comprehensive test files in CI environments
        if '_comprehensive' in str(path):
            return True
    
    return False 