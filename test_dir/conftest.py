"""
Pytest configuration and fixtures.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as requiring API keys")
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    """Process test items to skip tests with missing dependencies."""
    for item in items:
        if 'requires_tiktoken' in item.keywords and not _is_module_available('tiktoken'):
            item.add_marker(pytest.mark.skip(reason="tiktoken not available"))
        if 'requires_yaml' in item.keywords and not _is_module_available('yaml'):
            item.add_marker(pytest.mark.skip(reason="yaml not available"))
        if 'requires_gemini' in item.keywords and not _is_module_available('google.generativeai'):
            item.add_marker(pytest.mark.skip(reason="google.generativeai not available"))
        if 'requires_openai' in item.keywords and not _is_module_available('openai'):
            item.add_marker(pytest.mark.skip(reason="openai not available"))


def _is_module_available(module_name):
    """Check if a module is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


@pytest.fixture
def mock_module():
    """Create a MagicMock for a module."""
    return MagicMock()


@pytest.fixture
def temp_dir(tmpdir):
    """Provide a temporary directory."""
    return tmpdir 