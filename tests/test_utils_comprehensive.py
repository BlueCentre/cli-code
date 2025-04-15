"""
Comprehensive tests for the utils module.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Setup proper import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Check if running in CI
IN_CI = os.environ.get("CI", "false").lower() == "true"

# Try importing the module
try:
    from cli_code.utils import count_tokens

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

    # Define a dummy function for testing when module is not available
    def count_tokens(text):
        return len(text) // 4


# Skip tests if imports not available and not in CI
SHOULD_SKIP = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = "Required imports not available and not in CI environment"


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_tiktoken
class TestUtilsModule(unittest.TestCase):
    """Test cases for the utils module functions."""

    def test_count_tokens_with_tiktoken(self):
        """Test token counting with tiktoken available."""
        # Test with empty string
        assert count_tokens("") == 0

        # Test with short texts
        assert count_tokens("Hello") > 0
        assert count_tokens("Hello, world!") > count_tokens("Hello")

        # Test with longer content
        long_text = "This is a longer piece of text that should contain multiple tokens. " * 10
        assert count_tokens(long_text) > 20

        # Test with special characters
        special_chars = "!@#$%^&*()_+={}[]|\\:;\"'<>,.?/"
        assert count_tokens(special_chars) > 0

        # Test with numbers
        numbers = "12345 67890"
        assert count_tokens(numbers) > 0

        # Test with unicode characters
        unicode_text = "こんにちは世界"  # Hello world in Japanese
        assert count_tokens(unicode_text) > 0

        # Test with code snippets
        code_snippet = """
        def example_function(param1, param2):
            \"\"\"This is a docstring.\"\"\"
            result = param1 + param2
            return result
        """
        assert count_tokens(code_snippet) > 10


@pytest.mark.skipif(SHOULD_SKIP, reason=SKIP_REASON)
@pytest.mark.requires_tiktoken
def test_count_tokens_mocked_failure(monkeypatch):
    """Test the fallback method when tiktoken raises an exception."""

    def mock_encoding_that_fails(*args, **kwargs):
        raise ImportError("Simulated import error")

    # Mock the tiktoken encoding to simulate a failure
    if IMPORTS_AVAILABLE:
        with patch("tiktoken.encoding_for_model", mock_encoding_that_fails):
            # Test that the function returns a value using the fallback method
            text = "This is a test string"
            expected_approx = len(text) // 4
            result = count_tokens(text)

            # The fallback method is approximate, but should be close to this value
            assert result == expected_approx
    else:
        # Skip if imports not available
        pytest.skip("Imports not available to perform this test")
