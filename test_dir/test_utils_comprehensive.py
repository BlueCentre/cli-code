"""
Comprehensive tests for the utils module.
"""

import unittest
import pytest
from cli_code.utils import count_tokens


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


def test_count_tokens_mocked_failure(monkeypatch):
    """Test the fallback method when tiktoken raises an exception."""
    def mock_encoding_that_fails(*args, **kwargs):
        raise ImportError("Simulated import error")
    
    # Mock the tiktoken encoding to simulate a failure
    monkeypatch.setattr("tiktoken.encoding_for_model", mock_encoding_that_fails)
    
    # Test that the function returns a value using the fallback method
    text = "This is a test string"
    expected_approx = len(text) // 4
    result = count_tokens(text)
    
    # The fallback method is approximate, but should be close to this value
    assert result == expected_approx 