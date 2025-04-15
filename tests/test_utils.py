"""
Tests for utility functions in src/cli_code/utils.py.
"""

from unittest.mock import MagicMock, patch

import pytest

# Force module import for coverage
import src.cli_code.utils

# Update import to use absolute import path including 'src'
from src.cli_code.utils import count_tokens


def test_count_tokens_simple():
    """Test count_tokens with simple strings using tiktoken."""
    # These counts are based on gpt-4 tokenizer via tiktoken
    assert count_tokens("Hello world") == 2
    assert count_tokens("This is a test.") == 5
    assert count_tokens("") == 0
    assert count_tokens("   ") == 1  # Spaces are often single tokens


def test_count_tokens_special_chars():
    """Test count_tokens with special characters using tiktoken."""
    assert count_tokens("Hello, world! How are you?") == 8
    # Emojis can be multiple tokens
    # Note: Actual token count for emojis can vary
    assert count_tokens("Testing emojis 👍🚀") > 3


@patch("tiktoken.encoding_for_model")
def test_count_tokens_tiktoken_fallback(mock_encoding_for_model):
    """Test count_tokens fallback mechanism when tiktoken fails."""
    # Simulate tiktoken raising an exception
    mock_encoding_for_model.side_effect = Exception("Tiktoken error")

    # Test fallback (length // 4)
    assert count_tokens("This is exactly sixteen chars") == 7  # 28 // 4
    assert count_tokens("Short") == 1  # 5 // 4
    assert count_tokens("") == 0  # 0 // 4
    assert count_tokens("123") == 0  # 3 // 4
    assert count_tokens("1234") == 1  # 4 // 4


@patch("tiktoken.encoding_for_model")
def test_count_tokens_tiktoken_mocked_success(mock_encoding_for_model):
    """Test count_tokens main path with tiktoken mocked."""
    # Create a mock encoding object with a mock encode method
    mock_encode = MagicMock()
    mock_encode.encode.return_value = [1, 2, 3, 4, 5]  # Simulate encoding returning 5 tokens

    # Configure the mock context manager returned by encoding_for_model
    mock_encoding_for_model.return_value = mock_encode

    assert count_tokens("Some text that doesn't matter now") == 5
    mock_encoding_for_model.assert_called_once_with("gpt-4")
    mock_encode.encode.assert_called_once_with("Some text that doesn't matter now")
