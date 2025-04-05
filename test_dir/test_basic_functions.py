"""
Tests for basic functions defined (originally in test.py).
"""

# Assuming the functions to test are accessible
# If they were meant to be part of the main package, they should be moved
# or imported appropriately. For now, define them here for testing.


def greet(name):
    """Say hello to someone."""
    return f"Hello, {name}!"


def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b


# --- Pytest Tests ---


def test_greet():
    """Test the greet function."""
    assert greet("World") == "Hello, World!"
    assert greet("Alice") == "Hello, Alice!"
    assert greet("") == "Hello, !"


def test_calculate_sum():
    """Test the calculate_sum function."""
    assert calculate_sum(2, 2) == 4
    assert calculate_sum(0, 0) == 0
    assert calculate_sum(-1, 1) == 0
    assert calculate_sum(100, 200) == 300
