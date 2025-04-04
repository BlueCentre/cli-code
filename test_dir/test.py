"""
A test file to verify that Gemini code tool calling works properly.
"""


def greet(name):
    """Say hello to someone."""
    return f"Hello, {name}!"


def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b


if __name__ == "__main__":
    print(greet("World"))
    print(f"2 + 2 = {calculate_sum(2, 2)}")
