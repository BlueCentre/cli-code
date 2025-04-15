# Testing Guide

This document provides guidelines and best practices for writing and maintaining tests for the CLI Code project.

## Table of Contents

1. [Testing Structure](#testing-structure)
2. [Running Tests](#running-tests)
3. [Mock Objects and API Interactions](#mock-objects-and-api-interactions)
4. [API Version Compatibility](#api-version-compatibility)
5. [Lessons Learned](#lessons-learned)

## Testing Structure

Tests are organized in the `tests/` directory, organized by module (e.g., `tests/models`, `tests/tools`).

Test file naming follows these conventions:

- Basic test files: `test_<component>.py`
- Coverage-focused tests: `test_<component>_coverage.py`
- Tests for edge cases: `test_<component>_edge_cases.py`
- Advanced/comprehensive tests: `test_<component>_comprehensive.py`

## Running Tests

### Running All Tests

```bash
python -m pytest
```

### Running with Coverage

```bash
python -m pytest --cov=src
```

### Running Specific Tests

```bash
# Run tests in a specific file
python -m pytest tests/models/test_gemini.py

# Run a specific test
python -m pytest tests/models/test_gemini.py::test_generate_simple_text_response
```

## Mock Objects and API Interactions

When testing components that interact with external APIs (like Gemini or Ollama), proper mocking is essential. Here are some guidelines:

### Creating Mock Objects

Use `mocker.MagicMock()` (provided by pytest-mock) instead of direct `unittest.mock.MagicMock` when creating mock objects:

```python
# Preferred
mock_object = mocker.MagicMock()

# Avoid using spec unless necessary
# Avoid: mock_object = mock.MagicMock(spec=SomeClass)
```

### Mocking Response Objects

When mocking API response objects:

1. Build mock objects hierarchically from inside out
2. Set all necessary attributes explicitly
3. Avoid using `__getattr__` or other magic methods in mocks
4. For complex objects, create separate variables for each level to keep the code readable

Example:
```python
# Create the innermost part
mock_response_part = mocker.MagicMock()
mock_response_part.text = "Hello, world"
mock_response_part.function_call = None

# Create the content object that contains parts
mock_content = mocker.MagicMock()
mock_content.parts = [mock_response_part]
mock_content.role = "model"

# Create the candidate object that contains content
mock_candidate = mocker.MagicMock()
mock_candidate.content = mock_content
mock_candidate.finish_reason = "STOP"

# Create the final response
mock_api_response = mocker.MagicMock()
mock_api_response.candidates = [mock_candidate]
```

### Mocking User Interaction

When mocking confirmation prompts (e.g., `questionary.confirm`):

```python
# Create a mock object that has an .ask method
mock_confirm_obj = mocker.MagicMock()
mock_confirm_obj.ask.return_value = True  # or False
mock_confirm = mocker.patch("path.to.questionary.confirm", return_value=mock_confirm_obj)
```

## API Version Compatibility

External APIs evolve over time, which can break tests. Follow these practices to make tests more resilient:

1. Use loose coupling to implementation details
2. Avoid importing classes directly from unstable APIs when possible
3. For required imports, use try/except blocks to handle missing imports
4. Consider using conditional test execution with `@pytest.mark.skipif`

Example of conditional imports:
```python
try:
    from google.generativeai.types.content_types import FunctionCallingMode as FunctionCall
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create mock class as fallback
    class FunctionCall: pass

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
def test_feature_requiring_imports():
    # Test code here
```

## Lessons Learned

### Gemini API Testing

Recent work with the Google Generative AI (Gemini) API highlighted several key lessons:

1. **API Structure Evolution**: The Gemini API structure has changed over time. Classes like `Candidate`, `Content`, and `FunctionCall` have moved between modules.

2. **Import Strategies**:
   - Import specifically from submodules rather than top-level packages
   - Use alternative imports when direct imports aren't available:
     ```python
     # Instead of
     from google.generativeai.types import Candidate
     # Use
     from google.ai.generativelanguage_v1beta.types.generative_service import Candidate
     ```

3. **Mock Object Limitations**:
   - Setting `__getattr__` on mock objects isn't supported
   - Using `.spec` can make mocks too restrictive
   - Mock objects directly with the attributes they need instead of trying to mimic class behavior exactly

4. **Test Assertions**:
   - Focus assertions on behavior, not implementation
   - Verify key interactions rather than every intermediate step
   - For error messages, match the message pattern rather than expecting exact strings

5. **Questionary Mocking**:
   - Mocking `questionary.confirm()` requires special attention since it returns an object with an `.ask()` method
   - Create a proper mock structure: `mock_confirm_obj.ask.return_value = True/False`

### Maintaining Test Stability

1. **Focus on Key Behaviors**: Test that the core functionality works, not the implementation details.

2. **Isolate External Dependencies**: Always mock external dependencies to prevent tests from being impacted by API changes or availability.

3. **Regular Updates**: Update tests when APIs change, focusing on the behavior rather than the exact implementation.

4. **Error Handling**: Include proper error handling in tests to make them more robust against changes.

### Known Test Workarounds

1. **Gemini Agent Loop Issues**: The Gemini agent loop has limitations in handling sequences of tool calls.
   - Several tests in `tests/models/test_gemini.py` have modified assertions to accommodate these limitations:
     - `test_generate_simple_tool_call` has commented-out assertions for the second tool execution (`mock_task_complete_tool.execute`) and final result check.
     - History count assertions are adjusted to reflect actual behavior rather than ideal behavior.
   - When writing new tests that involve sequential tool calls, be aware of these limitations and adjust assertions accordingly.
   - If you're improving the agent loop functionality, consult `TODO_gemini_loop.md` for details on remaining issues.

2. **Mock API Response Structure**: Some tests may have extra or adjusted mock structures to handle the model's specific response processing.
   - Look for comments like `# Mock response adapted for agent loop` to identify these cases.
   - When updating these tests, ensure you maintain the adjusted structure until the underlying issues are resolved.
