# Known Issues

This document tracks known issues within the `cli-code` project, covering both application behavior and test suite status. It is intended to help users and contributors understand the current state and limitations.

## Testing Issues

Currently, the primary known issues relate to the test suite (`pytest`).

### Mocking `load_config` in `test_server_manager.py`

*   **File:** `tests/mcp_tests/mcp_client/host/test_server_manager.py`
*   **Problem:** 10 tests related to the `run_command` function are skipped. These tests consistently fail because attempts to mock the asynchronous `load_config` function (defined in `src/mcp_code/config.py`) have been unsuccessful when run within the `anyio` task group used by `run_command`. Standard mocking techniques (`patch`, `AsyncMock`, `side_effect` with `async def`) failed to intercept the call correctly; the real `load_config` function appears to execute instead of the mock, leading to `FileNotFoundError` (as the dummy config file doesn't exist) and subsequent assertion failures checking `mock_load_config.await_count`.
*   **Status:** Tests are marked with `@pytest.mark.skip(reason="Known issue mocking load_config...")`. Resolving this likely requires deeper investigation into the interaction between `pytest-mock`, `anyio`, and async function patching, potentially needing more advanced mocking strategies.

### Other Skipped Tests

*   **Summary:** Approximately 89 other tests are currently skipped across the suite (as reported by `pytest -v -r s`).
*   **Reasons:** The reasons for these skips vary and are documented directly within the respective test files using the `@pytest.mark.skip(reason=...)` decorator. Common reasons include:
    *   Tests becoming outdated due to significant code refactoring (e.g., in `tests/models/test_gemini_agent_loop.py`, `tests/models/test_gemini_model_refactored.py`).
    *   Import errors (`ImportError: No module named 'mcp'`), suggesting potential issues with project structure or test environment setup for certain modules.
    *   Difficulties with specific patching scenarios (e.g., patching `os.path.expanduser`).
    *   Missing dependencies or environment-specific issues (`Required imports not available...`).
    *   Complex mocking interactions that require further debugging.
*   **Action:** Contributors looking to address these should refer to the specific skip reasons provided in the test files for detailed context.

## Application Bugs

*(No application-level bugs are formally documented here yet. This section will be updated as needed.)*
