#!/bin/bash

# Script to find hanging tests
echo "Running tests individually to find hanging tests..."

# Clean up cache files first
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -exec rm -f {} + 2>/dev/null || true

# Set timeout (in seconds)
TIMEOUT=15

# Function to run a single test with timeout
run_test_with_timeout() {
    TEST_FILE=$1
    echo "Testing: $TEST_FILE"
    if timeout $TIMEOUT python -m pytest "$TEST_FILE" -v; then
        echo "✅ $TEST_FILE completed successfully"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "❌ $TEST_FILE TIMEOUT - Test is hanging!"
        else
            echo "❌ $TEST_FILE failed with exit code $EXIT_CODE"
        fi
    fi
    echo "----------------------------------------"
}

# Test files to check - base list from most critical files
TEST_FILES=(
    "test_dir/test_file_tools.py"
    "test_dir/test_system_tools.py"
    "test_dir/test_directory_tools.py"
    "test_dir/improved/test_quality_tools.py"
    "test_dir/improved/test_summarizer_tool.py"
    "test_dir/improved/test_tree_tool.py"
    "test_dir/test_models_base.py"
    "test_dir/test_model_basic.py"
    "test_dir/test_model_integration.py"
    "test_dir/test_gemini_model.py"
    "test_dir/test_gemini_model_advanced.py"
    "test_dir/test_gemini_model_coverage.py"
    "test_dir/test_gemini_model_error_handling.py"
    "test_dir/test_ollama_model.py"
    "test_dir/test_ollama_model_advanced.py"
    "test_dir/test_ollama_model_coverage.py"
    "test_dir/test_ollama_model_context.py"
    "test_dir/test_ollama_model_error_handling.py"
    "test_dir/test_config.py"
    "test_dir/test_config_comprehensive.py"
    "test_dir/test_config_edge_cases.py"
    "test_dir/test_config_missing_methods.py"
    "test_dir/test_main.py"
    "test_dir/test_main_comprehensive.py"
    "test_dir/test_main_edge_cases.py"
    "test_dir/test_main_improved.py"
    "test_dir/test_task_complete_tool.py"
    "test_dir/test_tools_base.py"
    "test_dir/test_tools_init_coverage.py"
    "test_dir/test_utils.py"
    "test_dir/test_utils_comprehensive.py"
    "test_dir/test_test_runner_tool.py"
    "test_dir/test_basic_functions.py"
    "test_dir/test_tools_basic.py"
    "test_dir/test_tree_tool_edge_cases.py"
)

# Run each test file individually
for TEST_FILE in "${TEST_FILES[@]}"; do
    run_test_with_timeout "$TEST_FILE"
done

echo "Test scan complete. Check output for any hanging tests." 