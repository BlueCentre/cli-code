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
# Find all test files in the test_dir directory
TEST_FILES=$(find test_dir -name "test_*.py" -print)

# Run each test file individually
for TEST_FILE in "${TEST_FILES[@]}"; do
    run_test_with_timeout "$TEST_FILE"
done

echo "Test scan complete. Check output for any hanging tests." 