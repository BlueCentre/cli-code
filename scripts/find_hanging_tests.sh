#!/bin/bash

# Script to find hanging tests
echo "Running tests individually to find hanging tests..."

# Clean up cache files first
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -exec rm -f {} + 2>/dev/null || true

# Set timeout (in seconds) - use environment variable if set, otherwise default to 15 seconds
TIMEOUT=${TEST_TIMEOUT:-15}
echo "Using timeout value of $TIMEOUT seconds (set TEST_TIMEOUT env var to change)"

# Function to run a single test with timeout
run_test_with_timeout() {
    TEST_FILE=$1
    echo "Testing: $TEST_FILE"
    if timeout $TIMEOUT python -m pytest "$TEST_FILE" -v; then
        echo "✅ $TEST_FILE completed successfully"
        return 0
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "❌ $TEST_FILE TIMEOUT - Test is hanging!"
        else
            echo "❌ $TEST_FILE failed with exit code $EXIT_CODE"
            # Print more details about the failure
            python -m pytest "$TEST_FILE" -v --no-header --showlocals 2>&1 | tail -20
        fi
        return $EXIT_CODE
    fi
    echo "----------------------------------------"
}

# Automatically find all test files
echo "Finding test files..."
TEST_FILES=$(find test_dir -name "test_*.py" -type f)

# Check if we found any test files
if [ -z "$TEST_FILES" ]; then
    echo "Error: No test files found in test_dir/"
    exit 1
fi

# Count the files found
FILE_COUNT=$(echo "$TEST_FILES" | wc -l)
echo "Found $FILE_COUNT test files to check"
echo "----------------------------------------"

# Run each test file individually
FAILED_TESTS=()
HANGING_TESTS=()

for TEST_FILE in $TEST_FILES; do
    run_test_with_timeout "$TEST_FILE"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 124 ]; then
        HANGING_TESTS+=("$TEST_FILE")
    elif [ $EXIT_CODE -ne 0 ]; then
        FAILED_TESTS+=("$TEST_FILE")
    fi
    echo "----------------------------------------"
done

# Print summary at the end
echo "Test scan complete."
echo "----------------------------------------"
echo "Summary:"
echo "Total test files: $FILE_COUNT"
echo "Failed tests: ${#FAILED_TESTS[@]}"
echo "Hanging tests: ${#HANGING_TESTS[@]}"

if [ ${#HANGING_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Hanging tests:"
    for TEST in "${HANGING_TESTS[@]}"; do
        echo "- $TEST"
    done
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for TEST in "${FAILED_TESTS[@]}"; do
        echo "- $TEST"
    done
fi

# Exit with error if any tests were hanging or failed
if [ ${#HANGING_TESTS[@]} -gt 0 ] || [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    exit 1
fi

echo "All tests passed successfully." 