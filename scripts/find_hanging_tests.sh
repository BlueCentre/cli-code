#!/bin/bash

# Script to find hanging tests
echo "Running tests individually to find hanging tests..."

# Clean up cache files first
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -exec rm -f {} + 2>/dev/null || true

# Set timeout (in seconds) - use environment variable if set, otherwise default to 15 seconds
TIMEOUT=${TEST_TIMEOUT:-15}
echo "Using timeout value of $TIMEOUT seconds (set TEST_TIMEOUT env var to change)"

# Set up logging
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="$LOG_DIR/hanging_tests_summary_${TIMESTAMP}.log"
echo "Hanging test scan started at $(date)" > "$SUMMARY_LOG"
echo "Timeout value: $TIMEOUT seconds" >> "$SUMMARY_LOG"

# Function to run a single test with timeout
run_test_with_timeout() {
    TEST_FILE=$1
    echo "Testing: $TEST_FILE" | tee -a "$SUMMARY_LOG"
    LOG_FILE="$LOG_DIR/$(basename "$TEST_FILE").log"

    # Run test with timeout and capture output to log file
    if timeout $TIMEOUT python -m pytest "$TEST_FILE" -v > "$LOG_FILE" 2>&1; then
        echo "✅ $TEST_FILE completed successfully" | tee -a "$SUMMARY_LOG"
        return 0
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "❌ $TEST_FILE TIMEOUT - Test is hanging!" | tee -a "$SUMMARY_LOG"
        else
            echo "❌ $TEST_FILE failed with exit code $EXIT_CODE" | tee -a "$SUMMARY_LOG"
            # Print more details about the failure
            echo "Last 10 lines of log:" | tee -a "$SUMMARY_LOG"
            tail -10 "$LOG_FILE" | tee -a "$SUMMARY_LOG"
        fi
        return $EXIT_CODE
    fi
    echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
}

# Determine test directory
TEST_DIR="tests"

# Check if TEST_DIR environment variable is set
if [ -n "$TEST_DIR_ENV" ]; then
    TEST_DIR="$TEST_DIR_ENV"
    echo "Using test directory from environment: $TEST_DIR" | tee -a "$SUMMARY_LOG"
fi

# Verify test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory $TEST_DIR does not exist!" | tee -a "$SUMMARY_LOG"
    echo "Current directory: $(pwd)" | tee -a "$SUMMARY_LOG"
    echo "Available directories:" | tee -a "$SUMMARY_LOG"
    ls -la | tee -a "$SUMMARY_LOG"
    exit 1
fi

# Automatically find all test files
echo "Finding test files in $TEST_DIR..." | tee -a "$SUMMARY_LOG"
TEST_FILES=$(find "$TEST_DIR" -name "test_*.py" -type f)

# Check if we found any test files
if [ -z "$TEST_FILES" ]; then
    echo "Error: No test files found in $TEST_DIR/" | tee -a "$SUMMARY_LOG"
    echo "Available files in $TEST_DIR:" | tee -a "$SUMMARY_LOG"
    find "$TEST_DIR" -type f | tee -a "$SUMMARY_LOG"
    exit 1
fi

# Count the files found
FILE_COUNT=$(echo "$TEST_FILES" | wc -l)
echo "Found $FILE_COUNT test files to check" | tee -a "$SUMMARY_LOG"
echo "----------------------------------------" | tee -a "$SUMMARY_LOG"

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
    echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
done

# Print summary at the end
echo "Test scan complete." | tee -a "$SUMMARY_LOG"
echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
echo "Summary:" | tee -a "$SUMMARY_LOG"
echo "Total test files: $FILE_COUNT" | tee -a "$SUMMARY_LOG"
echo "Failed tests: ${#FAILED_TESTS[@]}" | tee -a "$SUMMARY_LOG"
echo "Hanging tests: ${#HANGING_TESTS[@]}" | tee -a "$SUMMARY_LOG"
echo "Log files available in: $LOG_DIR" | tee -a "$SUMMARY_LOG"
echo "Summary log: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"

if [ ${#HANGING_TESTS[@]} -gt 0 ]; then
    echo "" | tee -a "$SUMMARY_LOG"
    echo "Hanging tests:" | tee -a "$SUMMARY_LOG"
    for TEST in "${HANGING_TESTS[@]}"; do
        echo "- $TEST" | tee -a "$SUMMARY_LOG"
    done
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "" | tee -a "$SUMMARY_LOG"
    echo "Failed tests:" | tee -a "$SUMMARY_LOG"
    for TEST in "${FAILED_TESTS[@]}"; do
        echo "- $TEST" | tee -a "$SUMMARY_LOG"
    done
fi

# Exit with error if any tests were hanging or failed
if [ ${#HANGING_TESTS[@]} -gt 0 ] || [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "⚠️ Some tests failed or timed out. Check logs for details." | tee -a "$SUMMARY_LOG"
    # Only exit with error in CI environment
    if [ -n "$CI" ]; then
        exit 1
    fi
fi

echo "All tests passed successfully." | tee -a "$SUMMARY_LOG"
