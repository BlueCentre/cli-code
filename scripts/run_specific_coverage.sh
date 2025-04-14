#!/bin/bash
# run_specific_coverage.sh - Run comprehensive coverage for a specific module

# Set error handling
set -e

MODULE=$1
MODULE_PATH=$2

if [ -z "$MODULE" ] || [ -z "$MODULE_PATH" ]; then
    echo "Usage: $0 module_name module_path"
    echo "Example: $0 test_runner src/cli_code/tools/test_runner.py"
    exit 1
fi

echo "Running comprehensive coverage for $MODULE..."

# Clear existing coverage data
coverage erase

# Find all test files that might test this module
TEST_FILES=$(find tests -name "test_*.py" -type f -exec grep -l "$MODULE" {} \;)

if [ -z "$TEST_FILES" ]; then
    echo "No test files found for $MODULE"
    exit 1
fi

echo "Found test files:"
echo "$TEST_FILES"

# Run each test file
for TEST_FILE in $TEST_FILES; do
    echo "=== Running $TEST_FILE ==="
    MODULE_DOT_PATH=$(echo "$MODULE_PATH" | sed 's/\//./' | sed 's/\.py$//')
    if [[ "$MODULE_DOT_PATH" != src.* ]]; then
        MODULE_DOT_PATH="src.$MODULE_DOT_PATH"
    fi
    python -m pytest "$TEST_FILE" -v --cov="$MODULE_DOT_PATH" --cov-append
done

# Generate detailed report
echo "=== Generating coverage report ==="
coverage report --include="$MODULE_PATH"
coverage html --include="$MODULE_PATH"

echo "Coverage analysis complete for $MODULE" 