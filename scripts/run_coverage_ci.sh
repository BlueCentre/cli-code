#!/bin/bash
# Script to generate coverage for CI pipeline with timeouts to prevent hanging

set -e  # Exit on error
set -x  # Print commands for debugging

echo "Starting coverage generation for CI..."

# Set up coverage directory
mkdir -p coverage_html

# Set environment variables for CI 
export CI_EXIT_ON_TEST_FAILURE=1  # Exit on test failures to ensure code quality
export CI_TEST_TIMEOUT=60  # Default timeout

# Special handling for GitHub Actions environment
if [ -n "$GITHUB_WORKSPACE" ]; then
  echo "Running in GitHub Actions environment"
  echo "GITHUB_WORKSPACE: $GITHUB_WORKSPACE"
  echo "Current directory: $(pwd)"
  echo "Directory contents:"
  ls -la
fi

# Define the test directory
TEST_DIR="tests"
echo "Using test directory: $TEST_DIR"

# Check if the test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory '$TEST_DIR' not found!"
    echo "Current directory: $(pwd)"
    echo "Available directories:"
    ls -la
    exit 1 # Fail the script if the main test directory is missing
fi

# Set timeout duration (in seconds) from environment variable or use default
CI_TIMEOUT=${CI_TEST_TIMEOUT:-60}
echo "Using test timeout of $CI_TIMEOUT seconds (set CI_TEST_TIMEOUT env var to change)"

# Set up logging
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="$LOG_DIR/test_summary_${TIMESTAMP}.log"
echo "Test run started at $(date)" > "$SUMMARY_LOG"
echo "Timeout value: $CI_TIMEOUT seconds" >> "$SUMMARY_LOG"
echo "Test directory: $TEST_DIR" >> "$SUMMARY_LOG"

# Function to handle test errors with better reporting
handle_test_error() {
  TEST_FILE=$1
  EXIT_CODE=$2
  LOG_FILE=$3
  
  echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
  if [ $EXIT_CODE -eq 124 ]; then
    echo "⚠️ WARNING: $TEST_FILE TIMED OUT (after $CI_TIMEOUT seconds)" | tee -a "$SUMMARY_LOG"
  else
    echo "⚠️ WARNING: $TEST_FILE FAILED with exit code $EXIT_CODE" | tee -a "$SUMMARY_LOG"
  fi
  
  # If we have a log file, show the last few lines
  if [ -f "$LOG_FILE" ]; then
    echo "Last 10 lines from log:" | tee -a "$SUMMARY_LOG"
    tail -10 "$LOG_FILE" | tee -a "$SUMMARY_LOG"
  fi
  echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
}

# Clean up any pycache files to avoid import conflicts
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# Run tests with coverage enabled
# We can run all tests together now that conflicts are resolved
echo "Running test suite with coverage enabled..."

python -m pytest \
  --cov=src/cli_code \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term \
  --timeout=$CI_TIMEOUT \
  "$TEST_DIR" # Run all tests within the tests directory

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
  if [ $EXIT_CODE -eq 124 ]; then
    echo "⚠️ WARNING: Pytest run TIMED OUT (after $CI_TIMEOUT seconds)" | tee -a "$SUMMARY_LOG"
    TIMED_OUT_TESTS=1
    FAILED_TESTS=0 # Treat timeout as a special case, not necessarily failed tests
  else
    echo "⚠️ WARNING: Pytest run FAILED with exit code $EXIT_CODE" | tee -a "$SUMMARY_LOG"
    FAILED_TESTS=1
    TIMED_OUT_TESTS=0
  fi
  echo "Check logs in $LOG_DIR for details." | tee -a "$SUMMARY_LOG"
  echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
else
  echo "✅ Pytest run completed successfully" | tee -a "$SUMMARY_LOG"
  FAILED_TESTS=0
  TIMED_OUT_TESTS=0
fi

# Final Summary
echo "=======================================" | tee -a "$SUMMARY_LOG"
echo "Test Run Summary:" | tee -a "$SUMMARY_LOG"
echo "- Failed Tests: $FAILED_TESTS" | tee -a "$SUMMARY_LOG"
echo "- Timed Out Tests: $TIMED_OUT_TESTS" | tee -a "$SUMMARY_LOG"
echo "Test run finished at $(date)" | tee -a "$SUMMARY_LOG"
echo "=======================================" | tee -a "$SUMMARY_LOG"

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ] && [ "$CI_EXIT_ON_TEST_FAILURE" = "1" ]; then
  echo "Exiting with failure code due to test failures."
  exit 1
elif [ $TIMED_OUT_TESTS -gt 0 ]; then
  echo "Exiting with failure code due to test timeouts."
  exit 1 # Or a different code if desired for timeouts
fi

echo "Coverage generation script completed."
exit 0

# Old logic for running tests individually (Removed)
# # Function to run tests with a common pattern
# run_test_group() {
#   ...
# }
# 
# # Run gemini model tests individually
# run_test_group "gemini model" \
#   "tests/models/test_gemini.py" \
#   ...
# 
# # Run ollama model tests individually
# run_test_group "ollama model" \
#   "tests/models/test_ollama.py" \
#   ...
# 
# # Run config tests individually
# run_test_group "config" \
#   "tests/test_config.py" # Assuming config tests are at root of tests?
#   ...
# 
# # Run main tests individually
# run_test_group "main" \
#   "tests/test_main.py" \
#   ...
# 
# # Run remaining tests individually
# run_test_group "remaining" \
#   "tests/tools/test_task_complete_tool.py" \
#   "tests/tools/test_base_tool.py" \
#   "tests/test_utils.py" # Assuming utils test is at root of tests?
#   ...
