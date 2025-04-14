#!/bin/bash
# Script to test coverage generation locally with timeouts to prevent hanging

set -e  # Exit on error
set -x  # Print commands for debugging

echo "Starting local test coverage generation..."

# Set up coverage directory
mkdir -p coverage_html

# Set timeout duration (in seconds) from environment variable or use default
LOCAL_TIMEOUT=${LOCAL_TEST_TIMEOUT:-30}
echo "Using test timeout of $LOCAL_TIMEOUT seconds (set LOCAL_TEST_TIMEOUT env var to change)"

# Set up logging
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="$LOG_DIR/local_test_summary_${TIMESTAMP}.log"
echo "Local test run started at $(date)" > "$SUMMARY_LOG"
echo "Timeout value: $LOCAL_TIMEOUT seconds" >> "$SUMMARY_LOG"

# Function to handle test errors with better reporting
handle_test_error() {
  TEST_FILE=$1
  EXIT_CODE=$2
  LOG_FILE=$3
  
  echo "----------------------------------------" | tee -a "$SUMMARY_LOG"
  if [ $EXIT_CODE -eq 124 ]; then
    echo "⚠️ WARNING: $TEST_FILE TIMED OUT (after $LOCAL_TIMEOUT seconds)" | tee -a "$SUMMARY_LOG"
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

# Run tests in smaller batches with timeouts
echo "Running test suite with coverage enabled..."

# First, run the basic tools tests which are known to work
echo "Running tools tests (known to work well)..."
python -m pytest \
  --cov=src.cli_code \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term \
  --timeout=$LOCAL_TIMEOUT \
  test_dir/test_file_tools.py \
  test_dir/test_system_tools.py \
  test_dir/test_directory_tools.py \
  test_dir/improved/test_quality_tools.py \
  test_dir/improved/test_summarizer_tool.py \
  test_dir/improved/test_tree_tool.py

# Now run the model tests separately
echo "Running model tests..."
python -m pytest \
  --cov=src.cli_code \
  --cov-append \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term \
  --timeout=$LOCAL_TIMEOUT \
  test_dir/test_models_base.py \
  test_dir/test_model_basic.py \
  test_dir/test_model_integration.py

# Track failures
FAILED_TESTS=0
TIMED_OUT_TESTS=0

# Run gemini model tests individually
for test_file in \
  test_dir/test_gemini_model.py \
  test_dir/test_gemini_model_advanced.py \
  test_dir/test_gemini_model_coverage.py \
  test_dir/test_gemini_model_error_handling.py; do
  echo "Running $test_file with timeout $LOCAL_TIMEOUT seconds..."
  LOG_FILE="$LOG_DIR/local_$(basename $test_file).log"
  
  # Run test with timeout and capture output
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=$LOCAL_TIMEOUT \
    "$test_file" > "$LOG_FILE" 2>&1
  
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    if [ $EXIT_CODE -eq 124 ]; then
      TIMED_OUT_TESTS=$((TIMED_OUT_TESTS + 1))
    else
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    handle_test_error "$test_file" "$EXIT_CODE" "$LOG_FILE"
  else
    echo "✅ $test_file completed successfully" | tee -a "$SUMMARY_LOG"
  fi
done

# Run ollama model tests individually
for test_file in \
  test_dir/test_ollama_model.py \
  test_dir/test_ollama_model_advanced.py \
  test_dir/test_ollama_model_coverage.py \
  test_dir/test_ollama_model_context.py \
  test_dir/test_ollama_model_error_handling.py; do
  echo "Running $test_file with timeout $LOCAL_TIMEOUT seconds..."
  LOG_FILE="$LOG_DIR/local_$(basename $test_file).log"
  
  # Run test with timeout and capture output
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=$LOCAL_TIMEOUT \
    "$test_file" > "$LOG_FILE" 2>&1
  
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    if [ $EXIT_CODE -eq 124 ]; then
      TIMED_OUT_TESTS=$((TIMED_OUT_TESTS + 1))
    else
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    handle_test_error "$test_file" "$EXIT_CODE" "$LOG_FILE"
  else
    echo "✅ $test_file completed successfully" | tee -a "$SUMMARY_LOG"
  fi
done

# Run config tests individually
for test_file in \
  test_dir/test_config.py \
  test_dir/test_config_comprehensive.py \
  test_dir/test_config_edge_cases.py \
  test_dir/test_config_missing_methods.py; do
  echo "Running $test_file with timeout $LOCAL_TIMEOUT seconds..."
  LOG_FILE="$LOG_DIR/local_$(basename $test_file).log"
  
  # Run test with timeout and capture output
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=$LOCAL_TIMEOUT \
    "$test_file" > "$LOG_FILE" 2>&1
  
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    if [ $EXIT_CODE -eq 124 ]; then
      TIMED_OUT_TESTS=$((TIMED_OUT_TESTS + 1))
    else
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    handle_test_error "$test_file" "$EXIT_CODE" "$LOG_FILE"
  else
    echo "✅ $test_file completed successfully" | tee -a "$SUMMARY_LOG"
  fi
done

# Run main tests individually
for test_file in \
  test_dir/test_main.py \
  test_dir/test_main_comprehensive.py \
  test_dir/test_main_edge_cases.py \
  test_dir/test_main_improved.py; do
  echo "Running $test_file with timeout $LOCAL_TIMEOUT seconds..."
  LOG_FILE="$LOG_DIR/local_$(basename $test_file).log"
  
  # Run test with timeout and capture output
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=$LOCAL_TIMEOUT \
    "$test_file" > "$LOG_FILE" 2>&1
  
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    if [ $EXIT_CODE -eq 124 ]; then
      TIMED_OUT_TESTS=$((TIMED_OUT_TESTS + 1))
    else
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    handle_test_error "$test_file" "$EXIT_CODE" "$LOG_FILE"
  else
    echo "✅ $test_file completed successfully" | tee -a "$SUMMARY_LOG"
  fi
done

# Run remaining tests individually
for test_file in \
  test_dir/test_task_complete_tool.py \
  test_dir/test_tools_base.py \
  test_dir/test_tools_init_coverage.py \
  test_dir/test_utils.py \
  test_dir/test_utils_comprehensive.py \
  test_dir/test_test_runner_tool.py \
  test_dir/test_basic_functions.py \
  test_dir/test_tools_basic.py \
  test_dir/test_tree_tool_edge_cases.py; do
  echo "Running $test_file with timeout $LOCAL_TIMEOUT seconds..."
  LOG_FILE="$LOG_DIR/local_$(basename $test_file).log"
  
  # Run test with timeout and capture output
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=$LOCAL_TIMEOUT \
    "$test_file" > "$LOG_FILE" 2>&1
  
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    if [ $EXIT_CODE -eq 124 ]; then
      TIMED_OUT_TESTS=$((TIMED_OUT_TESTS + 1))
    else
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    handle_test_error "$test_file" "$EXIT_CODE" "$LOG_FILE"
  else
    echo "✅ $test_file completed successfully" | tee -a "$SUMMARY_LOG"
  fi
done

# Generate a final coverage report
python -m pytest \
  --cov=src.cli_code \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term

echo "Coverage report generated in coverage.xml and coverage_html/"
echo "This is the format SonarCloud expects."

# Print summary of test results
echo ""
echo "Test Summary:" | tee -a "$SUMMARY_LOG"
echo "-------------" | tee -a "$SUMMARY_LOG"
echo "Failed tests: $FAILED_TESTS" | tee -a "$SUMMARY_LOG"
echo "Timed out tests: $TIMED_OUT_TESTS" | tee -a "$SUMMARY_LOG"
echo "Log files available in: $LOG_DIR" | tee -a "$SUMMARY_LOG"
echo "Summary log: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"

# Optional: Verify XML structure
echo "Checking XML coverage report structure..."
if [ -f "coverage.xml" ]; then
  echo "✅ coverage.xml file exists" | tee -a "$SUMMARY_LOG"
  # Extract source paths to verify they're correct
  python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); sources = root.find('sources'); print('Source paths in coverage.xml:'); [print(f'  {s.text}') for s in sources.findall('source')]" | tee -a "$SUMMARY_LOG"
  
  # Extract overall coverage percentage
  COVERAGE=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); line_rate = float(root.attrib['line-rate'])*100; print('{:.2f}%'.format(line_rate))")
  echo "Overall coverage percentage: $COVERAGE" | tee -a "$SUMMARY_LOG"
else
  echo "❌ coverage.xml file not generated!" | tee -a "$SUMMARY_LOG"
fi

echo "Local coverage testing completed."

# Print a warning if there were any failing or hanging tests
if [ $FAILED_TESTS -gt 0 -o $TIMED_OUT_TESTS -gt 0 ]; then
  echo "⚠️ Warning: Test run completed with $FAILED_TESTS failing tests and $TIMED_OUT_TESTS timed out tests" | tee -a "$SUMMARY_LOG"
  echo "Check the logs in $LOG_DIR for details"
fi 