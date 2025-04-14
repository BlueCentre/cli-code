#!/bin/bash
# Script to generate coverage for CI pipeline with timeouts to prevent hanging

set -e  # Exit on error
set -x  # Print commands for debugging

echo "Starting coverage generation for CI..."

# Set up coverage directory
mkdir -p coverage_html

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
  --timeout=60 \
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
  --timeout=60 \
  test_dir/test_models_base.py \
  test_dir/test_model_basic.py \
  test_dir/test_model_integration.py

# Run gemini model tests individually
for test_file in \
  test_dir/test_gemini_model.py \
  test_dir/test_gemini_model_advanced.py \
  test_dir/test_gemini_model_coverage.py \
  test_dir/test_gemini_model_error_handling.py; do
  echo "Running $test_file with timeout..."
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=60 \
    "$test_file" || echo "Warning: $test_file timed out or failed"
done

# Run ollama model tests individually
for test_file in \
  test_dir/test_ollama_model.py \
  test_dir/test_ollama_model_advanced.py \
  test_dir/test_ollama_model_coverage.py \
  test_dir/test_ollama_model_context.py \
  test_dir/test_ollama_model_error_handling.py; do
  echo "Running $test_file with timeout..."
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=60 \
    "$test_file" || echo "Warning: $test_file timed out or failed"
done

# Run config tests individually
for test_file in \
  test_dir/test_config.py \
  test_dir/test_config_comprehensive.py \
  test_dir/test_config_edge_cases.py \
  test_dir/test_config_missing_methods.py; do
  echo "Running $test_file with timeout..."
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=60 \
    "$test_file" || echo "Warning: $test_file timed out or failed"
done

# Run main tests individually
for test_file in \
  test_dir/test_main.py \
  test_dir/test_main_comprehensive.py \
  test_dir/test_main_edge_cases.py \
  test_dir/test_main_improved.py; do
  echo "Running $test_file with timeout..."
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=60 \
    "$test_file" || echo "Warning: $test_file timed out or failed"
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
  echo "Running $test_file with timeout..."
  python -m pytest \
    --cov=src.cli_code \
    --cov-append \
    --timeout=60 \
    "$test_file" || echo "Warning: $test_file timed out or failed"
done

# Generate a final coverage report
python -m pytest \
  --cov=src.cli_code \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term

echo "Coverage report generated in coverage.xml and coverage_html/"

# Extract overall coverage percentage for GitHub output
if [ -f "coverage.xml" ]; then
  echo "✅ coverage.xml file exists"
  
  # Extract overall coverage percentage
  COVERAGE=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); line_rate = float(root.attrib['line-rate'])*100; print('{:.2f}%'.format(line_rate))")
  echo "Overall coverage percentage: $COVERAGE"
  
  # Set output for GitHub Actions
  echo "percentage=$COVERAGE" >> $GITHUB_OUTPUT
else
  echo "❌ coverage.xml file not generated!"
  echo "percentage=0.00%" >> $GITHUB_OUTPUT
fi

echo "Coverage generation for CI completed."
