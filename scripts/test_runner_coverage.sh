#!/bin/bash
# test_runner_coverage.sh - Run detailed coverage for test_runner.py

# Set error handling
set -e

echo "Running detailed coverage for test_runner.py..."

# Clear existing coverage data
coverage erase

# Run all test_runner.py tests with coverage
echo "=== Running tests/tools/test_test_runner_tool.py ==="
python -m pytest tests/tools/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner --cov-report=term

echo "=== Running test_dir/test_test_runner_tool.py ==="
python -m pytest test_dir/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner --cov-append --cov-report=term

# Combine coverage data and generate reports
echo "=== Generating combined coverage report ==="
coverage combine
coverage report --include="src/cli_code/tools/test_runner.py"
coverage html --include="src/cli_code/tools/test_runner.py"

echo "Test runner coverage complete!" 