#!/bin/bash
# focused_test_runner_coverage.sh - Run targeted coverage for test_runner.py

set -e

echo "Running targeted test_runner.py coverage..."

# Clean any existing coverage data
coverage erase

# Run only the tests from tests/tools with direct imports
echo "Running tests with direct imports..."
python -m pytest tests/tools/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner

# Generate HTML report
coverage html

echo "Coverage complete. Check coverage_html/index.html for detailed report."
