#!/bin/bash
# Script to generate coverage for CI pipeline

set -e  # Exit on error

echo "Starting coverage generation for CI..."

# Set up coverage directory
mkdir -p coverage_html

# Run pytest with coverage enabled and generate reports
echo "Running test suite with coverage enabled..."
python -m pytest \
  --cov=src.cli_code \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term \
  test_dir/test_file_tools.py test_dir/test_directory_tools.py test_dir/test_system_tools.py \
  test_dir/improved/test_quality_tools.py test_dir/improved/test_summarizer_tool.py test_dir/improved/test_tree_tool.py

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
