#!/bin/bash
# Script to test coverage generation locally

set -e  # Exit on error

echo "Starting local test coverage generation..."

# Set up coverage directory
mkdir -p coverage_html

# Run pytest with coverage enabled and generate reports
echo "Running test suite with coverage enabled..."
python -m pytest \
  --cov=src.cli_code \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term \
  test_dir/test_file_tools.py test_dir/test_directory_tools.py test_dir/test_system_tools.py

echo "Coverage report generated in coverage.xml and coverage_html/"
echo "This is the format SonarCloud expects."

# Optional: Verify XML structure
echo "Checking XML coverage report structure..."
if [ -f "coverage.xml" ]; then
  echo "✅ coverage.xml file exists"
  # Extract source paths to verify they're correct
  python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); sources = root.find('sources'); print('Source paths in coverage.xml:'); [print(f'  {s.text}') for s in sources.findall('source')]"
  
  # Extract overall coverage percentage
  COVERAGE=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); line_rate = float(root.attrib['line-rate'])*100; print('{:.2f}%'.format(line_rate))")
  echo "Overall coverage percentage: $COVERAGE"
else
  echo "❌ coverage.xml file not generated!"
fi

echo "Local coverage testing completed." 